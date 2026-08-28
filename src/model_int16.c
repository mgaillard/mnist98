/* model_int16.c — Load quantized MLP weights and run integer inference on MNIST digits.
 * C89-compatible.
 *
 * The forward pass uses integer arithmetic only (mirrors QuantizedMLP in
 * training/model_quant.py):
 *
 *   input   : int16 per pixel (pixel 0 → -4939, pixel 255 → 32767)
 *   layer 0 : int16 x int16 + int32 bias → int32, ReLU, >> 16 → int16
 *   layer 1 : int16 x int16 + int32 bias → int32
 *   output  : argmax over int32 logits (scale-invariant)
 *
 * The matmuls use the int16_dot() helper (see include/int16_dot.h):
 * on 32-bit x86 builds compiled with MMX support it is the assembly
 * kernel (src/int16_dot.S, pmaddwd), everywhere else the pure C
 * fallback defined below.
 */

#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "model_int16.h"
#include "int16_dot.h"

/*
 * Assembly int16 dot product support. Requires 32-bit mode (__i386__) and
 * MMX (__MMX__, via -mmmx or a -march that includes MMX). In 64-bit mode
 * __MMX__ is predefined (MMX is in the x86-64 baseline ISA) but the
 * assembly kernel only exists for 32-bit targets. In 64-bit mode, the
 * compiler will auto-vectorize the scalar loop with much better
 * SSE/AVX performance than MMX.
 */
#if defined(__i386__) && defined(__MMX__)
#define MODEL_INT16_USE_ASM_DOT 1
#endif

#ifndef MODEL_INT16_USE_ASM_DOT
/*
 * Pure C fallback for int16_dot(), used in builds without the 32-bit MMX
 * assembly kernel. int32 accumulation, matching the assembly version.
 */
int32_t int16_dot(const int16_t *a, const int16_t *b, int n)
{
    int32_t sum = 0;
    int i;
    for (i = 0; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}
#endif

/* Magic number: "NMST" in little-endian (same header as the fp32 format). */
#define WEIGHTS_MAGIC 0x4E4D5354U

/* Half of the 8-bit pixel step, for rounding p * QPIX_SPAN / 255. */
#define PIXEL_ROUND 127

static uint32_t read_u32_le_stream(FILE *f, int *ok)
{
    uint8_t buf[4];
    if (fread(buf, 1, 4, f) != 4) {
        *ok = 0;
        return 0;
    }
    *ok = 1;
    return (uint32_t)buf[0]
         | ((uint32_t)buf[1] << 8)
         | ((uint32_t)buf[2] << 16)
         | ((uint32_t)buf[3] << 24);
}

/* Read `count` little-endian int16 values from the stream into dst. */
static int load_i16_block(FILE *f, int16_t *dst, size_t count)
{
    uint8_t *buf;
    size_t bytes;
    size_t i;

    bytes = count * 2;
    buf = (uint8_t *)malloc(bytes);
    if (!buf) {
        return -1;
    }

    if (fread(buf, 1, bytes, f) != bytes) {
        free(buf);
        return -1;
    }

    for (i = 0; i < count; i++) {
        dst[i] = (int16_t)((uint16_t)buf[2 * i]
                         | ((uint16_t)buf[2 * i + 1] << 8));
    }

    free(buf);
    return 0;
}

/* Read `count` little-endian int32 values from the stream into dst. */
static int load_i32_block(FILE *f, int32_t *dst, size_t count)
{
    uint8_t *buf;
    size_t bytes;
    size_t i;

    bytes = count * 4;
    buf = (uint8_t *)malloc(bytes);
    if (!buf) {
        return -1;
    }

    if (fread(buf, 1, bytes, f) != bytes) {
        free(buf);
        return -1;
    }

    for (i = 0; i < count; i++) {
        dst[i] = (int32_t)((uint32_t)buf[4 * i]
                         | ((uint32_t)buf[4 * i + 1] << 8)
                         | ((uint32_t)buf[4 * i + 2] << 16)
                         | ((uint32_t)buf[4 * i + 3] << 24));
    }

    free(buf);
    return 0;
}

int model_int16_load_weights(const char *path, model_int16_t *model)
{
    FILE *f;
    uint32_t magic;
    int32_t num_layers;
    int32_t in_dim[2];
    int32_t out_dim[2];
    int ok;

    if (!model) {
        fprintf(stderr, "Error: NULL model pointer\n");
        return -1;
    }

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open '%s'\n", path);
        return -1;
    }

    /* Read and validate magic number. */
    magic = read_u32_le_stream(f, &ok);
    if (!ok || magic != WEIGHTS_MAGIC) {
        fprintf(stderr, "Error: bad magic number (expected 0x%08X, got 0x%08X)\n",
                WEIGHTS_MAGIC, (unsigned)magic);
        fclose(f);
        return -1;
    }

    /* Read num_layers. */
    num_layers = (int32_t)read_u32_le_stream(f, &ok);
    if (!ok || num_layers != 2) {
        fprintf(stderr, "Error: expected 2 layers, got %d\n", (int)num_layers);
        fclose(f);
        return -1;
    }

    /* Read dimensions safely */
    in_dim[0]  = (int32_t)read_u32_le_stream(f, &ok);
    in_dim[1]  = (int32_t)read_u32_le_stream(f, &ok);
    out_dim[0] = (int32_t)read_u32_le_stream(f, &ok);
    out_dim[1] = (int32_t)read_u32_le_stream(f, &ok);

    if (!ok) {
        fprintf(stderr, "Error: truncated layer dimensions\n");
        fclose(f);
        return -1;
    }

    /* Validate architecture */
    if (in_dim[0] != MODEL_INPUT || in_dim[1] != MODEL_HIDDEN ||
        out_dim[0] != MODEL_HIDDEN || out_dim[1] != MODEL_OUTPUT) {
        fprintf(stderr, "Error: model dimension mismatch\n");
        fclose(f);
        return -1;
    }

    /* Read int16 weights and int32 biases (little-endian). */
    if (load_i16_block(f, (int16_t *)model->w0, (size_t)MODEL_HIDDEN * MODEL_INPUT) != 0) {
        fprintf(stderr, "Error: truncated int16 weights layer 0\n");
        fclose(f);
        return -1;
    }

    if (load_i32_block(f, model->b0, (size_t)MODEL_HIDDEN) != 0) {
        fprintf(stderr, "Error: truncated int32 bias layer 0\n");
        fclose(f);
        return -1;
    }

    if (load_i16_block(f, (int16_t *)model->w1, (size_t)MODEL_OUTPUT * MODEL_HIDDEN) != 0) {
        fprintf(stderr, "Error: truncated int16 weights layer 1\n");
        fclose(f);
        return -1;
    }

    if (load_i32_block(f, model->b1, (size_t)MODEL_OUTPUT) != 0) {
        fprintf(stderr, "Error: truncated int32 bias layer 1\n");
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

void model_int16_quantize_input(const int pixels_raw[MODEL_INPUT], int16_t input[MODEL_INPUT])
{
    int i;
    int p;
    for (i = 0; i < MODEL_INPUT; i++) {
        p = pixels_raw[i];

        /* Defensive clamp: raw pixels are expected in [0, 255]. */
        if (p < 0)
            p = 0;
        if (p > 255)
            p = 255;

        /* Linear map (rounded): pixel 0 → -4939, pixel 255 → 32767. */
        input[i] = (int16_t)((p * QPIX_SPAN + PIXEL_ROUND) / 255 + QPIX_MIN);
    }
}

int model_int16_predict(const model_int16_t *model, const int16_t input[MODEL_INPUT])
{
    int32_t hidden[MODEL_HIDDEN];
    int16_t hidden_q[MODEL_HIDDEN];
    int32_t logits[MODEL_OUTPUT];
    int32_t best_val;
    int i, best;

    if (!model)
        return -1;

    /* Layer 0: linear(784, 64) + ReLU, int32 accumulation. */
    for (i = 0; i < MODEL_HIDDEN; i++) {
        int32_t sum = model->b0[i] + int16_dot(input, model->w0[i], MODEL_INPUT);
        hidden[i] = (sum > 0) ? sum : 0;
    }

    /* Rescale to int16 range for layer 1. */
    for (i = 0; i < MODEL_HIDDEN; i++) {
        hidden_q[i] = (int16_t)(hidden[i] >> 16);
    }

    /* Layer 1: linear(64, 10), int32 accumulation. */
    for (i = 0; i < MODEL_OUTPUT; i++) {
        logits[i] = model->b1[i] + int16_dot(hidden_q, model->w1[i], MODEL_HIDDEN);
    }

    /* Argmax (scale-invariant, no dequantization needed). */
    best_val = logits[0];
    best = 0;
    for (i = 1; i < MODEL_OUTPUT; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }

    return best;
}
