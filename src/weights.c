/* weights.c — Load MLP weights and run inference on MNIST digits.
 * C89-compatible.
 */

#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "weights.h"

/* Magic number: "NMST" in little-endian. */
#define WEIGHTS_MAGIC 0x4E4D5354U

/* MNIST dataset statistics (computed from training set). */
#define MNIST_MEAN 0.1307
#define MNIST_STD  0.3081

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

int load_weights(const char *path, model_t *model)
{
    FILE *f;
    uint32_t magic;
    int32_t num_layers;
    int32_t in_dim[2];
    int32_t out_dim[2];
    size_t n, expected;
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

    /* Read floats directly (assumes binary float compatibility) */
    expected = (size_t)(MODEL_HIDDEN * MODEL_INPUT);
    n = fread(model->w0, sizeof(float), expected, f);
    if (n != expected) {
        fprintf(stderr, "Error: truncated weights layer 0\n");
        fclose(f);
        return -1;
    }

    n = fread(model->b0, sizeof(float), MODEL_HIDDEN, f);
    if (n != MODEL_HIDDEN) {
        fprintf(stderr, "Error: truncated bias layer 0\n");
        fclose(f);
        return -1;
    }

    expected = (size_t)(MODEL_OUTPUT * MODEL_HIDDEN);
    n = fread(model->w1, sizeof(float), expected, f);
    if (n != expected) {
        fprintf(stderr, "Error: truncated weights layer 1\n");
        fclose(f);
        return -1;
    }

    n = fread(model->b1, sizeof(float), MODEL_OUTPUT, f);
    if (n != MODEL_OUTPUT) {
        fprintf(stderr, "Error: truncated bias layer 1\n");
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

void normalize_input(float input[MODEL_INPUT])
{
    int i;
    for (i = 0; i < MODEL_INPUT; i++) {
        input[i] = (float)(((double)input[i] / 255.0 - MNIST_MEAN) / MNIST_STD);
    }
}

int model_predict(const model_t *model, const float input[MODEL_INPUT])
{
    float hidden[MODEL_HIDDEN];
    float logits[MODEL_OUTPUT];
    int i, j, best;
    float best_val;

    if (!model)
        return -1;

    /* Layer 0: linear(784, 64) + ReLU */
    for (i = 0; i < MODEL_HIDDEN; i++) {
        float sum = model->b0[i];
        for (j = 0; j < MODEL_INPUT; j++) {
            sum += input[j] * model->w0[i][j];
        }
        hidden[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    /* Layer 1: linear(64, 10) */
    for (i = 0; i < MODEL_OUTPUT; i++) {
        float sum = model->b1[i];
        for (j = 0; j < MODEL_HIDDEN; j++) {
            sum += hidden[j] * model->w1[i][j];
        }
        logits[i] = sum;
    }

    /* Argmax */
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
