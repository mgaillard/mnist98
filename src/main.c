/* main.c — MNIST digit classification from a BMP image.
 * C89-compatible.
 *
 * Two models are available:
 *   - fp32 model (default): float weights, float forward pass
 *   - quantized int16 model (--quant): integer-only forward pass
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bmp.h"
#include "model_fp32.h"
#include "model_int16.h"

/* Predict function pointer so the benchmark can time either model. */
typedef int (*predict_fn)(const void *model, const void *input);

static int predict_fp32(const void *model, const void *input)
{
    return model_fp32_predict((const model_fp32_t *)model, (const float *)input);
}

static int predict_int16(const void *model, const void *input)
{
    return model_int16_predict((const model_int16_t *)model, (const int16_t *)input);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s <image.bmp> [--quant] [--weights weights.bin] [--benchmark N]\n"
            "  --quant       use the quantized int16 model (default: fp32 model)\n"
            "  --weights     path to the weights file\n"
            "                (default: weights.bin, or weights_quant.bin with --quant)\n"
            "  --benchmark N run the forward pass N extra times and report throughput\n",
            prog);
    exit(EXIT_FAILURE);
}

static void benchmark(predict_fn predict, const void *model, const void *input, int count)
{
    clock_t t_start, t_end;
    double elapsed_sec;
    double avg_ms;
    double throughput;
    int i;

    /* Warm-up run */
    predict(model, input);

    t_start = clock();
    for (i = 0; i < count; i++) {
        predict(model, input);
    }
    t_end = clock();

    elapsed_sec = (double)(t_end - t_start) / (double)CLOCKS_PER_SEC;

    if (elapsed_sec <= 0.0) {
        printf("Benchmark (%d runs): Execution time too short to measure accurately with clock(). Try a larger N.\n", count);
        return;
    }

    avg_ms = (elapsed_sec / (double)count) * 1000.0;
    throughput = (double)count / elapsed_sec;

    printf("Benchmark (%d runs): avg %.3f ms/img, %.1f img/s\n",
           count, avg_ms, throughput);
}

int main(int argc, char *argv[])
{
    const char *img_path;
    const char *weights_path;
    const void *model;
    const void *input;
    predict_fn predict;
    int pixels_raw[MODEL_INPUT];
    float input_f32[MODEL_INPUT];
    int16_t input_q[MODEL_INPUT];
    int pixels[28][28];
    int quant;
    int benchmark_count;
    int predicted;
    int i, x, y;

    /* Defaults. */
    weights_path = NULL;
    quant = 0;
    benchmark_count = 0;

    if (argc < 2)
        usage(argv[0]);

    img_path = argv[1];

    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--weights") == 0) {
            if (i + 1 >= argc)
                usage(argv[0]);
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            if (i + 1 >= argc)
                usage(argv[0]);
            benchmark_count = atoi(argv[++i]);
            if (benchmark_count <= 0) {
                fprintf(stderr, "Error: benchmark count must be positive\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--quant") == 0) {
            quant = 1;
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            usage(argv[0]);
        }
    }

    if (!weights_path)
        weights_path = quant ? "weights_quant.bin" : "weights.bin";

    /* Load the BMP image. */
    if (load_bmp(img_path, pixels) != 0) {
        return EXIT_FAILURE;
    }

    /* Flatten 28x28 to 784 (row-major). */
    for (y = 0; y < 28; y++) {
        for (x = 0; x < 28; x++) {
            pixels_raw[y * 28 + x] = pixels[y][x];
        }
    }

    /* Allocate model on HEAP to avoid stack overflow (~100-200 KB) */
    if (quant) {
        model_int16_t *model_q = (model_int16_t *)malloc(sizeof(model_int16_t));
        if (!model_q) {
            fprintf(stderr, "Error: out of memory allocating model structure\n");
            return EXIT_FAILURE;
        }

        /* Convert raw pixels to the int16 encoding. */
        model_int16_quantize_input(pixels_raw, input_q);

        /* Load the quantized weights. */
        if (model_int16_load_weights(weights_path, model_q) != 0) {
            free(model_q);
            return EXIT_FAILURE;
        }

        /* Run inference */
        predicted = model_int16_predict(model_q, input_q);
        model = model_q;
        input = input_q;
        predict = predict_int16;
    } else {
        model_fp32_t *model_f = (model_fp32_t *)malloc(sizeof(model_fp32_t));
        if (!model_f) {
            fprintf(stderr, "Error: out of memory allocating model structure\n");
            return EXIT_FAILURE;
        }

        /* Normalize to MNIST training distribution. */
        model_fp32_normalize_input(pixels_raw, input_f32);

        /* Load the fp32 weights. */
        if (model_fp32_load_weights(weights_path, model_f) != 0) {
            free(model_f);
            return EXIT_FAILURE;
        }

        /* Run inference */
        predicted = model_fp32_predict(model_f, input_f32);
        model = model_f;
        input = input_f32;
        predict = predict_fp32;
    }

    printf("%d\n", predicted);

    if (benchmark_count > 0) {
        benchmark(predict, model, input, benchmark_count);
    }

    free((void *)model);
    return EXIT_SUCCESS;
}
