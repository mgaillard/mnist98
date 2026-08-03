/* inference.c — MNIST digit classification from a BMP image.
 * C89-compatible.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bmp.h"
#include "weights.h"

static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <image.bmp> [--weights weights.bin] [--benchmark N]\n", prog);
    exit(EXIT_FAILURE);
}

static void benchmark(const model_t *model, const float input[MODEL_INPUT], int count)
{
    clock_t t_start, t_end;
    double elapsed_sec;
    double avg_ms;
    double throughput;
    int i;

    /* Warm-up run */
    model_predict(model, input);

    t_start = clock();
    for (i = 0; i < count; i++) {
        model_predict(model, input);
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
    int benchmark_count;
    float input[MODEL_INPUT];
    float pixels[28][28];
    model_t *model;
    int predicted;
    int i, x, y;

    /* Defaults. */
    weights_path = "weights.bin";
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
        } else {
            fprintf(stderr, "Error: unknown option '%s'\n", argv[i]);
            usage(argv[0]);
        }
    }

    /* Allocate model on HEAP to avoid stack overflow (~204 KB) */
    model = (model_t *)malloc(sizeof(model_t));
    if (!model) {
        fprintf(stderr, "Error: out of memory allocating model structure\n");
        return EXIT_FAILURE;
    }

    /* Load the BMP image. */
    if (load_bmp(img_path, pixels) != 0) {
        free(model);
        return EXIT_FAILURE;
    }

    /* Flatten 28x28 to 784 (row-major). */
    for (y = 0; y < 28; y++) {
        for (x = 0; x < 28; x++) {
            input[y * 28 + x] = pixels[y][x];
        }
    }

    /* Normalize to MNIST training distribution. */
    normalize_input(input);

    /* Load model weights. */
    if (load_weights(weights_path, model) != 0) {
        free(model);
        return EXIT_FAILURE;
    }

    /* Run inference */
    predicted = model_predict(model, input);
    printf("%d\n", predicted);

    if (benchmark_count > 0) {
        benchmark(model, input, benchmark_count);
    }

    free(model);
    return EXIT_SUCCESS;
}
