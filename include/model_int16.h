/* model_int16.h — Quantized MLP weights and integer inference for MNIST digit classification.
 *
 * Same 784 → 64 → 10 architecture as model_fp32.h, but with int16 weights
 * and int32 accumulation so the forward pass uses no floating point.
 * Weights come from QuantizedMLP.export_weights() (training/model_quant.py).
 */

#ifndef MODEL_INT16_H
#define MODEL_INT16_H

#include "types.h"

/* Model dimensions (same as the fp32 model). */
#define MODEL_INPUT  784
#define MODEL_HIDDEN 64
#define MODEL_OUTPUT 10

/*
 * Integer encoding of the input pixels.
 *
 * Raw pixels (0-255) are mapped linearly to int16 so that the integer
 * forward pass matches the quantized training pipeline:
 *
 *   pixel 0   → -4939
 *   pixel 255 →  32767
 */
#define QPIX_MIN  (-4939)  /* int16 value of pixel 0   */
#define QPIX_MAX  (32767)  /* int16 value of pixel 255 */
#define QPIX_SPAN (QPIX_MAX - QPIX_MIN)

/*
 * Container for the full set of trained parameters (quantized int16 model).
 *
 * Layout matches the binary format produced by QuantizedMLP's
 * export_weights() — row-major, no padding.
 * Weights are int16, biases are int32 (quantized at 4x the precision so
 * they can be added directly to the int32 matmul output).
 */
typedef struct {
    int16_t w0[MODEL_HIDDEN][MODEL_INPUT];  /* Layer 0 weights (64 x 784) */
    int32_t b0[MODEL_HIDDEN];               /* Layer 0 bias    (64)       */
    int16_t w1[MODEL_OUTPUT][MODEL_HIDDEN]; /* Layer 1 weights (10 x 64)  */
    int32_t b1[MODEL_OUTPUT];               /* Layer 1 bias    (10)       */
} model_int16_t;

/*
 * Convert a raw 28x28 image (pixel values 0-255) to the int16 encoding
 * used by the quantized model.
 *
 * Applies per-pixel the linear map (with rounding):
 *   pixel 0 → -4939, pixel 255 → 32767
 *
 * pixels_raw must contain raw pixel values in the range [0, 255];
 * the int16 values are written into input.
 */
void model_int16_quantize_input(const int pixels_raw[MODEL_INPUT], int16_t input[MODEL_INPUT]);

/*
 * Load the quantized weights from the binary format produced by
 * QuantizedMLP.export_weights().
 *
 * Binary format (all little-endian):
 *   int32  magic       = 0x4E4D5354 ("NMST")
 *   int32  num_layers  = 2
 *   int32  in_dim[2]   = {784, 64}
 *   int32  out_dim[2]  = {64, 10}
 *   int16  weights[0][64][784]   (row-major)
 *   int32  biases[0][64]
 *   int16  weights[1][10][64]    (row-major)
 *   int32  biases[1][10]
 *
 * Returns 0 on success, -1 on error.
 */
int model_int16_load_weights(const char *path, model_int16_t *model);

/*
 * Run integer inference on a single digit image.
 *
 * Reads an int16-encoded 28x28 image (784 values, row-major)
 * and returns the predicted digit 0-9.
 *
 * Forward pass (matches QuantizedMLP):
 *   h = relu(w0 @ x + b0)   int32 accumulation
 *   h = h >> 16             rescale to int16 range
 *   y = w1 @ h + b1         int32 accumulation
 *   return argmax(y)        scale-invariant, no dequantization
 */
int model_int16_predict(const model_int16_t *model, const int16_t input[MODEL_INPUT]);

#endif
