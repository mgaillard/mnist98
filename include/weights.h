/* weights.h — MLP model weights and inference for MNIST digit classification. */

#ifndef WEIGHTS_H
#define WEIGHTS_H

/* Model dimensions. */
#define MODEL_INPUT  784
#define MODEL_HIDDEN 64
#define MODEL_OUTPUT 10

/*
 * Container for the full set of trained parameters.
 *
 * Layout matches the binary format produced by PyTorch's
 * MLP.export_weights() — row-major, no padding.
 */
typedef struct {
    float w0[MODEL_HIDDEN][MODEL_INPUT];  /* Layer 0 weights (64 x 784) */
    float b0[MODEL_HIDDEN];               /* Layer 0 bias    (64)       */
    float w1[MODEL_OUTPUT][MODEL_HIDDEN]; /* Layer 1 weights (10 x 64)  */
    float b1[MODEL_OUTPUT];               /* Layer 1 bias    (10)       */
} model_t;

/*
 * Normalize a raw 28x28 image to match the MNIST training distribution.
 *
 * Applies per-pixel: (pixel / 255.0f - mean) / std
 * where mean = 0.1307 and std = 0.3081 (MNIST dataset stats).
 *
 * Writes in-place into the input array, which must contain
 * raw pixel values in [0, 255].
 */
void normalize_input(float input[MODEL_INPUT]);

/*
 * Load model weights from the binary format produced by
 * PyTorch's MLP.export_weights().
 *
 * Binary format (all little-endian):
 *   int32  magic       = 0x4E4D5354 ("NMST")
 *   int32  num_layers  = 2
 *   int32  in_dim[2]   = {784, 64}
 *   int32  out_dim[2]  = {64, 10}
 *   float  weights[0][64][784]   (row-major)
 *   float  biases[0][64]
 *   float  weights[1][10][64]    (row-major)
 *   float  biases[1][10]
 *
 * Returns 0 on success, -1 on error.
 */
int load_weights(const char *path, model_t *model);

/*
 * Run inference on a single digit image.
 *
 * Reads a normalized 28x28 image (784 float values, row-major)
 * and returns the predicted digit 0-9.
 *
 * Forward pass (matches PyTorch):
 *   x = relu(w0 @ x + b0)
 *   y = w1 @ x + b1
 *   return argmax(y)
 */
int model_predict(const model_t *model, const float input[MODEL_INPUT]);

#endif
