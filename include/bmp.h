/* bmp.h — Load a 28x28 grayscale image from a BMP file. */

#ifndef BMP_H
#define BMP_H

/*
 * Load a BMP image and resize to a 28x28 float array.
 *
 * Supports:
 *   - 24-bit color BMPs (converted to grayscale)
 *   - 8-bit grayscale BMPs
 *
 * Pixel values are raw intensity in the range [0, 255].
 * The caller is responsible for any normalization.
 *
 * Returns 0 on success, -1 on error.
 */
int load_bmp(const char *path, float pixels[28][28]);

#endif
