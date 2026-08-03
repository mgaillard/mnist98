/* bmp.c — Load a BMP image into 28x28 grayscale array.
 * C89-compatible.
 */

#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "bmp.h"

typedef struct {
    uint8_t sig[2];
    uint8_t file_size[4];
    uint8_t reserved[4];
    uint8_t data_offset[4];
} bmp_file_header_t;

typedef struct {
    uint8_t header_size[4];
    uint8_t width[4];
    uint8_t height[4];
    uint8_t planes[2];
    uint8_t bits_per_pixel[2];
    uint8_t compression[4];
    uint8_t image_size[4];
    uint8_t x_ppm[4];
    uint8_t y_ppm[4];
    uint8_t colors_used[4];
    uint8_t colors_important[4];
} bmp_info_header_t;

static int rgb_to_grayscale(int r, int g, int b)
{
    return (int)(0.299 * (double)r + 0.587 * (double)g + 0.114 * (double)b);
}

static uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static uint16_t read_u16_le(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

int load_bmp(const char *path, float pixels[28][28])
{
    FILE *f;
    bmp_file_header_t fh;
    bmp_info_header_t ih;
    uint32_t data_offset;
    int width, height, bpp, row_size;
    int y, x;
    uint8_t palette[256][4]; /* Palette for 8-bit images: [B, G, R, Reserved] */

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open '%s'\n", path);
        return -1;
    }

    if (fread(&fh, sizeof(fh), 1, f) != 1 ||
        fh.sig[0] != 'B' || fh.sig[1] != 'M') {
        fprintf(stderr, "Error: invalid BMP signature\n");
        fclose(f);
        return -1;
    }

    if (fread(&ih, sizeof(ih), 1, f) != 1) {
        fprintf(stderr, "Error: cannot read info header\n");
        fclose(f);
        return -1;
    }

    data_offset = read_u32_le(fh.data_offset);
    width       = (int)read_u32_le(ih.width);
    height      = (int)read_u32_le(ih.height);
    bpp         = (int)read_u16_le(ih.bits_per_pixel);

    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: invalid dimensions %dx%d\n", width, height);
        fclose(f);
        return -1;
    }

    /* Read palette if 8-bit */
    if (bpp == 8) {
        if (fread(palette, 4, 256, f) != 256) {
            fprintf(stderr, "Error: failed to read 8-bit palette\n");
            fclose(f);
            return -1;
        }
    }

    /* Jump directly to pixel data start */
    if (fseek(f, (long)data_offset, SEEK_SET) != 0) {
        fprintf(stderr, "Error: failed to seek to pixel data\n");
        fclose(f);
        return -1;
    }

    row_size = (width * bpp + 31) / 32 * 4;

    /* Allocate dynamic buffer for image loading before resizing */
    {
        float *temp_img;
        uint8_t *row;

        temp_img = (float *)malloc((size_t)(width * height) * sizeof(float));
        row      = (uint8_t *)malloc((size_t)row_size);

        if (!temp_img || !row) {
            fprintf(stderr, "Error: out of memory\n");
            free(temp_img);
            free(row);
            fclose(f);
            return -1;
        }

        /* Read BMP rows bottom-to-top */
        for (y = 0; y < height; y++) {
            int src_y;

            if (fread(row, 1, (size_t)row_size, f) != (size_t)row_size) {
                fprintf(stderr, "Error: truncated pixel data\n");
                free(temp_img);
                free(row);
                fclose(f);
                return -1;
            }

            src_y = height - 1 - y;

            if (bpp == 24) {
                for (x = 0; x < width; x++) {
                    temp_img[src_y * width + x] = (float)rgb_to_grayscale(
                        row[x * 3 + 2], row[x * 3 + 1], row[x * 3 + 0]);
                }
            } else if (bpp == 8) {
                for (x = 0; x < width; x++) {
                    uint8_t idx = row[x];
                    temp_img[src_y * width + x] = (float)rgb_to_grayscale(
                        palette[idx][2], palette[idx][1], palette[idx][0]);
                }
            }
        }

        /* Sample temp_img into final 28x28 output */
        {
            double scale_x = (double)width / 28.0;
            double scale_y = (double)height / 28.0;

            for (y = 0; y < 28; y++) {
                for (x = 0; x < 28; x++) {
                    int sx = (int)((double)x * scale_x);
                    int sy = (int)((double)y * scale_y);
                    if (sx >= width)  sx = width - 1;
                    if (sy >= height) sy = height - 1;

                    pixels[y][x] = temp_img[sy * width + sx];
                }
            }
        }

        free(temp_img);
        free(row);
    }

    fclose(f);
    return 0;
}
