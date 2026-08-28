/* int16_dot.c — Pure C version of int16_dot().
 * C89-compatible.
 * Used in builds without the 32-bit MMX assembly kernel.
 */

#include "int16_dot.h"

int32_t int16_dot(const int16_t *a, const int16_t *b, int n)
{
    int32_t sum = 0;
    int i;
    for (i = 0; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}
