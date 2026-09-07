/* scale_int32_to_int16.c — Pure C version of scale_int32_to_int16().
 * C89-compatible.
 * Used in builds without the 32-bit MMX assembly kernel.
 */

#include "scale_int32_to_int16.h"

void scale_int32_to_int16(const int32_t *hidden, int16_t *hidden_q, int n) {
    int i;
    for (i = 0; i < n; i++) {
        hidden_q[i] = (int16_t)(hidden[i] >> 16);
    }
}
