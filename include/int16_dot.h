/* int16_dot.h — Dot product of int16 vectors (32-bit x86, MMX assembly). */

#ifndef INT16_DOT_H
#define INT16_DOT_H

#include "types.h"

/*
 * Dot product: returns sum(a[i] * b[i]) for i in [0, n).
 *
 * Two implementations exist; exactly one is linked into each build:
 *   - 32-bit x86 with MMX: assembly kernel (src/int16_dot.S) using
 *     pmaddwd — 4 int16 multiplies and 2 int32 additions per instruction.
 *     n must be a multiple of 4.
 *   - all other builds: pure C fallback (src/model_int16.c).
 *
 * In either case the accumulation is int32; the caller guarantees it
 * cannot overflow by construction.
 */
int32_t int16_dot(const int16_t *a, const int16_t *b, int n);

#endif
