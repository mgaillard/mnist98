/* scale_int32_to_int16.h — Rescaling int32 vectors to int16 (32-bit x86, MMX assembly). */

#ifndef SCALE_INT32_TO_INT16_H
#define SCALE_INT32_TO_INT16_H

#include "types.h"

/*
 * Rescales hidden layer activations: hidden_q[i] = (int16_t)(hidden[i] >> 16) for i in [0, n).
 *
 * Two implementations exist; exactly one is linked into each build:
 *   - 32-bit x86 with MMX: assembly kernel (src/scale_int32_to_int16.S) using
 *     psrad and packssdw — arithmetic right shifts and signed 32-to-16 packing.
 *     n must be a multiple of 16.
 *   - all other builds: pure C fallback (src/model_int16.c).
 */
void scale_int32_to_int16(const int32_t *hidden, int16_t *hidden_q, int n);

#endif
