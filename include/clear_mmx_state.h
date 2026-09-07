/* clear_mmx_state.h — Reset x87 FPU state after MMX operations (C89 compatible). */

#ifndef CLEAR_MMX_STATE_H
#define CLEAR_MMX_STATE_H

/**
 * clear_mmx_state() — Reset x87 FPU state after MMX operations (C89 compatible).
 * MMX registers (%mm0-%mm7) alias onto the physical x87 FPU stack.
 * Executing MMX instructions marks the FPU tag word as in-use, which causes
 * subsequent x87 floating-point operations to trigger stack overflow exceptions.
 */
void clear_mmx_state(void);

#endif /* CLEAR_MMX_STATE_H */
