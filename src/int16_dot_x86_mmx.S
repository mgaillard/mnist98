/* int16_dot_x86_mmx.S — Dot product of int16 vectors using MMX (32-bit x86, cdecl).
 *
 * C interface (see include/int16_dot.h):
 *
 *    int32_t int16_dot(const int16_t *a, const int16_t *b, int n);
 *
 * Returns sum(a[i] * b[i]) for i in [0, n). n must be a multiple of 16.
 * Specialized for aligned vector sizes such as n = 64 and n = 784.
 *
 * One pmaddwd multiplies 4 packed int16 pairs into 2 int32 partial sums.
 * Partial sums are accumulated using paddd (Packed Add 32-bit Doublewords)
 * across 4 independent MMX accumulators to maximize pipeline throughput.
 *
 * Unrolled 4x (16 elements / 32 bytes per iteration) to interleave the
 * 3-cycle PMADDWD execution latency on P6 cores (Pentium II/III) and eliminate
 * loop overhead.
 *
 * Requires base MMX (Pentium or later). The caller must not pass n values
 * that overflow int32 accumulation.
 *
 * Clobbered: %eax (result), %ecx, %edx, %esi (saved), MMX registers (emms).
 *
 * Diagrams:
 *
 * INPUT VECTORS IN MEMORY (16 elements = 32 bytes)
 *                   ================================================
 *  Vector A (%esi): [ a0  a1 | a2  a3 ][ a4  a5 | a6  a7 ][ a8  a9 | a10 a11 ][ a12 a13 | a14 a15 ]
 *  Vector B (%edx): [ b0  b1 | b2  b3 ][ b4  b5 | b6  b7 ][ b8  b9 | b10 b11 ][ b12 b13 | b14 b15 ]
 *                   \_________________/\_________________/\__________________/\___________________/
 *                         Block 0           Block 1             Block 2             Block 3
 *                        (0..7 B)          (8..15 B)           (16..23 B)          (24..31 B)
 * 
 *                                            │
 *                                            ▼
 * ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
 * │ ONE UNROLLED LOOP PASS (Pipelined Execution)                                                │
 * ├─────────────────────────────────────────────────────────────────────────────────────────────┤
 * │                                                                                             │
 * │  [Block 0] Load & Multiply:                                                                 │
 * │     movq (%esi), %mm1        ──> %mm1 = [ a0  a1 | a2  a3 ]                                 │
 * │     pmaddwd (%edx), %mm1     ──> %mm1 = [ (a0*b0 + a1*b1)  |  (a2*b2 + a3*b3) ]             │
 * │                                           (int32 Lane 0)         (int32 Lane 1)             │
 * │                                                                                             │
 * │  [Block 1] Load & Multiply + Accumulate Block 0:                                            │
 * │     movq 8(%esi), %mm4       ──> %mm4 = [ a4  a5 | a6  a7 ]                                 │
 * │     pmaddwd 8(%edx), %mm4    ──> %mm4 = [ (a4*b4 + a5*b5)  |  (a6*b6 + a7*b7) ]             │
 * │     paddd %mm1, %mm0         ──> ACCUMULATOR 0 (%mm0) += %mm1  <── (Hides PMADDWD Latency)  │
 * │                                                                                             │
 * │  [Block 2] Load & Multiply + Accumulate Block 1:                                            │
 * │     movq 16(%esi), %mm1      ──> %mm1 = [ a8  a9 | a10 a11 ]                                │
 * │     pmaddwd 16(%edx), %mm1   ──> %mm1 = [ (a8*b8 + a9*b9)  |  (a10*b10 + a11*b11) ]         │
 * │     paddd %mm4, %mm3         ──> ACCUMULATOR 1 (%mm3) += %mm4                               │
 * │                                                                                             │
 * │  [Block 3] Load & Multiply + Accumulate Blocks 2 & 3:                                       │
 * │     movq 24(%esi), %mm4      ──> %mm4 = [ a12 a13 | a14 a15 ]                               │
 * │     pmaddwd 24(%edx), %mm4   ──> %mm4 = [ (a12*b12 + a13*b13) | (a14*b14 + a15*b15) ]       │
 * │     paddd %mm1, %mm6         ──> ACCUMULATOR 2 (%mm6) += %mm1                               │
 * │     paddd %mm4, %mm7         ──> ACCUMULATOR 3 (%mm7) += %mm4                               │
 * │                                                                                             │
 * │  Pointer & Loop Counter Update:                                                             │
 * │     addl $32, %esi ; addl $32, %edx ; decl %ecx                                             │
 * └─────────────────────────────────────────────────────────────────────────────────────────────┘
 *                                            │
 *                                            ▼
 *                       4 INDEPENDENT PARALLEL ACCUMULATORS
 *                       ===================================
 *              %mm0 : [ Sum(Block 0, Lane 0)  |  Sum(Block 0, Lane 1) ]
 *              %mm3 : [ Sum(Block 1, Lane 0)  |  Sum(Block 1, Lane 1) ]
 *              %mm6 : [ Sum(Block 2, Lane 0)  |  Sum(Block 2, Lane 1) ]
 *              %mm7 : [ Sum(Block 3, Lane 0)  |  Sum(Block 3, Lane 1) ]
 */

    .text

/* Windows C symbols have a leading underscore. */
#ifdef _WIN32
#define CNAME(name) _ ## name
#else
#define CNAME(name) name
#endif

    .globl CNAME(int16_dot)
CNAME(int16_dot):
    pushl   %esi
    movl    8(%esp), %esi              /* a */
    movl    12(%esp), %edx             /* b */
    movl    16(%esp), %ecx             /* n (64 or 784) */

    /* Zero out 4 independent accumulators to maximize CPU pipeline throughput */
    pxor    %mm0, %mm0
    pxor    %mm3, %mm3
    pxor    %mm6, %mm6
    pxor    %mm7, %mm7

    /* ecx = n / 16 (number of 32-byte / 16-element iterations) */
    shr     $4, %ecx                   

.p2align 4
.Ldot_loop_unrolled_4x:
    /* Stage 1: Load vector A (all 4 blocks / 16 elements = 32 bytes) */
    movq    (%esi), %mm1
    movq    8(%esi), %mm2
    movq    16(%esi), %mm4
    movq    24(%esi), %mm5

    /* Stage 2: Multiply with vector B directly from memory.
     * PMADDWD takes 2 memory/register operands, avoiding extra MOVQ loads for B.
     * Hides load-to-use latency of Stage 1. */
    pmaddwd (%edx), %mm1
    pmaddwd 8(%edx), %mm2
    pmaddwd 16(%edx), %mm4
    pmaddwd 24(%edx), %mm5

    /* Stage 3: Accumulate into four independent MMX accumulators (%mm0, %mm3, %mm6, %mm7).
     * Hides PMADDWD execution latency (typically 3 cycles on P6). */
    paddd   %mm1, %mm0
    paddd   %mm2, %mm3
    paddd   %mm4, %mm6
    paddd   %mm5, %mm7

    /* Advance memory pointers and loop counter */
    addl    $32, %esi
    addl    $32, %edx
    decl    %ecx
    jnz     .Ldot_loop_unrolled_4x

    /* Sum all 4 MMX accumulators down to %mm0 */
    paddd   %mm3, %mm0
    paddd   %mm7, %mm6
    paddd   %mm6, %mm0

    /* Horizontal reduction to 32-bit scalar integer in %eax */
    movd    %mm0, %eax                 /* low 32 bits */
    psrlq   $32, %mm0
    movd    %mm0, %edx                 /* high 32 bits */
    popl    %esi
    addl    %edx, %eax                 /* final dot product result */
    ret

/* Mark the stack as non-executable (ELF only; suppresses the
 * "missing .note.GNU-stack" linker warning). */
#if defined(__ELF__)
    .section .note.GNU-stack, "", @progbits
#endif
