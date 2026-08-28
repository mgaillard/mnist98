/* int16_dot.S — Dot product of int16 vectors using MMX (32-bit x86, cdecl).
 *
 * C interface (see include/int16_dot.h):
 *
 *   int32_t int16_dot(const int16_t *a, const int16_t *b, int n);
 *
 * Returns sum(a[i] * b[i]) for i in [0, n). n must be a multiple of 4.
 *
 * One pmaddwd multiplies 4 packed int16 pairs and produces 2 int32
 * partial sums; the partial sums are accumulated with paddq (one 64-bit
 * add = two independent 32-bit lane adds). The result is the sum of the
 * two final lanes.
 *
 * Requires MMX (Pentium or later). The caller must not pass n values
 * that overflow int32 accumulation.
 *
 * Clobbered: %eax (result), %edx, %esi (saved), MMX registers (emms).
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
    movl    8(%esp), %esi             /* a */
    movl    12(%esp), %edx            /* b */
    movl    16(%esp), %ecx            /* n */
    shr     $2, %ecx                  /* number of 8-byte blocks */
    xor     %eax, %eax
    movd    %eax, %mm0                /* mm0 = (0, 0) (int32 accumulator) */

.Ldot_loop:
    movq    (%esi), %mm1              /* 4 int16 of a */
    movq    (%edx), %mm2              /* 4 int16 of b */
    pmaddwd %mm1, %mm2                /* (a0*b0+a1*b1, a2*b2+a3*b3) stored in mm2 */
    paddq   %mm2, %mm0                /* accumulate both int32 in mm2 into mm0 */
    addl    $8, %esi
    addl    $8, %edx
    decl    %ecx
    jnz     .Ldot_loop

    movd    %mm0, %eax                /* low int32 lane */
    psrlq   $32, %mm0
    movd    %mm0, %edx                /* high int32 lane */
    emms
    popl    %esi
    addl    %edx, %eax                /* total */
    ret

/* Mark the stack as non-executable (ELF only; suppresses the
 * "missing .note.GNU-stack" linker warning). */
#if defined(__ELF__)
    .section .note.GNU-stack, "", @progbits
#endif
