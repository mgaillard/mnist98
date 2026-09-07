/* scale_int32_to_int16_x86_mmx.S — Downscale int32_t vector to int16_t using MMX (32-bit x86, cdecl).
 *
 * C interface (see include/scale_int32_to_int16.h):
 *
 *    void scale_int32_to_int16(const int32_t *hidden, int16_t *hidden_q, int n);
 *
 * Shifts each 32-bit integer right by 16 bits and packs with signed saturation into
 * 16-bit integers: hidden_q[i] = (int16_t)(hidden[i] >> 16).
 *
 * Requires n to be a multiple of 16 (e.g. n = 64). Memory pointers should be 
 * 8-byte aligned for optimal performance.
 *
 * Clobbered: %eax, %ecx, %edx, MMX registers (%mm0-%mm7, emms).
 *
 * Diagrams:
 *
 * INPUT (32-bit integers after PSRAD $16)
 *                =======================================
 *    %mm0 (low dst):  [ 0x0000AAAA (hidden[0]) | 0x0000BBBB (hidden[1]) ]
 *    %mm1 (high src): [ 0x0000CCCC (hidden[2]) | 0x0000DDDD (hidden[3]) ]
 * 
 *                                   │
 *                                   ▼
 *                       packssdw %mm1, %mm0
 *                                   │
 *                                   ▼
 *                OUTPUT (4 packed 16-bit integers in %mm0)
 *                =========================================
 *    %mm0: [ 0xAAAA (q0) | 0xBBBB (q1) | 0xCCCC (q2) | 0xDDDD (q3) ]
 */

    .text

/* Windows C symbols have a leading underscore. */
#ifdef _WIN32
#define CNAME(name) _ ## name
#else
#define CNAME(name) name
#endif

    .globl CNAME(scale_int32_to_int16)
CNAME(scale_int32_to_int16):
    movl    4(%esp), %eax              /* hidden (const int32_t *) */
    movl    8(%esp), %edx              /* hidden_q (int16_t *) */
    movl    12(%esp), %ecx             /* n */

    /* ecx = n / 16 (number of 64-byte input / 32-byte output blocks) */
    shr     $4, %ecx                   

.p2align 4
.Lscale_loop:
    /* Block 0: elements 0..3 (16 bytes in -> 8 bytes out) */
    movq    (%eax), %mm0
    movq    8(%eax), %mm1
    psrad   $16, %mm0
    psrad   $16, %mm1
    packssdw %mm1, %mm0
    movq    %mm0, (%edx)

    /* Block 1: elements 4..7 (16 bytes in -> 8 bytes out) */
    movq    16(%eax), %mm2
    movq    24(%eax), %mm3
    psrad   $16, %mm2
    psrad   $16, %mm3
    packssdw %mm3, %mm2
    movq    %mm2, 8(%edx)

    /* Block 2: elements 8..11 (16 bytes in -> 8 bytes out) */
    movq    32(%eax), %mm4
    movq    40(%eax), %mm5
    psrad   $16, %mm4
    psrad   $16, %mm5
    packssdw %mm5, %mm4
    movq    %mm4, 16(%edx)

    /* Block 3: elements 12..15 (16 bytes in -> 8 bytes out) */
    movq    48(%eax), %mm6
    movq    56(%eax), %mm7
    psrad   $16, %mm6
    psrad   $16, %mm7
    packssdw %mm7, %mm6
    movq    %mm6, 24(%edx)

    /* Advance memory pointers and loop counter */
    addl    $64, %eax                  /* 16 x int32 = 64 bytes read */
    addl    $32, %edx                  /* 16 x int16 = 32 bytes written */
    decl    %ecx
    jnz     .Lscale_loop

    ret

/* Mark stack as non-executable for ELF linkers */
#if defined(__ELF__)
    .section .note.GNU-stack, "", @progbits
#endif
