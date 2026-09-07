/* clear_mmx_state.S — Clear MMX register state / reset x87 FPU tag word (32-bit x86, cdecl).
 *
 * C interface:
 *
 *    void clear_mmx_state(void);
 *
 * Executes the EMMS instruction to restore the x87 FPU stack tag word.
 * Call this once after executing MMX pipeline steps before returning to
 * floating-point arithmetic.
 */

    .text

/* Windows C symbols have a leading underscore under cdecl. */
#ifdef _WIN32
#define CNAME(name) _ ## name
#else
#define CNAME(name) name
#endif

    .globl CNAME(clear_mmx_state)
CNAME(clear_mmx_state):
    emms
    ret

/* Mark stack as non-executable for ELF linkers */
#if defined(__ELF__)
    .section .note.GNU-stack, "", @progbits
#endif
