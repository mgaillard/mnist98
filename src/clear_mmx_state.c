/* clear_mmx_state.c — Pure C version of clear_mmx_state().
 * C89-compatible.
 * Does nothing because the C89 compiler does not need the emms instruction.
 */

#include "clear_mmx_state.h"

void clear_mmx_state(void)
{
     /* No-op. */
}
