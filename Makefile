# Makefile — MNIST inference for Windows 98 and Linux
#
# Targets:
#   build/inference           — MNIST inference (Linux, release)
#   build/inference-debug     — MNIST inference (Linux, dev)
#   build/inference-linux32   — MNIST inference (Linux, x86 32-bit baseline, no MMX/SSE)
#   build/inference-linux32-sse — MNIST inference (Linux, x86 32-bit with MMX; tests the int16 MMX path)
#   build/inference-win32     — MNIST inference (Windows, baseline)
#   build/inference-win32-sse — MNIST inference (Windows, SSE)
#   disasm                    — Disassemble all built executables into .asm files

CC_WIN32    := i686-w64-mingw32-gcc
CC_LINUX    := gcc

OBJDUMP_LINUX := objdump
OBJDUMP_WIN32 := i686-w64-mingw32-objdump
OBJDUMP_FLAGS := -d -S --no-show-raw-insn

CFLAGS_BASE          := -std=c89 -pedantic -Wall -Wextra -Werror -Iinclude
CFLAGS_LINUX_DEBUG   := $(CFLAGS_BASE) -g
CFLAGS_LINUX_RELEASE := $(CFLAGS_BASE) -O3 -ffast-math
CFLAGS_LINUX32       := $(CFLAGS_BASE) -O3 -m32
CFLAGS_LINUX32_SSE   := $(CFLAGS_BASE) -O3 -m32 -mmmx -msse
CFLAGS_WIN32         := $(CFLAGS_BASE) -O3
CFLAGS_WIN32_SSE     := $(CFLAGS_BASE) -O3 -ffast-math -march=pentium3 -mmmx -msse

LDFLAGS     :=

BUILD_DIR   := build

INF_SRC_BASE    := src/main.c src/model_fp32.c src/model_int16.c src/bmp.c
INF_SRC         := $(INF_SRC_BASE) src/int16_dot.c       # 64-bit targets or 32-bit targets (pure C dot product)
INF_SRC_X86_MMX := $(INF_SRC_BASE) src/int16_dot_x86_mmx.S   # 32-bit targets with MMX only
INF_HDR         := include/types.h include/model_fp32.h include/model_int16.h include/int16_dot.h include/bmp.h

INF_TARGET_LINUX_DEBUG   := $(BUILD_DIR)/inference-debug
INF_TARGET_LINUX_RELEASE := $(BUILD_DIR)/inference
INF_TARGET_LINUX32       := $(BUILD_DIR)/inference-linux32
INF_TARGET_LINUX32_SSE   := $(BUILD_DIR)/inference-linux32-sse
INF_TARGET_WIN32         := $(BUILD_DIR)/inference-win32.exe
INF_TARGET_WIN32_SSE     := $(BUILD_DIR)/inference-win32-sse.exe

# Disassembly target output files
DISASM_TARGETS := \
    $(INF_TARGET_LINUX_DEBUG).asm \
    $(INF_TARGET_LINUX_RELEASE).asm \
    $(INF_TARGET_LINUX32).asm \
    $(INF_TARGET_LINUX32_SSE).asm \
    $(INF_TARGET_WIN32:.exe=.asm) \
    $(INF_TARGET_WIN32_SSE:.exe=.asm)

.PHONY: all clean test disasm

all: $(INF_TARGET_LINUX_DEBUG) $(INF_TARGET_LINUX_RELEASE) $(INF_TARGET_LINUX32) $(INF_TARGET_LINUX32_SSE) $(INF_TARGET_WIN32) $(INF_TARGET_WIN32_SSE)

disasm: $(DISASM_TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(INF_TARGET_LINUX_DEBUG): $(INF_SRC) $(INF_HDR) | $(BUILD_DIR)
	$(CC_LINUX) $(CFLAGS_LINUX_DEBUG) $(LDFLAGS) -o $@ $(INF_SRC)

$(INF_TARGET_LINUX_RELEASE): $(INF_SRC) $(INF_HDR) | $(BUILD_DIR)
	$(CC_LINUX) $(CFLAGS_LINUX_RELEASE) $(LDFLAGS) -o $@ $(INF_SRC)

# Requires gcc-multilib (32-bit libc) — e.g. `sudo apt install gcc-multilib`.
# Baseline 32-bit build: no MMX/SSE, uses the pure C int16 dot product.
$(INF_TARGET_LINUX32): $(INF_SRC) $(INF_HDR) | $(BUILD_DIR)
	$(CC_LINUX) $(CFLAGS_LINUX32) $(LDFLAGS) -o $@ $(INF_SRC)

# Uses the MMX int16 dot product (src/int16_dot_x86_mmx.S).
$(INF_TARGET_LINUX32_SSE): $(INF_SRC_X86_MMX) $(INF_HDR) | $(BUILD_DIR)
	$(CC_LINUX) $(CFLAGS_LINUX32_SSE) $(LDFLAGS) -o $@ $(INF_SRC_X86_MMX)

$(INF_TARGET_WIN32): $(INF_SRC) $(INF_HDR) | $(BUILD_DIR)
	$(CC_WIN32) $(CFLAGS_WIN32) $(LDFLAGS) -o $@ $(INF_SRC)

# Uses the MMX int16 dot product (src/int16_dot.S).
$(INF_TARGET_WIN32_SSE): $(INF_SRC_X86_MMX) $(INF_HDR) | $(BUILD_DIR)
	$(CC_WIN32) $(CFLAGS_WIN32_SSE) $(LDFLAGS) -o $@ $(INF_SRC_X86_MMX)

# --- Disassembly Rules ---

# Disassemble Linux Executables
$(BUILD_DIR)/%.asm: $(BUILD_DIR)/%
	$(OBJDUMP_LINUX) $(OBJDUMP_FLAGS) $< > $@

# Disassemble Windows Executables
$(BUILD_DIR)/%-win32.asm: $(BUILD_DIR)/%-win32.exe
	$(OBJDUMP_WIN32) $(OBJDUMP_FLAGS) $< > $@

$(BUILD_DIR)/%-win32-sse.asm: $(BUILD_DIR)/%-win32-sse.exe
	$(OBJDUMP_WIN32) $(OBJDUMP_FLAGS) $< > $@

# Run inference on each digit image and verify the prediction matches.
test: $(INF_TARGET_LINUX_RELEASE)
	@bash run_tests.sh

clean:
	rm -rf $(BUILD_DIR)