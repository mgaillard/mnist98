# Makefile — MNIST inference for Windows 98 and Linux
#
# Targets:
#   build/inference           — MNIST inference (Linux, release + debug symbols)
#   build/inference-debug     — MNIST inference (Linux, dev)
#   build/inference-win32     — MNIST inference (Windows, baseline)
#   build/inference-win32-sse — MNIST inference (Windows, SSE)

CC_WIN32    := i686-w64-mingw32-gcc
CC_LINUX    := gcc

CFLAGS_BASE          := -std=c89 -pedantic -Wall -Wextra -Werror -Iinclude
CFLAGS_LINUX_DEBUG   := $(CFLAGS_BASE) -g
CFLAGS_LINUX_RELEASE := $(CFLAGS_BASE) -O3 -ffast-math
CFLAGS_WIN32         := $(CFLAGS_BASE) -O3
CFLAGS_WIN32_SSE     := $(CFLAGS_BASE) -O3 -ffast-math -march=pentium3 -mmmx -msse

LDFLAGS     :=

BUILD_DIR   := build

INF_SRC         := src/inference.c src/weights.c src/bmp.c
INF_HDR         := include/types.h include/weights.h include/bmp.h

INF_TARGET_LINUX_DEBUG   := $(BUILD_DIR)/inference-debug
INF_TARGET_LINUX_RELEASE := $(BUILD_DIR)/inference
INF_TARGET_WIN32         := $(BUILD_DIR)/inference-win32.exe
INF_TARGET_WIN32_SSE     := $(BUILD_DIR)/inference-win32-sse.exe

.PHONY: all clean

all: $(INF_TARGET_LINUX_DEBUG) $(INF_TARGET_LINUX_RELEASE) $(INF_TARGET_WIN32) $(INF_TARGET_WIN32_SSE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(INF_TARGET_LINUX_DEBUG): $(INF_SRC) $(INF_HDR) | $(BUILD_DIR)
	$(CC_LINUX) $(CFLAGS_LINUX_DEBUG) $(LDFLAGS) -o $@ $(INF_SRC)

$(INF_TARGET_LINUX_RELEASE): $(INF_SRC) $(INF_HDR) | $(BUILD_DIR)
	$(CC_LINUX) $(CFLAGS_LINUX_RELEASE) $(LDFLAGS) -o $@ $(INF_SRC)

$(INF_TARGET_WIN32): $(INF_SRC) $(INF_HDR) | $(BUILD_DIR)
	$(CC_WIN32) $(CFLAGS_WIN32) $(LDFLAGS) -o $@ $(INF_SRC)

$(INF_TARGET_WIN32_SSE): $(INF_SRC) $(INF_HDR) | $(BUILD_DIR)
	$(CC_WIN32) $(CFLAGS_WIN32_SSE) $(LDFLAGS) -o $@ $(INF_SRC)

clean:
	rm -rf $(BUILD_DIR)
