#!/usr/bin/env bash
# run_tests.sh — Run MNIST inference on digit images and verify predictions.
# Tests both the fp32 model and the quantized int16 model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/build/inference"
WEIGHTS_FP32="${SCRIPT_DIR}/training/checkpoints/weights.bin"
WEIGHTS_QUANT="${SCRIPT_DIR}/training/checkpoints/weights_quant.bin"
DATA_DIR="${SCRIPT_DIR}/data"

pass=0
fail=0

# Run the digit test suite once. The model is selected by the arguments:
#   run_suite <label> [inference args...]
run_suite() {
    local label="$1"
    shift
    for bmp in "${DATA_DIR}"/[0-9].bmp; do
        expected="$(basename "$bmp" .bmp)"
        predicted="$("$BINARY" "$bmp" "$@")"
        if [ "$predicted" = "$expected" ]; then
            echo "  PASS [$label] $bmp -> $predicted"
            pass=$((pass + 1))
        else
            echo "  FAIL [$label] $bmp -> expected $expected, got $predicted"
            fail=$((fail + 1))
        fi
    done
}

run_suite "fp32"  --weights "$WEIGHTS_FP32"
run_suite "int16" --quant --weights "$WEIGHTS_QUANT"

echo "Results: $pass passed, $fail failed out of $((pass + fail))"

if [ "$fail" -gt 0 ]; then
    exit 1
fi
