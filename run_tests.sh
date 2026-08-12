#!/usr/bin/env bash
# run_tests.sh — Run MNIST inference on digit images and verify predictions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/build/inference"
WEIGHTS="${SCRIPT_DIR}/training/checkpoints/weights.bin"
DATA_DIR="${SCRIPT_DIR}/data"

pass=0
fail=0

for bmp in "${DATA_DIR}"/[0-9].bmp; do
    expected="$(basename "$bmp" .bmp)"
    predicted="$("$BINARY" "$bmp" --weights "$WEIGHTS")"
    if [ "$predicted" = "$expected" ]; then
        echo "  PASS $bmp -> $predicted"
        pass=$((pass + 1))
    else
        echo "  FAIL $bmp -> expected $expected, got $predicted"
        fail=$((fail + 1))
    fi
done

echo "Results: $pass passed, $fail failed out of $((pass + fail))"

if [ "$fail" -gt 0 ]; then
    exit 1
fi
