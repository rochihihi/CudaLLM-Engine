#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <fp32_model.bin> <tokenizer> <q8_model.bin> [prompt ...]" >&2
  exit 1
fi

FP32_MODEL=$1
TOKENIZER=$2
Q8_MODEL=$3
shift 3
if [[ $# -eq 0 ]]; then
  PROMPTS=("hello" "Once upon a time" "The little boy")
else
  PROMPTS=("$@")
fi

DEMO=${DEMO:-"./build/demo/llm_infer"}
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

extract_text() {
  awk '/^Generating\.\.\.$/ {capture=1; next} /^Benchmark:$/ {capture=0} capture {print}' "$1"
}

run_and_hash() {
  local output=$1
  shift
  "$DEMO" "$@" >"$output" 2>&1
  local text_file="${output}.text"
  extract_text "$output" > "$text_file"
  sha256sum "$text_file" | awk '{print $1}'
}

echo "# FP32/Q8 Output Comparison"
echo
echo "| Prompt | FP32 Tokens | Q8 Tokens | Exact Text Match | FP32 SHA-256 | Q8 SHA-256 |"
echo "|---|---:|---:|---|---|---|"

all_match=yes
for index in "${!PROMPTS[@]}"; do
  prompt=${PROMPTS[$index]}
  fp32_log="$TMP_DIR/fp32-$index.log"
  q8_log="$TMP_DIR/q8-$index.log"
  fp32_hash=$(run_and_hash "$fp32_log" "$FP32_MODEL" "$TOKENIZER" "$prompt")
  q8_hash=$(run_and_hash "$q8_log" "$Q8_MODEL" "$TOKENIZER" --quant "$prompt")
  fp32_tokens=$(awk -F': ' '/  generated_tokens:/ {print $2; exit}' "$fp32_log")
  q8_tokens=$(awk -F': ' '/  generated_tokens:/ {print $2; exit}' "$q8_log")
  match=no
  if [[ "$fp32_hash" == "$q8_hash" ]]; then
    match=yes
  else
    all_match=no
  fi
  printf '| %s | %s | %s | %s | `%s` | `%s` |\n' \
    "$prompt" "$fp32_tokens" "$q8_tokens" "$match" "$fp32_hash" "$q8_hash"
done

echo
if [[ "$all_match" == yes ]]; then
  echo "Result: PASS (all prompts matched exactly)"
else
  echo "Result: FAIL (at least one prompt differed)"
  exit 1
fi
