#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <fp32_model.bin> <tokenizer> [q8_model.bin] [rounds]" >&2
  exit 1
fi

FP32_MODEL=$1
TOKENIZER=$2
Q8_MODEL=${3:-}
ROUNDS=${4:-3}
PROMPT=${PROMPT:-"Once upon a time"}
DEMO=${DEMO:-"./build/demo/llm_infer"}

if [[ ! -x "$DEMO" ]]; then
  echo "Demo executable not found: $DEMO" >&2
  exit 1
fi
if [[ ! -f "$FP32_MODEL" || ! -f "$TOKENIZER" ]]; then
  echo "FP32 model or tokenizer does not exist" >&2
  exit 1
fi
if ! [[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "rounds must be a positive integer" >&2
  exit 1
fi
if [[ -n "$Q8_MODEL" && ! -f "$Q8_MODEL" ]]; then
  echo "Q8 model does not exist: $Q8_MODEL" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
RESULTS="$TMP_DIR/results.tsv"
printf 'round\tformat\tttft_ms\tdecode_ms\ttotal_ms\ttokens_per_second\n' > "$RESULTS"

extract_metric() {
  local key=$1
  local file=$2
  awk -F': ' -v key="$key" '$1 ~ "  " key {print $2; exit}' "$file"
}

run_one() {
  local round=$1
  local format=$2
  local model=$3
  shift 3
  local output="$TMP_DIR/${format}-${round}.log"
  "$DEMO" "$model" "$TOKENIZER" "$@" "$PROMPT" >"$output" 2>&1
  local ttft decode total tps
  ttft=$(extract_metric ttft_ms "$output")
  decode=$(extract_metric avg_decode_latency_ms "$output")
  total=$(extract_metric total_latency_ms "$output")
  tps=$(extract_metric tokens_per_second "$output")
  if [[ -z "$ttft" || -z "$decode" || -z "$total" || -z "$tps" ]]; then
    echo "Failed to parse metrics from $output" >&2
    cat "$output" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$round" "$format" "$ttft" "$decode" "$total" "$tps" >> "$RESULTS"
}

for ((round = 1; round <= ROUNDS; ++round)); do
  run_one "$round" fp32 "$FP32_MODEL"
  if [[ -n "$Q8_MODEL" ]]; then
    run_one "$round" q8 "$Q8_MODEL" --quant
  fi
done

echo "# Inference Benchmark"
echo
printf '%s\n' "- Prompt: \`$PROMPT\`"
echo "- Rounds: $ROUNDS"
printf '%s\n' "- Demo: \`$DEMO\`"
echo
echo "| Round | Format | TTFT (ms) | Decode (ms/token) | Total (ms) | Tokens/s |"
echo "|---:|---|---:|---:|---:|---:|"
tail -n +2 "$RESULTS" | awk -F '\t' '{printf "| %s | %s | %.3f | %.3f | %.3f | %.3f |\n", $1,$2,$3,$4,$5,$6}'
echo
echo "## Averages"
echo
echo "| Format | TTFT (ms) | Decode (ms/token) | Total (ms) | Tokens/s |"
echo "|---|---:|---:|---:|---:|"
tail -n +2 "$RESULTS" | awk -F '\t' '
  {n[$2]++; ttft[$2]+=$3; decode[$2]+=$4; total[$2]+=$5; tps[$2]+=$6}
  END {for (format in n) printf "| %s | %.3f | %.3f | %.3f | %.3f |\n", format, ttft[format]/n[format], decode[format]/n[format], total[format]/n[format], tps[format]/n[format]}' | sort
