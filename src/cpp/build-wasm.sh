#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
output_path="${1:-"$script_dir/wasm/parakeet-fbank.wasm"}"
emxx_bin="${EMXX:-$(command -v em++ || true)}"

if [[ -z "$emxx_bin" || ! -x "$emxx_bin" ]]; then
  echo "Emscripten C++ is unavailable. Install emsdk or set EMXX." >&2
  exit 1
fi

actual_version="$("$emxx_bin" --version | sed -n '1p')"
echo "Using ${actual_version}"

output_dir="$(dirname "$output_path")"
output_name="$(basename "$output_path")"
mkdir -p "$output_dir"

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/parakeet-fbank.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT

export LC_ALL=C
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

"$emxx_bin" \
  "$script_dir/fbank/parakeet_fbank.cpp" \
  -std=c++20 \
  -O3 \
  -flto \
  -fno-exceptions \
  -fno-rtti \
  -Wall \
  -Wextra \
  -Werror \
  -msimd128 \
  --no-entry \
  -s STANDALONE_WASM=1 \
  -s FILESYSTEM=0 \
  -s ALLOW_MEMORY_GROWTH=0 \
  -s INITIAL_MEMORY=2818048 \
  -s STACK_SIZE=131072 \
  -s ERROR_ON_UNDEFINED_SYMBOLS=1 \
  -s EXPORTED_FUNCTIONS='["_parakeet_fbank_init","_parakeet_fbank_input_ptr","_parakeet_fbank_output_ptr","_parakeet_fbank_rms_bits_ptr","_parakeet_fbank_compute_rms_bits","_parakeet_fbank_begin_batch","_parakeet_fbank_compute_item","_parakeet_fbank_finish_batch","_parakeet_fbank_reset_reuse","_parakeet_fbank_dispose","_parakeet_fbank_last_reused_frames","_parakeet_fbank_max_batch","_parakeet_fbank_max_samples","_parakeet_fbank_max_frames","_parakeet_fbank_bins","_parakeet_fbank_sample_rate","_parakeet_fbank_hop_length","_parakeet_fbank_rms_frame_samples","_parakeet_fbank_max_rms_frames"]' \
  -Wl,--strip-all \
  -o "$build_dir/$output_name"

install -m 0644 "$build_dir/$output_name" "$output_path"
byte_count="$(wc -c < "$output_path" | tr -d '[:space:]')"
digest="$(shasum -a 256 "$output_path" | awk '{print $1}')"
echo "Built $output_path ($byte_count bytes, SHA-256 $digest)"
