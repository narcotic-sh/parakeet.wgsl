# Accuracy benchmark

This directory contains the reproducible LibriSpeech accuracy benchmark for
parakeet.wgsl. It evaluates the complete `test-clean` and `test-other` splits
and reports corpus word error rate (WER) for each split separately.

## Run

From the repository root:

```bash
uv run --project benchmark --frozen --python 3.13 \
  python3 benchmark/benchmark.py
```

The command installs the locked JavaScript workspace, builds the public package,
downloads and verifies the pinned dataset, launches installed Google Chrome,
and evaluates all 5,559 utterances. It requires `uv`, Node.js, pnpm, Google
Chrome, and enough free space for the dataset and selected model package.

LibriSpeech is public, so authentication is optional. When `HF_TOKEN` is
present in the environment, the downloader passes it to Hugging Face.

Results are written incrementally beneath `benchmark/results/full/`. Rerunning
the same command validates the existing records and resumes at the first
missing utterance. A changed package build, benchmark implementation, dataset,
scorer, browser version, or execution profile cannot resume into the same
result directory.

For a non-reportable smoke test:

```bash
uv run --project benchmark --frozen --python 3.13 \
  python3 benchmark/benchmark.py --limit 10
```

Pass `--headed` to show Chrome while the benchmark runs. See `--help` for split,
cache, browser, and output controls.

## Methodology

The benchmark downloads the LibriSpeech Parquet files from
[`hf-audio/open-asr-leaderboard`](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)
at revision `b6bdcd0beb34f8975dc659796176d88f43aff502`. Before any evaluation it
checks the exact downloaded-file length and SHA-256 of each split:

| Split | Utterances | Parquet SHA-256 |
| --- | ---: | --- |
| `test-clean` | 2,620 | `e576cf31312bf67d5ab714ac1dd93fb52d60d62659b76402ba1f97484464ddc3` |
| `test-other` | 2,939 | `5a5e64cd25df8e094cba0b5b1cc8e68c031ae4483bc7f7c0a191ebe3aa582f66` |

Every embedded LibriSpeech source is 16 kHz mono PCM16 FLAC. The harness decodes
those samples directly to signed 16-bit integers and wraps them in a canonical
WAV without resampling or passing through floating point. This benchmarks ASR
accuracy rather than compressed-container decoding.

Chrome loads `dist/index.js`, the same public build consumed by an application.
One transcriber is reused for every utterance, which is submitted independently
in pinned dataset order. Only the final transcript is scored. A failed or
missing transcription stops the run; it is never dropped from the denominator.

Both reported metrics use the exact English normalizer from the
[`huggingface/open_asr_leaderboard`](https://github.com/huggingface/open_asr_leaderboard)
repository at commit `9585fc39bff55697a2ec1c5f13921b18812bfde8`.
They differ only in how normalized word sequences are aligned:

1. **Standard WER** uses ordinary word-level Levenshtein alignment through
   [JiWER 3.1.0](https://github.com/jitsi/jiwer/tree/f2f13e0231d2de54dbd7abeea951de676159a715).
   This follows the Open ASR
   [NeMo evaluation path](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/nemo_asr/run_eval.py#L181-L184)
   used for the model-card-comparable LibriSpeech figures.
2. **Open ASR Leaderboard WER** reproduces the repository's current central
   [scorer](https://github.com/huggingface/open_asr_leaderboard/blob/9585fc39bff55697a2ec1c5f13921b18812bfde8/normalizer/eval_utils.py#L323-L330)
   with Kaldialign 0.12.0 and `merge_compounds=True`. It treats adjacent words
   split or joined only by tokenization as zero-cost equivalents, so it must
   not be compared directly with conventional WER figures.

For each metric, the harness preserves utterance boundaries, sums integer
insertions, deletions, substitutions, and reference words over the complete
split, then calculates
`WER = (insertions + deletions + substitutions) / reference words`. Both are
corpus micro-averages, not averages of utterance WERs, and the two LibriSpeech
splits are never pooled.

An independent dynamic-programming implementation recomputes both edit totals
for every utterance and the final corpus. Standard totals must agree between
JiWER, Kaldialign without compound merging, and the independent implementation;
compound-aware totals must agree between Kaldialign and the independent
implementation. Raw hypotheses, normalized text, both per-utterance scores,
input hashes, exact numerators and denominators, and complete run provenance
remain in the generated JSON/JSONL artifacts.

Execution precision is selected by the public runtime from browser capability:
`shader-f16` selects FP16 and its absence selects FP32. Results must therefore
be identified by their recorded execution profile rather than treated as
cross-precision measurements.

## Results

The full pinned benchmark was run from package commit
[`ff37866`](https://github.com/narcotic-sh/parakeet.wgsl/commit/ff37866ea59807176503641d81d5faeb08c195bb)
with Chrome 151.0.7922.76, the production FP16/subgroups execution profile, and
model manifest SHA-256
`11a359db3d050fd82b002c745b24a5280f3ff13a76834b548df671c95c786c65`.

| Scoring | Split | Utterances | Substitutions | Deletions | Insertions | Errors / reference words | WER |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard WER | `test-clean` | 2,620 | 668 | 129 | 100 | 897 / 53,004 | **1.69%** |
| Standard WER | `test-other` | 2,939 | 1,339 | 154 | 190 | 1,683 / 52,841 | **3.19%** |
| Open ASR Leaderboard WER (compound-aware) | `test-clean` | 2,620 | 565 | 72 | 57 | 694 / 53,004 | 1.31% |
| Open ASR Leaderboard WER (compound-aware) | `test-other` | 2,939 | 1,220 | 94 | 144 | 1,458 / 52,841 | 2.76% |

Displayed WER values are rounded to two decimal places; the integer error and
reference-word totals are authoritative. The standard WER rows are the
conventional, model-card-comparable results. The compound-aware rows reproduce
the separate Open ASR Leaderboard scorer described above.

This benchmark measures accuracy only. It does not collect or report latency,
throughput, RTFx, or other performance figures.
