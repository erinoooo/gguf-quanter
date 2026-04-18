# gguf-quanter

A CLI tool for bulk-creating GGUF quantizations of HuggingFace models. It
shells out to your **system-installed** `llama.cpp` (the real C++ binary
`llama-quantize` and the script `convert_hf_to_gguf.py`) — it does **not**
use `llama-cpp-python`.

## What it does

For a given HuggingFace model ID it:

1. Downloads the model snapshot from HuggingFace.
2. Converts it to a base GGUF file (F16 by default) using
   `convert_hf_to_gguf.py`.
3. Runs `llama-quantize` for each requested quant type, producing a separate
   `.gguf` file per type.
4. Optionally cleans up the source model and/or base GGUF when done.

Each step is skip-on-exists by default so you can resume a run and add more
quant types later.

## Prerequisites

- Python 3.10+
- A working **llama.cpp** install:
  - `llama-quantize` binary on `$PATH` (or pass `--quantize-bin`)
  - `convert_hf_to_gguf.py` from the llama.cpp source tree (point at it with
    `--llama-cpp-dir /path/to/llama.cpp` or `--convert-script`)
- The Python deps that `convert_hf_to_gguf.py` itself needs (typically
  `numpy`, `safetensors`, `sentencepiece`, `transformers`, `torch`,
  `gguf`). The easiest way is to `pip install -r requirements.txt` from the
  llama.cpp checkout into the same venv you use for this tool.

## Install

```bash
git clone <this repo> gguf-quanter
cd gguf-quanter
pip install -e .
```

This gives you a `gguf-quant` command.

You can also run it without installing:

```bash
python -m gguf_quanter ...
```

## Quick usage

```bash
# A common selection of quants
gguf-quant quantize meta-llama/Llama-3.2-3B-Instruct -q preset:common

# Specific types
gguf-quant quantize Qwen/Qwen2.5-7B -q Q4_K_M Q5_K_M Q8_0 -o ./out

# Everything llama.cpp supports, then nuke the source files
gguf-quant quantize mistralai/Mistral-7B-v0.3 -q preset:all \
    --cleanup-hf --cleanup-base

# List what's available
gguf-quant list-quants
```

Example output layout (with basename `Llama-3.2-3B-Instruct`):

```
quants/
├── Llama-3.2-3B-Instruct-hf/        # downloaded HF snapshot
├── Llama-3.2-3B-Instruct-F16.gguf   # base GGUF (input to llama-quantize)
├── Llama-3.2-3B-Instruct-Q4_K_M.gguf
├── Llama-3.2-3B-Instruct-Q5_K_M.gguf
├── Llama-3.2-3B-Instruct-Q6_K.gguf
└── Llama-3.2-3B-Instruct-Q8_0.gguf
```

## Supported quant types

Run `gguf-quant list-quants` for the full table. Categories supported:

- **Float**: `F32`, `F16`, `BF16`
- **Legacy**: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
- **K-quants**: `Q2_K`, `Q2_K_S`, `Q3_K_S/M/L`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K`
- **I-quants**: `IQ1_S/M`, `IQ2_XXS/XS/S/M`, `IQ3_XXS/XS/S/M`, `IQ4_XS/NL`
- **Ternary**: `TQ1_0`, `TQ2_0`

Presets: `common`, `small`, `balanced`, `quality`, `iquants`, `k-only`, `all`.

## Useful flags

| Flag | What it does |
| --- | --- |
| `-q / --quants` | List of types and/or `preset:NAME` |
| `-o / --output-dir` | Where everything goes (default `./quants`) |
| `--base-type` | `f16` (default), `bf16`, or `f32` for the base GGUF |
| `--imatrix FILE` | Pass an importance matrix to `llama-quantize`. Strongly recommended for IQ-quants ≤ 3 bpw |
| `-t / --threads N` | Threads for `llama-quantize` |
| `--prefer-safetensors` | Skip `pytorch_model.bin` / `consolidated.pth` when both formats are on the repo |
| `--hf-token TOKEN` | For gated/private models (or set `$HF_TOKEN`) |
| `--revision REV` | Pin to a specific HF branch/tag/commit |
| `--overwrite` | Re-quantize even if the output exists |
| `--skip-download` | Use an already-downloaded HF dir |
| `--force-download` | Re-download even if it's already there |
| `--force-convert` | Re-run base conversion even if the base GGUF exists |
| `--cleanup-hf` | Delete the HF snapshot when done |
| `--cleanup-base` | Delete the base GGUF when done (kept if it is itself a requested quant) |
| `--fail-fast` | Stop on the first failed quant |
| `--allow-requantize` | Pass `--allow-requantize` to `llama-quantize` |
| `--quantize-bin PATH` | Override binary discovery |
| `--convert-script PATH` | Override convert-script discovery |
| `--llama-cpp-dir DIR` | Point at your llama.cpp checkout (or set `$LLAMA_CPP_DIR`) |

## Notes

- The `F16/BF16/F32` "quants" are produced by the base-conversion step
  itself; if you request the same float type as `--base-type`, the tool
  recognizes the base GGUF *is* that output and won't re-run quantize on it.
- Disk usage can be large. A 7B model in F16 is ~14 GB, plus each quant on
  top. Use `--cleanup-hf` and/or `--cleanup-base` to reclaim space when
  you're done.
- For the lowest-bit IQ quants, generate an imatrix with llama.cpp's
  `llama-imatrix` against a calibration text first, then pass it via
  `--imatrix`.

## License

MIT
