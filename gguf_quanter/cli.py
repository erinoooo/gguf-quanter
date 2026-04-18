"""
gguf-quant: Create GGUF quantizations of HuggingFace models using system
llama.cpp tools (the `llama-quantize` binary and the `convert_hf_to_gguf.py`
script that ships with llama.cpp).

This deliberately does NOT use llama-cpp-python; it shells out to the real
llama.cpp binaries/scripts you already have installed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


# ============================================================
# Quant catalog
# ============================================================

@dataclass(frozen=True)
class QuantType:
    name: str
    bpw: float        # approximate bits-per-weight
    family: str       # 'float' | 'legacy' | 'k' | 'i' | 'ternary'
    description: str
    imatrix_recommended: bool = False


# Order within each family is roughly highest -> lowest precision.
QUANT_TYPES: dict[str, QuantType] = {qt.name: qt for qt in [
    # Float types (passed through llama-quantize as well)
    QuantType("F32",  32.0, "float",  "32-bit float, no quantization"),
    QuantType("F16",  16.0, "float",  "16-bit float"),
    QuantType("BF16", 16.0, "float",  "16-bit bfloat"),

    # Legacy quants
    QuantType("Q8_0", 8.5,  "legacy", "8-bit, near-lossless"),
    QuantType("Q5_1", 5.9,  "legacy", "Legacy 5-bit (Q5_1)"),
    QuantType("Q5_0", 5.5,  "legacy", "Legacy 5-bit (Q5_0)"),
    QuantType("Q4_1", 4.8,  "legacy", "Legacy 4-bit (Q4_1)"),
    QuantType("Q4_0", 4.5,  "legacy", "Legacy 4-bit (Q4_0)"),

    # K-quants
    QuantType("Q6_K",   6.6, "k", "6-bit K-quant, very high quality"),
    QuantType("Q5_K_M", 5.7, "k", "5-bit K-quant medium"),
    QuantType("Q5_K_S", 5.5, "k", "5-bit K-quant small"),
    QuantType("Q4_K_M", 4.8, "k", "4-bit K-quant medium (most popular)"),
    QuantType("Q4_K_S", 4.6, "k", "4-bit K-quant small"),
    QuantType("Q3_K_L", 4.0, "k", "3-bit K-quant large"),
    QuantType("Q3_K_M", 3.7, "k", "3-bit K-quant medium"),
    QuantType("Q3_K_S", 3.4, "k", "3-bit K-quant small"),
    QuantType("Q2_K",   3.0, "k", "2-bit K-quant"),
    QuantType("Q2_K_S", 2.8, "k", "2-bit K-quant small"),

    # I-quants (importance-aware; imatrix helps a lot below ~3 bpw)
    QuantType("IQ4_NL",  4.5,  "i", "4-bit I-quant, non-linear"),
    QuantType("IQ4_XS",  4.3,  "i", "4-bit I-quant, extra small"),
    QuantType("IQ3_M",   3.7,  "i", "3-bit I-quant medium",   imatrix_recommended=True),
    QuantType("IQ3_S",   3.4,  "i", "3-bit I-quant small",    imatrix_recommended=True),
    QuantType("IQ3_XS",  3.3,  "i", "3-bit I-quant XS",       imatrix_recommended=True),
    QuantType("IQ3_XXS", 3.1,  "i", "3-bit I-quant XXS",      imatrix_recommended=True),
    QuantType("IQ2_M",   2.7,  "i", "2-bit I-quant medium",   imatrix_recommended=True),
    QuantType("IQ2_S",   2.5,  "i", "2-bit I-quant small",    imatrix_recommended=True),
    QuantType("IQ2_XS",  2.3,  "i", "2-bit I-quant XS",       imatrix_recommended=True),
    QuantType("IQ2_XXS", 2.1,  "i", "2-bit I-quant XXS",      imatrix_recommended=True),
    QuantType("IQ1_M",   1.75, "i", "1-bit I-quant medium",   imatrix_recommended=True),
    QuantType("IQ1_S",   1.5,  "i", "1-bit I-quant small",    imatrix_recommended=True),

    # Ternary (newer in llama.cpp)
    QuantType("TQ2_0", 2.06, "ternary", "Ternary 2.0 (experimental)"),
    QuantType("TQ1_0", 1.69, "ternary", "Ternary 1.0 (experimental)"),
]}


PRESETS: dict[str, list[str]] = {
    "common":   ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
    "small":    ["Q2_K", "Q3_K_M", "Q4_K_M"],
    "balanced": ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    "quality":  ["Q5_K_M", "Q6_K", "Q8_0"],
    "iquants":  ["IQ2_M", "IQ3_M", "IQ4_XS"],
    "k-only":   ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"],
    "all":      list(QUANT_TYPES.keys()),
}


# ============================================================
# Tool discovery
# ============================================================

@dataclass
class LlamaCppTools:
    quantize_bin: str       # path to llama-quantize
    convert_script: Path    # path to convert_hf_to_gguf.py


def discover_llama_cpp(
    quantize_bin: Optional[str],
    convert_script: Optional[str],
    llama_cpp_dir: Optional[str],
) -> LlamaCppTools:
    """Locate llama.cpp's quantize binary and convert script.

    Search order for the binary:
      1. Explicit --quantize-bin
      2. `llama-quantize` (or `quantize`) in $PATH
      3. Common subpaths inside --llama-cpp-dir (./, ./build/bin)

    Search order for the convert script:
      1. Explicit --convert-script
      2. --llama-cpp-dir / convert_hf_to_gguf.py
      3. Alongside the quantize binary
      4. Common system locations
    """
    # --- binary ---
    qbin: Optional[str] = None
    if quantize_bin:
        if not Path(quantize_bin).exists():
            raise FileNotFoundError(f"--quantize-bin not found: {quantize_bin}")
        qbin = quantize_bin
    else:
        for name in ("llama-quantize", "quantize"):
            found = shutil.which(name)
            if found:
                qbin = found
                break
        if qbin is None and llama_cpp_dir:
            for rel in ("llama-quantize", "quantize",
                        "build/bin/llama-quantize",
                        "build/bin/quantize"):
                cand = Path(llama_cpp_dir) / rel
                if cand.exists():
                    qbin = str(cand)
                    break
    if qbin is None:
        raise FileNotFoundError(
            "Could not find llama-quantize. Install llama.cpp so it's on $PATH, "
            "or pass --quantize-bin / --llama-cpp-dir."
        )

    # --- convert script ---
    cscript: Optional[Path] = None
    if convert_script:
        cscript = Path(convert_script)
        if not cscript.exists():
            raise FileNotFoundError(f"--convert-script not found: {convert_script}")
    else:
        candidates: list[Path] = []
        if llama_cpp_dir:
            candidates.append(Path(llama_cpp_dir) / "convert_hf_to_gguf.py")
        candidates.extend([
            Path(qbin).parent / "convert_hf_to_gguf.py",
            Path(qbin).parent.parent / "convert_hf_to_gguf.py",
            Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
            Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        ])
        for c in candidates:
            if c.exists():
                cscript = c
                break
    if cscript is None:
        raise FileNotFoundError(
            "Could not find convert_hf_to_gguf.py. Pass --llama-cpp-dir pointing "
            "to your llama.cpp source checkout, or --convert-script /path/to/it. "
            "(This script is shipped with llama.cpp's source, not its binaries.)"
        )

    return LlamaCppTools(quantize_bin=qbin, convert_script=cscript)


# ============================================================
# Pipeline steps
# ============================================================

def hf_basename(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def run_streaming(cmd: Sequence[str | Path]) -> int:
    """Run a subprocess inheriting our stdout/stderr."""
    pretty = " ".join(shlex.quote(str(c)) for c in cmd)
    print(f"\n[run] {pretty}", flush=True)
    proc = subprocess.run([str(c) for c in cmd], check=False)
    return proc.returncode


def download_model(
    model_id: str,
    dest_dir: Path,
    revision: Optional[str],
    token: Optional[str],
    allow_patterns: Optional[list[str]],
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for downloading. Install with:\n"
            "  pip install huggingface_hub"
        ) from e

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {model_id} -> {dest_dir}")
    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(dest_dir),
        token=token,
        allow_patterns=allow_patterns,
    )
    return Path(path)


def convert_to_base_gguf(
    tools: LlamaCppTools,
    hf_model_dir: Path,
    output_path: Path,
    base_type: str,
    python_exe: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str | Path] = [
        python_exe,
        tools.convert_script,
        hf_model_dir,
        "--outfile", output_path,
        "--outtype", base_type.lower(),
    ]
    rc = run_streaming(cmd)
    if rc != 0:
        raise RuntimeError(f"convert_hf_to_gguf.py failed with exit code {rc}")
    if not output_path.exists():
        raise RuntimeError(f"Conversion did not produce {output_path}")
    return output_path


def quantize_one(
    tools: LlamaCppTools,
    base_gguf: Path,
    output_path: Path,
    quant_type: str,
    threads: Optional[int],
    imatrix: Optional[Path],
    allow_requantize: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str | Path] = [tools.quantize_bin]
    if allow_requantize:
        cmd.append("--allow-requantize")
    if imatrix:
        cmd.extend(["--imatrix", imatrix])
    cmd.extend([base_gguf, output_path, quant_type])
    if threads is not None:
        cmd.append(str(threads))
    rc = run_streaming(cmd)
    if rc != 0:
        raise RuntimeError(f"llama-quantize failed for {quant_type} (exit {rc})")
    if not output_path.exists():
        raise RuntimeError(f"Quantization did not produce {output_path}")
    return output_path


# ============================================================
# CLI
# ============================================================

def parse_quants(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token.lower().startswith("preset:"):
                name = token.split(":", 1)[1].strip().lower()
                if name not in PRESETS:
                    raise SystemExit(
                        f"Unknown preset '{name}'. "
                        f"Available: {', '.join(sorted(PRESETS))}"
                    )
                out.extend(PRESETS[name])
            else:
                out.append(token.upper())
    for q in out:
        if q not in QUANT_TYPES:
            raise SystemExit(
                f"Unknown quant type '{q}'. "
                f"Run `gguf-quant list-quants` to see options."
            )
    # de-dupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for q in out:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    if not uniq:
        raise SystemExit("No quant types specified.")
    return uniq


def cmd_list_quants(_args: argparse.Namespace) -> int:
    print("Available quant types (* = imatrix recommended for best quality):\n")
    family_titles = [
        ("float",   "Float types"),
        ("legacy",  "Legacy quants"),
        ("k",       "K-quants"),
        ("i",       "I-quants (importance-aware)"),
        ("ternary", "Ternary (experimental)"),
    ]
    for family, title in family_titles:
        members = [qt for qt in QUANT_TYPES.values() if qt.family == family]
        if not members:
            continue
        print(f"  {title}:")
        for qt in members:
            mark = "*" if qt.imatrix_recommended else " "
            print(f"   {mark} {qt.name:<10} ~{qt.bpw:>5.2f} bpw   {qt.description}")
        print()
    print("Presets (use as `preset:NAME`):")
    for name, quants in PRESETS.items():
        joined = " ".join(quants) if name != "all" else f"<all {len(quants)} types>"
        print(f"  preset:{name:<10} = {joined}")
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    quants = parse_quants(args.quants)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = args.basename or hf_basename(args.model)
    hf_dir = output_dir / f"{basename}-hf"
    base_type_upper = args.base_type.upper()
    base_gguf = output_dir / f"{basename}-{base_type_upper}.gguf"

    tools = discover_llama_cpp(
        quantize_bin=args.quantize_bin,
        convert_script=args.convert_script,
        llama_cpp_dir=args.llama_cpp_dir,
    )
    print(f"[info] llama-quantize: {tools.quantize_bin}")
    print(f"[info] convert script: {tools.convert_script}")
    print(f"[info] output dir:     {output_dir}")
    print(f"[info] basename:       {basename}")
    print(f"[info] base type:      {base_type_upper}")
    print(f"[info] quants:         {' '.join(quants)}")

    # ---- 1. Download ----
    if args.skip_download:
        if not hf_dir.exists():
            raise SystemExit(f"--skip-download given but {hf_dir} does not exist.")
        print(f"[skip] download (using existing {hf_dir})")
    else:
        already = hf_dir.exists() and any(hf_dir.iterdir())
        if already and not args.force_download:
            print(f"[skip] download ({hf_dir} exists; --force-download to redo)")
        else:
            allow_patterns = None
            if args.prefer_safetensors:
                # Skip pytorch .bin and consolidated .pth shards if safetensors
                # are present. We keep config/tokenizer/etc.
                allow_patterns = [
                    "*.safetensors", "*.json", "*.model", "*.txt",
                    "*.tiktoken", "tokenizer.*", "*.py",
                ]
            download_model(
                model_id=args.model,
                dest_dir=hf_dir,
                revision=args.revision,
                token=args.hf_token,
                allow_patterns=allow_patterns,
            )

    # ---- 2. Convert to base GGUF ----
    if base_gguf.exists() and not args.force_convert:
        print(f"[skip] base conversion ({base_gguf.name} exists, "
              f"{human_size(base_gguf.stat().st_size)})")
    else:
        convert_to_base_gguf(
            tools=tools,
            hf_model_dir=hf_dir,
            output_path=base_gguf,
            base_type=args.base_type,
            python_exe=args.python or sys.executable,
        )
        print(f"[done] base gguf: {base_gguf.name} "
              f"({human_size(base_gguf.stat().st_size)})")

    # ---- 3. Quantize ----
    imatrix_path: Optional[Path] = None
    if args.imatrix:
        imatrix_path = Path(args.imatrix).expanduser().resolve()
        if not imatrix_path.exists():
            raise SystemExit(f"--imatrix file not found: {imatrix_path}")
        print(f"[info] imatrix:        {imatrix_path}")

    successes: list[tuple[str, Path, float]] = []  # (quant, path, seconds)
    failures: list[tuple[str, str]] = []

    for q in quants:
        out_path = output_dir / f"{basename}-{q}.gguf"

        # Special case: requested quant is identical to the base type — the
        # base GGUF *is* that output, no need to re-run llama-quantize.
        if q == base_type_upper and out_path == base_gguf:
            print(f"[skip] {q} (already produced as the base GGUF)")
            successes.append((q, out_path, 0.0))
            continue

        if out_path.exists() and not args.overwrite:
            print(f"[skip] {q}: {out_path.name} exists (use --overwrite to redo)")
            successes.append((q, out_path, 0.0))
            continue

        print(f"\n[quantize] {q} -> {out_path.name}")
        t0 = time.time()
        try:
            quantize_one(
                tools=tools,
                base_gguf=base_gguf,
                output_path=out_path,
                quant_type=q,
                threads=args.threads,
                imatrix=imatrix_path,
                allow_requantize=args.allow_requantize,
            )
            dt = time.time() - t0
            sz = out_path.stat().st_size
            print(f"[done] {q}: {out_path.name} "
                  f"({human_size(sz)}, {dt:.1f}s)")
            successes.append((q, out_path, dt))
        except Exception as e:
            print(f"[error] {q} failed: {e}", file=sys.stderr)
            failures.append((q, str(e)))
            if args.fail_fast:
                break

    # ---- 4. Cleanup ----
    if args.cleanup_hf and hf_dir.exists():
        print(f"\n[cleanup] removing HF source dir {hf_dir}")
        shutil.rmtree(hf_dir, ignore_errors=True)

    if args.cleanup_base:
        produced_paths = {p for _, p, _ in successes}
        if base_gguf in produced_paths:
            print(f"[cleanup] keeping base {base_gguf.name} "
                  f"(it was a requested quant)")
        elif base_gguf.exists():
            print(f"[cleanup] removing base {base_gguf.name}")
            try:
                base_gguf.unlink()
            except OSError as e:
                print(f"[warn] could not delete base gguf: {e}", file=sys.stderr)

    # ---- 5. Summary ----
    print("\n=== Summary ===")
    print(f"Output dir: {output_dir}")
    if successes:
        print(f"Succeeded ({len(successes)}):")
        for q, p, dt in successes:
            sz = human_size(p.stat().st_size) if p.exists() else "missing"
            extra = f", {dt:.1f}s" if dt > 0 else ""
            print(f"  {q:<10}  {p.name}  ({sz}{extra})")
    if failures:
        print(f"\nFailed ({len(failures)}):")
        for q, err in failures:
            print(f"  {q:<10}  {err}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gguf-quant",
        description=(
            "Create GGUF quantizations of a HuggingFace model using your "
            "system-installed llama.cpp tools. Pulls from HF, converts to a "
            "base GGUF, then runs llama-quantize for each requested type."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  gguf-quant quantize meta-llama/Llama-3.2-3B-Instruct -q preset:common\n"
            "  gguf-quant quantize Qwen/Qwen2.5-7B -q Q4_K_M Q5_K_M Q8_0 -o ./out\n"
            "  gguf-quant quantize mistralai/Mistral-7B-v0.3 -q preset:all --cleanup-hf\n"
            "  gguf-quant list-quants\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # list-quants
    p_list = sub.add_parser(
        "list-quants",
        help="List supported quant types and presets, then exit.",
    )
    p_list.set_defaults(func=cmd_list_quants)

    # quantize
    p_q = sub.add_parser(
        "quantize",
        help="Download an HF model, convert it to GGUF, and produce one or "
             "more quantizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_q.add_argument(
        "model",
        help="HuggingFace model ID, e.g. meta-llama/Llama-3.2-3B-Instruct.",
    )
    p_q.add_argument(
        "-q", "--quants", nargs="+", required=True,
        metavar="QUANT",
        help="Quant types to produce. Accepts type names (Q4_K_M, IQ3_M, ...) "
             "and 'preset:NAME' (preset:common, preset:all, ...). Can be "
             "comma-separated.",
    )
    p_q.add_argument(
        "-o", "--output-dir", default="./quants",
        help="Where to put the downloaded model, base GGUF, and quants.",
    )
    p_q.add_argument(
        "--basename", default=None,
        help="Override the output filename stem. Defaults to the model name "
             "portion of the HF ID.",
    )
    p_q.add_argument(
        "--base-type", default="f16", choices=["f16", "bf16", "f32"],
        help="Numeric type for the base GGUF used as input to llama-quantize.",
    )

    # HF download options
    p_q.add_argument(
        "--revision", default=None,
        help="HF revision/branch/tag/commit to download.",
    )
    p_q.add_argument(
        "--hf-token", default=os.environ.get("HF_TOKEN")
                              or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HuggingFace token for gated/private models. "
             "Defaults to $HF_TOKEN or $HUGGING_FACE_HUB_TOKEN.",
    )
    p_q.add_argument(
        "--prefer-safetensors", action="store_true",
        help="When downloading, request only .safetensors weights and skip "
             "pytorch_model.bin / consolidated.pth duplicates.",
    )

    # Quantize options
    p_q.add_argument(
        "--imatrix", default=None,
        help="Optional importance matrix .dat file. Strongly recommended "
             "for IQ-quants at 3 bpw and below.",
    )
    p_q.add_argument(
        "-t", "--threads", type=int, default=None,
        help="Threads passed to llama-quantize (default: tool's own default).",
    )
    p_q.add_argument(
        "--allow-requantize", action="store_true",
        help="Pass --allow-requantize to llama-quantize (needed if the base "
             "GGUF is itself already quantized, e.g. Q8_0 -> Q4_K_M).",
    )

    # Resume / overwrite controls
    p_q.add_argument(
        "--overwrite", action="store_true",
        help="Re-quantize even if the output file already exists.",
    )
    p_q.add_argument(
        "--skip-download", action="store_true",
        help="Don't touch HF; assume the source is already at "
             "<output-dir>/<basename>-hf.",
    )
    p_q.add_argument(
        "--force-download", action="store_true",
        help="Re-download from HF even if the local copy exists.",
    )
    p_q.add_argument(
        "--force-convert", action="store_true",
        help="Re-run base-GGUF conversion even if the base file exists.",
    )

    # Cleanup
    p_q.add_argument(
        "--cleanup-hf", action="store_true",
        help="Delete the downloaded HF source directory after quantizing.",
    )
    p_q.add_argument(
        "--cleanup-base", action="store_true",
        help="Delete the base GGUF after all requested quants are produced "
             "(skipped if it is itself one of the requested quants).",
    )

    # Errors
    p_q.add_argument(
        "--fail-fast", action="store_true",
        help="Stop on the first failed quant instead of continuing.",
    )

    # llama.cpp tool locations
    p_q.add_argument(
        "--quantize-bin", default=None,
        help="Path to the llama-quantize binary. Default: search $PATH.",
    )
    p_q.add_argument(
        "--convert-script", default=None,
        help="Path to convert_hf_to_gguf.py. Default: search common locations.",
    )
    p_q.add_argument(
        "--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR"),
        help="Path to a llama.cpp source checkout (helps find the convert "
             "script). Defaults to $LLAMA_CPP_DIR.",
    )
    p_q.add_argument(
        "--python", default=None,
        help="Python interpreter used to run convert_hf_to_gguf.py. "
             "Defaults to the current interpreter.",
    )

    p_q.set_defaults(func=cmd_quantize)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[abort] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
