"""Resumable ablation runner: trains baseline -> SE -> CBAM sequentially.

Usage (smoke test — safe to run any time):
    python -m src.run_ablation --smoke-test

Usage (full training — only with explicit user approval):
    python -m src.run_ablation

Resume behaviour:
    If results/metrics_test_<variant>.json already exists the variant is
    skipped.  Delete the file to force a re-run.

Log files:
    outputs/logs/ablation_<variant>.log  — stdout + stderr for each variant
    outputs/logs/ablation_summary.log    — one-liner per variant when done
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANTS = ["baseline", "se", "cbam"]

RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")
LOGS_DIR = OUTPUTS_DIR / "logs"


def _metrics_path(variant: str, split: str = "test") -> Path:
    return RESULTS_DIR / f"metrics_{split}_{variant}.json"


def _log_path(variant: str) -> Path:
    return LOGS_DIR / f"ablation_{variant}.log"


def _summary_path() -> Path:
    return LOGS_DIR / "ablation_summary.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _run_step(
    cmd: list[str],
    label: str,
    log_file: Path,
    append: bool = False,
) -> int:
    """Run *cmd*, tee output to *log_file*, return exit code.

    Uses raw byte I/O on stdout so Windows CP1252 consoles do not crash on
    non-ASCII characters.  Sets PYTHONUTF8=1 in the child process environment
    so the child itself emits clean UTF-8.
    """
    import os

    mode = "ab" if append else "wb"
    header = (
        f"\n{'='*70}\n{label}\ncmd: {' '.join(cmd)}\n{'='*70}\n"
    ).encode("utf-8")

    _print_ascii(f"\n{'='*70}")
    _print_ascii(f"[{datetime.now().strftime('%H:%M:%S')}] {label}")
    _print_ascii(f"  cmd : {' '.join(cmd)}")
    _print_ascii(f"  log : {log_file}")
    _print_ascii(f"{'='*70}")

    # Propagate UTF-8 mode to the child process
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"

    with log_file.open(mode) as fh:
        fh.write(header)
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
        for raw_line in proc.stdout:
            # Write raw bytes to log file
            fh.write(raw_line)
            fh.flush()
            # Write safely to console (replace unencodable chars)
            try:
                sys.stdout.buffer.write(raw_line)
                sys.stdout.buffer.flush()
            except Exception:
                sys.stdout.write(raw_line.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        proc.wait()
        rc = proc.returncode
        fh.write(f"\n[exit code: {rc}]\n".encode("utf-8"))

    return rc


def _print_ascii(msg: str) -> None:
    """Print ``msg`` safely to stdout, replacing non-ASCII chars."""
    try:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def _append_summary(line: str) -> None:
    with _summary_path().open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable ablation runner: trains baseline, SE and CBAM with "
            "identical settings.  Skip variants whose metrics file exists."
        )
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Smoke-test mode: tiny subset, 1 epoch.  Safe to run any time.",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml).",
    )
    parser.add_argument(
        "--variants", nargs="+", default=VARIANTS,
        choices=VARIANTS,
        help="Which variants to run (default: all three).",
    )
    args = parser.parse_args(argv)

    _ensure_dirs()

    python = sys.executable  # always use the same interpreter as this script
    smoke_flag = ["--smoke-test"] if args.smoke_test else []
    summary_lines: list[str] = []

    _append_summary(
        f"\n--- ablation run started {datetime.now().isoformat()} "
        f"({'smoke' if args.smoke_test else 'full'}) ---"
    )

    overall_start = time.time()
    failed: list[str] = []

    for variant in args.variants:
        metrics_file = _metrics_path(variant)
        log_file = _log_path(variant)

        # ----------------------------------------------------------------
        # Resume check
        # ----------------------------------------------------------------
        if metrics_file.exists() and not args.smoke_test:
            try:
                m = json.loads(metrics_file.read_text(encoding="utf-8"))
                acc = m.get("accuracy", "?")
                f1 = m.get("f1_macro", "?")
                msg = (
                    f"[SKIP] {variant}: metrics file exists "
                    f"(accuracy={acc:.4f}, f1_macro={f1:.4f})"
                )
            except Exception:
                msg = f"[SKIP] {variant}: metrics file exists (could not parse)"
            print(f"\n{msg}", flush=True)
            _append_summary(msg)
            summary_lines.append(msg)
            continue

        t0 = time.time()

        # ----------------------------------------------------------------
        # Step 1: Train
        # ----------------------------------------------------------------
        train_cmd = [
            python, "-m", "src.train",
            "--config", args.config,
            "--attention", variant,
            "--run-name", variant,
            *smoke_flag,
        ]
        rc = _run_step(
            train_cmd,
            f"TRAIN  variant={variant}",
            log_file,
            append=False,
        )
        if rc != 0:
            msg = f"[FAIL]  {variant}: train exited with code {rc}"
            print(msg, flush=True)
            _append_summary(msg)
            failed.append(variant)
            continue

        # ----------------------------------------------------------------
        # Step 2: Evaluate on test split
        # ----------------------------------------------------------------
        ckpt = f"checkpoints/best_model_{variant}.pt"
        eval_cmd = [
            python, "-m", "src.evaluate",
            "--config", args.config,
            "--checkpoint", ckpt,
            "--split", "test",
            "--run-name", variant,
        ]
        rc = _run_step(
            eval_cmd,
            f"EVAL   variant={variant}  split=test",
            log_file,
            append=True,
        )
        if rc != 0:
            msg = f"[FAIL]  {variant}: evaluate exited with code {rc}"
            print(msg, flush=True)
            _append_summary(msg)
            failed.append(variant)
            continue

        # ----------------------------------------------------------------
        # Step 3: Grad-CAM on val split
        # ----------------------------------------------------------------
        gradcam_cmd = [
            python, "-m", "src.gradcam",
            "--config", args.config,
            "--checkpoint", ckpt,
            "--split", "val",
            "--run-name", variant,
        ]
        rc = _run_step(
            gradcam_cmd,
            f"GRADCAM variant={variant}  split=val",
            log_file,
            append=True,
        )
        if rc != 0:
            # Grad-CAM failure is non-fatal; log and continue
            msg = f"[WARN]  {variant}: gradcam exited with code {rc} (non-fatal)"
            print(msg, flush=True)
            _append_summary(msg)

        # ----------------------------------------------------------------
        # Summary for this variant
        # ----------------------------------------------------------------
        elapsed = time.time() - t0
        try:
            m = json.loads(metrics_file.read_text(encoding="utf-8"))
            acc = m.get("accuracy", "?")
            f1 = m.get("f1_macro", "?")
            metric_str = f"accuracy={acc:.4f}  f1_macro={f1:.4f}"
        except Exception:
            metric_str = "metrics unavailable"

        msg = (
            f"[OK]    {variant}: {metric_str}  elapsed={elapsed/60:.1f} min"
        )
        print(f"\n{msg}", flush=True)
        _append_summary(msg)
        summary_lines.append(msg)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    total_elapsed = time.time() - overall_start
    print(f"\n{'='*70}", flush=True)
    print(f"Ablation run complete in {total_elapsed/60:.1f} min", flush=True)
    for line in summary_lines:
        print(f"  {line}", flush=True)
    if failed:
        print(f"\nFailed variants: {failed}", flush=True)
        return 1
    _append_summary(
        f"--- ablation run finished {datetime.now().isoformat()} "
        f"total={total_elapsed/60:.1f} min ---"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
