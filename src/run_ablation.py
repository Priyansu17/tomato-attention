"""Resumable ablation runner: trains baseline -> SE -> CBAM sequentially.

Usage (smoke test â€” safe to run any time):
    python -m src.run_ablation --smoke-test

Usage (full training â€” only with explicit user approval):
    python -m src.run_ablation

Resume behaviour:
    If results/metrics_test_<variant>.json already exists the variant is
    skipped.  Delete the file to force a re-run.

Log files:
    outputs/logs/ablation_<variant>.log  â€” stdout + stderr for each variant
    outputs/logs/ablation_summary.log    â€” one-liner per variant when done
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


# Main log file for the runner's own output (set by main())
_runner_log_fh = None


def _log(msg: str) -> None:
    """Write msg to stdout and to the runner's own log file."""
    print(msg, flush=True)
    if _runner_log_fh is not None:
        try:
            _runner_log_fh.write(msg + "\n")
            _runner_log_fh.flush()
        except Exception:
            pass


def _run_step(
    cmd: list[str],
    label: str,
    log_file: Path,
    append: bool = False,
) -> int:
    """Run *cmd*, tee output to *log_file*, return exit code.

    Uses PYTHONUTF8=1 in the child env and text mode with utf-8 + replace so
    the runner is safe on Windows CP1252 consoles and in headless/detached mode.
    """
    import os

    mode = "a" if append else "w"
    header = f"\n{'='*70}\n{label}\ncmd: {' '.join(cmd)}\n{'='*70}\n"

    _log(f"\n{'='*70}")
    _log(f"[{datetime.now().strftime('%H:%M:%S')}] {label}")
    _log(f"  cmd : {' '.join(cmd)}")
    _log(f"  log : {log_file}")
    _log(f"{'='*70}")

    # Propagate UTF-8 mode to the child process
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"

    with log_file.open(mode, encoding="utf-8", errors="replace") as fh:
        fh.write(header)
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            # Also mirror to our own log / stdout
            _log(line.rstrip())
        proc.wait()
        rc = proc.returncode
        fh.write(f"\n[exit code: {rc}]\n")

    return rc


def _append_summary(line: str) -> None:
    with _summary_path().open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    global _runner_log_fh

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
    parser.add_argument(
        "--log-file", default=None,
        help=(
            "Path for runner's own log (default: outputs/ablation_run.log). "
            "Set to '-' to suppress."
        ),
    )
    args = parser.parse_args(argv)

    _ensure_dirs()

    # Open the runner's own log file (so it works in detached/headless mode)
    log_file_path = args.log_file or str(OUTPUTS_DIR / "ablation_run.log")
    if log_file_path != "-":
        try:
            _runner_log_fh = open(log_file_path, "a", encoding="utf-8", errors="replace")  # noqa: WPS515
        except Exception as exc:
            print(f"[WARN] Could not open runner log: {exc}")

    python = sys.executable
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
        variant_log = _log_path(variant)

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
            _log(f"\n{msg}")
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
        rc = _run_step(train_cmd, f"TRAIN  variant={variant}", variant_log, append=False)
        if rc != 0:
            msg = f"[FAIL]  {variant}: train exited with code {rc}"
            _log(msg)
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
        rc = _run_step(eval_cmd, f"EVAL   variant={variant}  split=test", variant_log, append=True)
        if rc != 0:
            msg = f"[FAIL]  {variant}: evaluate exited with code {rc}"
            _log(msg)
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
        rc = _run_step(gradcam_cmd, f"GRADCAM variant={variant}  split=val", variant_log, append=True)
        if rc != 0:
            msg = f"[WARN]  {variant}: gradcam exited with code {rc} (non-fatal)"
            _log(msg)
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

        msg = f"[OK]    {variant}: {metric_str}  elapsed={elapsed/60:.1f} min"
        _log(f"\n{msg}")
        _append_summary(msg)
        summary_lines.append(msg)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    total_elapsed = time.time() - overall_start
    _log(f"\n{'='*70}")
    _log(f"Ablation run complete in {total_elapsed/60:.1f} min")
    for line in summary_lines:
        _log(f"  {line}")
    if failed:
        _log(f"\nFailed variants: {failed}")
        if _runner_log_fh:
            _runner_log_fh.close()
        return 1
    _append_summary(
        f"--- ablation run finished {datetime.now().isoformat()} "
        f"total={total_elapsed/60:.1f} min ---"
    )
    if _runner_log_fh:
        _runner_log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
