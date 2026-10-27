"""
deploy_to_hf.py — One-shot Hugging Face Spaces deployment script.

What this script does:
  1. Checks prerequisites (git, git-lfs, huggingface_hub installed)
  2. Prompts for your HF username and Space name
  3. Creates the HF Space repo via the HF Hub API
  4. Copies deployment-relevant files to a staging folder
  5. Clones the new Space repo, copies files in, pushes via git + git-lfs

Usage:
    .\\venv\\python.exe deploy_to_hf.py

Prerequisites (run once before this script):
    pip install huggingface_hub
    winget install Git.Git          # if git not installed
    git lfs install                 # run once globally
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Files/dirs to include in the HF Space (relative to PROJECT_ROOT)
INCLUDE_FILES = [
    "app_hf.py",
    "requirements_hf.txt",
    "packages.txt",
    "configs/config.yaml",
    "src/__init__.py",
    "src/config.py",
    "src/gradcam.py",
    "src/models/__init__.py",
    "src/models/attention.py",
    "src/models/backbone.py",
    "src/models/net.py",
    "src/data/__init__.py",
    "src/data/transforms.py",
    "src/data/dataset.py",
    "src/data/dataloaders.py",
    "src/data/discovery.py",
    "src/data/hashing.py",
    "src/data/splitting.py",
    "src/utils/__init__.py",
    "src/utils/logging_utils.py",
    "src/utils/paths.py",
    "src/utils/seed.py",
    "src/ablation.py",
    "src/evaluate.py",
    "src/train.py",
]

# Model checkpoint files (tracked via git-lfs)
CHECKPOINT_PATTERNS = ["best_model_baseline.pt", "best_model_se.pt", "best_model_cbam.pt"]


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, capture_output=False)


def check_prereqs() -> bool:
    ok = True
    for tool in ["git", "git-lfs"]:
        result = subprocess.run([tool, "--version"], capture_output=True)
        if result.returncode != 0:
            print(f"[ERROR] '{tool}' not found. Install it and re-run.")
            ok = False
        else:
            ver = result.stdout.decode().strip().splitlines()[0]
            print(f"  ✓ {ver}")
    try:
        import huggingface_hub  # noqa: F401
        print(f"  ✓ huggingface_hub installed")
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        ok = False
    return ok


def main() -> None:
    print("=" * 60)
    print("  Tomato Disease Intelligence — HF Spaces Deployer")
    print("=" * 60)

    # Check prerequisites
    print("\n[1/6] Checking prerequisites...")
    if not check_prereqs():
        sys.exit(1)

    # Get credentials
    print("\n[2/6] HF Account Setup")
    hf_username = input("  Enter your Hugging Face username: ").strip()
    space_name = input("  Enter Space name (e.g. tomato-disease-detector): ").strip()
    if not space_name:
        space_name = "tomato-disease-detector"

    repo_id = f"{hf_username}/{space_name}"
    print(f"\n  Will deploy to: https://huggingface.co/spaces/{repo_id}")

    # HF login
    print("\n[3/6] Logging in to Hugging Face Hub...")
    print("  (You'll need a token from https://huggingface.co/settings/tokens with WRITE access)")
    from huggingface_hub import HfApi, login, create_repo
    login()  # interactive browser or token prompt

    # Create the Space
    print(f"\n[4/6] Creating HF Space: {repo_id}")
    api = HfApi()
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        print(f"  ✓ Space ready: {url}")
    except Exception as e:
        print(f"[ERROR] Could not create Space: {e}")
        sys.exit(1)

    # Stage files into a temp dir and push
    print("\n[5/6] Staging deployment files...")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        # Clone the empty HF Space repo
        hf_repo_url = f"https://huggingface.co/spaces/{repo_id}"
        run(["git", "clone", hf_repo_url, str(tmp)])

        # Set up git-lfs for large model files
        run(["git", "lfs", "install"], cwd=tmp)

        # Write .gitattributes for LFS tracking of .pt files
        gitattributes = tmp / ".gitattributes"
        gitattributes.write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

        # Write the HF README (with YAML front matter) as README.md
        src_readme = PROJECT_ROOT / "README_HF.md"
        shutil.copy(src_readme, tmp / "README.md")

        # Copy all source files
        for rel in INCLUDE_FILES:
            src = PROJECT_ROOT / rel
            dst = tmp / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"    ✓ {rel}")
            else:
                print(f"    ⚠ MISSING: {rel} (skipped)")

        # Copy checkpoints
        ckpt_dir_src = PROJECT_ROOT / "checkpoints"
        ckpt_dir_dst = tmp / "checkpoints"
        ckpt_dir_dst.mkdir(exist_ok=True)
        for pattern in CHECKPOINT_PATTERNS:
            src_ckpt = ckpt_dir_src / pattern
            if src_ckpt.exists():
                shutil.copy2(src_ckpt, ckpt_dir_dst / pattern)
                print(f"    ✓ checkpoints/{pattern}  ({src_ckpt.stat().st_size // (1024*1024)} MB)")
            else:
                print(f"    ⚠ MISSING checkpoint: {pattern}")

        print("\n[6/6] Committing and pushing to HF Spaces...")
        run(["git", "add", "."], cwd=tmp)
        run(["git", "commit", "-m", "Initial deployment: Tomato Disease Intelligence"], cwd=tmp)
        run(["git", "push"], cwd=tmp)

    print(f"""
{'='*60}
  ✅ Deployment complete!

  Your app is live at:
  👉 https://huggingface.co/spaces/{repo_id}

  Note: HF Spaces takes ~2–3 minutes to build on first deploy.
  Watch the build log at the URL above.
{'='*60}
""")


if __name__ == "__main__":
    main()
