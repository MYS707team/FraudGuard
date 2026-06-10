"""
FraudGuard — Train, save weights, and push to GitHub (one command).

Run this in a shell and it will:

  1. (optional) Generate the dataset if ``data/raw/creditcard.csv`` is missing.
  2. Train & compare all models (``src.train.run``) and save the artifacts:
       models/best_model.pkl         winning model weights
       models/scaler.pkl             fitted StandardScaler
       models/model_metadata.json    best model name, threshold, metrics
       models/comparison_results.json   metrics for every model
  3. Regenerate the evaluation plots (confusion matrix + PR curve).
  4. Commit those trained-model artifacts (force-added past .gitignore) and
     ``git push`` them to GitHub.

Usage
-----
    python scripts/train_and_push.py                 # train, save, commit, push
    python scripts/train_and_push.py --no-push       # train & commit only
    python scripts/train_and_push.py --no-commit     # train & save only
    python scripts/train_and_push.py --branch main   # push to a specific branch
    python scripts/train_and_push.py --remote origin --message "retrain"

Notes
-----
* The push uses your existing git credentials (the same ones ``git push`` uses
  on your machine / Colab). Configure them once with a Personal Access Token or
  SSH key before running.
* Model artifacts are git-ignored by default; this script force-adds them on
  purpose so the trained weights land in the repository.
"""
from __future__ import annotations

# --- import bootstrap --------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# -----------------------------------------------------------------------------

import argparse
import json
import subprocess
from datetime import datetime, timezone

from src import config


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a git command, echoing it, and return the completed process."""
    print(f"  $ git {' '.join(args)}")
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True
    )


def ensure_dataset() -> None:
    """Generate the synthetic dataset if the raw CSV is not present."""
    if config.RAW_DATA_PATH.exists():
        return
    print("Raw dataset missing -> generating a synthetic one ...")
    from scripts.generate_dataset import generate_dataset

    config.ensure_dirs()
    df = generate_dataset()
    df.to_csv(config.RAW_DATA_PATH, index=False, float_format="%.6f")
    print(f"  saved -> {config.RAW_DATA_PATH}")


def train_and_save() -> tuple[str, dict]:
    """Run the training pipeline and regenerate plots. Returns (best_name, meta)."""
    from src.train import run as train_run

    best_name = train_run()

    # Regenerate evaluation plots for the saved best model (best effort).
    try:
        from src.evaluate import generate_plots_for_best_model

        cm_path, pr_path = generate_plots_for_best_model()
        print(f"Plots saved: {cm_path.name}, {pr_path.name}")
    except Exception as exc:  # plots are non-critical
        print(f"[warn] Could not generate plots: {exc}")

    meta = {}
    if config.METADATA_PATH.exists():
        meta = json.loads(config.METADATA_PATH.read_text())
    return best_name, meta


def list_artifacts() -> list[Path]:
    """Return the trained-model artifact paths that currently exist."""
    candidates = [
        config.BEST_MODEL_PATH,
        config.SCALER_PATH,
        config.METADATA_PATH,
        config.COMPARISON_PATH,
        config.CONFUSION_MATRIX_PATH,
        config.PR_CURVE_PATH,
    ]
    return [p for p in candidates if p.exists()]


def commit_and_push(
    best_name: str,
    meta: dict,
    *,
    do_commit: bool,
    do_push: bool,
    remote: str,
    branch: str | None,
    message: str | None,
) -> None:
    """Force-add the model artifacts, commit, and push to GitHub."""
    repo_root = config.PROJECT_ROOT
    if not (repo_root / ".git").exists():
        print(f"[skip] {repo_root} is not a git repository; nothing to push.")
        return

    artifacts = list_artifacts()
    if not artifacts:
        print("[skip] No trained artifacts found to commit.")
        return

    rel_paths = [str(p.relative_to(repo_root)) for p in artifacts]
    print("\nStaging trained-model artifacts (force-add past .gitignore):")
    for p in rel_paths:
        print(f"  + {p}")
    _run_git(["add", "-f", *rel_paths], repo_root)

    if not do_commit:
        print("[--no-commit] Artifacts staged but not committed.")
        return

    # Skip the commit if nothing actually changed.
    if _run_git(["diff", "--cached", "--quiet"], repo_root).returncode == 0:
        print("[skip] No changes in trained artifacts; nothing to commit.")
        return

    pr_auc = meta.get("metrics", {}).get("pr_auc", "n/a")
    recall = meta.get("metrics", {}).get("recall", "n/a")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = message or (
        f"Train models: best={best_name} (PR-AUC={pr_auc}, recall={recall}) [{stamp}]"
    )
    commit = _run_git(["commit", "-m", msg], repo_root)
    print(commit.stdout or commit.stderr)
    if commit.returncode != 0:
        print("[error] git commit failed; aborting push.")
        return

    if not do_push:
        print("[--no-push] Committed locally; skipping push.")
        return

    if branch is None:
        head = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        branch = head.stdout.strip() or "main"

    print(f"\nPushing to {remote}/{branch} ...")
    push = _run_git(["push", remote, f"HEAD:{branch}"], repo_root)
    out = (push.stdout + push.stderr).strip()
    print(out)
    if push.returncode != 0:
        print(
            "[error] git push failed. Check your credentials / remote and retry, "
            "or run with --no-push and push manually."
        )
        sys.exit(push.returncode)
    print(f"\nDone. Trained model weights pushed to {remote}/{branch}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FraudGuard models, save weights, and push to GitHub."
    )
    parser.add_argument("--no-commit", action="store_true", help="train & save only")
    parser.add_argument("--no-push", action="store_true", help="train & commit only")
    parser.add_argument("--remote", default="origin", help="git remote (default: origin)")
    parser.add_argument(
        "--branch", default=None, help="branch to push to (default: current branch)"
    )
    parser.add_argument("--message", default=None, help="custom commit message")
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="do not auto-generate the dataset if it is missing",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FraudGuard - train, save weights, and push")
    print("=" * 60)

    if not args.skip_generate:
        ensure_dataset()

    best_name, meta = train_and_save()

    commit_and_push(
        best_name,
        meta,
        do_commit=not args.no_commit,
        do_push=not args.no_push,
        remote=args.remote,
        branch=args.branch,
        message=args.message,
    )


if __name__ == "__main__":
    main()
