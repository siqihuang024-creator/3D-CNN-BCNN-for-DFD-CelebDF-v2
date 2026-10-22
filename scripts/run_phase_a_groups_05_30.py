"""Run the remaining fixed CelebDF Phase-A groups sequentially.

Run from the project root after loading remote_env.sh. Completed groups are
skipped on restart; a partial run stops the queue for manual inspection.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/v2/phase_a_celeb_video_disjoint.yaml"
MATRIX = "configs/v2/experiment_matrix.yaml"
MANIFEST_DIR = "artifacts/manifests/celeb_video_disjoint_seed42"
RUN_DIR = "artifacts/v2_video_disjoint"
EPOCHS = 100


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _check_protocol():
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    matrix = yaml.safe_load((ROOT / MATRIX).read_text(encoding="utf-8"))
    train = config["train"]
    checks = {
        "epochs": EPOCHS,
        "early_stopping_enabled": False,
        "epoch_protocol": "balanced_pool_pass",
        "physical_batch_size": 8,
        "precision": "fp32",
        "gradient_clip_norm": None,
    }
    for key, expected in checks.items():
        if train.get(key) != expected:
            raise RuntimeError(
                "Protocol config {} must be {!r}, found {!r}.".format(
                    key, expected, train.get(key)))
    if matrix["E2_celeb"]["train"].get("physical_batch_size") != 8:
        raise RuntimeError("E2_celeb must keep physical batch size 8.")
    summary = _read_json(ROOT / MANIFEST_DIR / "summary.json")
    if summary.get("groups") != 60 or summary.get("counts", {}).get("per_group_fake") != 616:
        raise RuntimeError("The precomputed 60-group manifest summary has changed.")


def _paths(group):
    manifest = "{}/group_{:02d}.csv".format(MANIFEST_DIR, group)
    run_dir = "{}/e2_group{:02d}_formal".format(RUN_DIR, group)
    return manifest, run_dir


def _command(group):
    manifest, run_dir = _paths(group)
    return [
        sys.executable, "pretrain_extractor.py",
        "--config", CONFIG,
        "--manifest", manifest,
        "--experiment", "E2_celeb",
        "--seed", str(41 + group),
        "--run-dir", run_dir,
    ]


def _display_command(command):
    return " ".join(shlex.quote(part) for part in command)


def _complete(group):
    manifest, run_dir_name = _paths(group)
    run_dir = ROOT / run_dir_name
    if not run_dir.exists():
        return False
    if not run_dir.is_dir():
        raise RuntimeError("{} exists but is not a directory.".format(run_dir))
    required = [
        run_dir / "logs" / "history.json",
        run_dir / "checkpoints" / "best.pt",
        run_dir / "checkpoints" / "last.pt",
        run_dir / "config.json",
        run_dir / "runtime.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Group {:02d} has an incomplete run directory; missing: {}."
                           .format(group, ", ".join(missing)))
    try:
        history = _read_json(required[0])
        config = _read_json(required[3])
        runtime = _read_json(required[4])
    except (OSError, ValueError) as exc:
        raise RuntimeError("Group {:02d} has unreadable run metadata: {}."
                           .format(group, exc)) from exc
    if (not isinstance(history, list) or len(history) != EPOCHS or
            [row.get("epoch") for row in history] != list(range(1, EPOCHS + 1))):
        raise RuntimeError("Group {:02d} did not finish all {} epochs."
                           .format(group, EPOCHS))
    if (config.get("seed") != 41 + group or
            config.get("train", {}).get("epochs") != EPOCHS or
            config.get("train", {}).get("early_stopping_enabled") is not False or
            runtime.get("manifest") != manifest or
            runtime.get("experiment") != "E2_celeb"):
        raise RuntimeError("Group {:02d} metadata does not match the fixed protocol."
                           .format(group))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-group", type=int, default=5)
    parser.add_argument("--end-group", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true",
                        help="Check the queue and show commands without training.")
    args = parser.parse_args()
    if not 5 <= args.start_group <= args.end_group <= 30:
        parser.error("Choose a contiguous group range within 05–30.")
    _check_protocol()
    groups = range(args.start_group, args.end_group + 1)
    status = {}
    for group in groups:
        manifest, _ = _paths(group)
        if not (ROOT / manifest).is_file():
            raise RuntimeError("Missing group manifest: {}".format(manifest))
        status[group] = "complete" if _complete(group) else "pending"
    for group in groups:
        command = _command(group)
        print("Group {:02d}/30: {}".format(group, status[group]), flush=True)
        if status[group] == "complete":
            continue
        if args.dry_run:
            print("  {}".format(_display_command(command)), flush=True)
            continue
        print("  starting: {}".format(_display_command(command)), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError("Group {:02d} failed with exit code {}; stopping."
                               .format(group, result.returncode))
        if not _complete(group):
            raise RuntimeError("Group {:02d} returned success without complete results."
                               .format(group))
        print("Group {:02d}/30: finished all {} epochs.".format(group, EPOCHS),
              flush=True)
    if args.dry_run:
        print("Dry run complete; no training or test evaluation started.", flush=True)
    else:
        print("Requested group range finished; no test set was evaluated.",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
