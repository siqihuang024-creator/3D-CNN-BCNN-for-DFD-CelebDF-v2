"""Small editable PyCharm launcher for the V2 protocol."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Change only these values, then press Run in PyCharm.
TASK = "phase_a"  # build_manifest | phase_a | phase_c | evaluate
DATASET = "celeb"  # celeb | dfd
EXPERIMENT = "E4_gap"
TEMPORAL_AGGREGATION = "gap"  # replace with the selected E4-prime winner for E7
SEED = 42
MANIFEST = ROOT / "artifacts" / "manifests" / "combined_manifest_p05.csv"
DATASET_TAG = "celebdfv3" if DATASET == "celeb" else "dfd"
SUFFIX = EXPERIMENT.lower()
if not SUFFIX.endswith("_" + TEMPORAL_AGGREGATION):
    SUFFIX += "_" + TEMPORAL_AGGREGATION
PHASE_A_CHECKPOINT = ROOT / "artifacts" / "v2" / "{}_{}_seed{}".format(
    DATASET_TAG, SUFFIX, SEED) / "checkpoints" / "best.pt"
PHASE_C_CHECKPOINT = ROOT / "artifacts" / "v2" / "{}_e6_{}_seed{}".format(
    DATASET_TAG, SUFFIX, SEED) / "checkpoints" / "best.pt"


def run(arguments):
    print("Running:", " ".join(str(value) for value in arguments))
    subprocess.run([str(value) for value in arguments], cwd=str(ROOT), check=True)


def main():
    phase_a_config = ROOT / "configs" / "v2" / "phase_a_{}.yaml".format(DATASET)
    phase_c_config = ROOT / "configs" / "v2" / "phase_c_{}.yaml".format(DATASET)
    if TASK == "build_manifest":
        run([sys.executable, "scripts/build_manifest.py", "--output-dir",
             ROOT / "artifacts" / "manifests", "--protocol", "p05"])
    elif TASK == "phase_a":
        run([sys.executable, "pretrain_extractor.py", "--config", phase_a_config,
             "--manifest", MANIFEST, "--experiment", EXPERIMENT, "--seed", SEED,
             "--temporal-aggregation", TEMPORAL_AGGREGATION])
    elif TASK == "phase_c":
        run([sys.executable, "train_3d_bcnn.py", "--config", phase_c_config,
             "--manifest", MANIFEST, "--init-extractor", PHASE_A_CHECKPOINT,
             "--experiment", EXPERIMENT, "--seed", SEED,
             "--temporal-aggregation", TEMPORAL_AGGREGATION])
    elif TASK == "evaluate":
        run([sys.executable, "evaluate_3d_bcnn.py", "--config", phase_c_config,
             "--manifest", MANIFEST, "--checkpoint", PHASE_C_CHECKPOINT,
             "--split", "test"])
    else:
        raise ValueError("Unknown TASK {!r}.".format(TASK))


if __name__ == "__main__":
    main()
