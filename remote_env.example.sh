#!/usr/bin/env bash
# Per-machine dataset locations.
#
# The repository is synced with git, so the configs must stay untouched or every
# `git pull` fights a local edit. These variables override
# `data.dataset_roots` in every config instead, leaving the working tree clean.
#
# Setup on a new workstation:
#   cp remote_env.example.sh remote_env.sh   # remote_env.sh is gitignored
#   $EDITOR remote_env.sh                    # point the two paths at this machine
#   source remote_env.sh
#
# To survive new shells, append the same two exports to ~/.bashrc.

export DFD_ROOT="/root/autodl-tmp/datasets/DFD-Kaggle"
export CELEBDFV3_ROOT="/root/autodl-tmp/datasets/CelebDFv3"

# DataLoader workers. Measure with scripts/benchmark_loader.py -- more is not
# automatically faster: Haar detection on full-resolution frames dominates and
# is memory-bandwidth bound. Leave unset to use the config value (2).
# export NUM_WORKERS=4

# Sanity check: these must list the dataset's top-level folders.
#   $DFD_ROOT       -> "DFD_original sequences", "DFD_manipulated_sequences"
#   $CELEBDFV3_ROOT -> "REAL", "FAKE"
echo "DFD_ROOT       = ${DFD_ROOT}"
echo "CELEBDFV3_ROOT = ${CELEBDFV3_ROOT}"
