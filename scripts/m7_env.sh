#!/usr/bin/env bash
# Source for M7 processes only. Matches Idea A's Conda C++ runtime requirement.
export LD_LIBRARY_PATH="/home/me/anaconda3/envs/latdefer/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
