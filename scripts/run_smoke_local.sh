#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${PWD}/.cache/matplotlib"
export XDG_CACHE_HOME="${PWD}/.cache"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

python -m mfrl_tutorial.env_check
LQ_SMOKE_DIR="outputs/smoke/lq/$(date +%Y%m%d_%H%M%S)_$$"
python -B -m mfrl_tutorial.lq_revision_train --config configs/lq_revision_1d.json --methods MKV Pop_N2 --seeds 20261028 --updates 1 --output "${LQ_SMOKE_DIR}/1d" --max-seconds 120
python -B -m mfrl_tutorial.lq_revision_train --config configs/lq_revision_2d.json --methods MKV Pop_N2 --seeds 20261211 --updates 1 --output "${LQ_SMOKE_DIR}/2d" --max-seconds 120
python -m mfrl_tutorial.cybersecurity_qlearning --config configs/smoke_cyber_qlearning.yaml
python -m mfrl_tutorial.cybersecurity_ddpg --config configs/smoke_cyber_ddpg.yaml
python -m mfrl_tutorial.distribution_planning_ddpg --config configs/smoke_distribution_planning.yaml
