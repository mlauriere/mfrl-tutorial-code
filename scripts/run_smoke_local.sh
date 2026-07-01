#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export MPLCONFIGDIR="${PWD}/.cache/matplotlib"
export XDG_CACHE_HOME="${PWD}/.cache"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

python -m mfrl_tutorial.env_check
python -m mfrl_tutorial.lq_generate_data --config configs/smoke_lq_generate.yaml
python -m mfrl_tutorial.cybersecurity_qlearning --config configs/smoke_cyber_qlearning.yaml
python -m mfrl_tutorial.cybersecurity_ddpg --config configs/smoke_cyber_ddpg.yaml
python -m mfrl_tutorial.distribution_planning_ddpg --config configs/smoke_distribution_planning.yaml
