#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${PWD}/.cache/matplotlib"
export XDG_CACHE_HOME="${PWD}/.cache"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

PYTHONPATH=src python -m mfrl_tutorial.env_check
