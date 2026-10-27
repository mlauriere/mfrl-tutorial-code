# MFRL Tutorial Experiments

Reproduce the numerical examples in the MFRL tutorial/book: cybersecurity,
discrete distribution planning and linear-quadratic mean-field control.

**This repository contains code, configurations and documentation only.** No
datasets, training histories, model checkpoints or precomputed figures are
included or downloaded. Run training to generate results locally, then redraw
figures from those saved results without repeating training.

## Get started

Clone the repository and run all commands from its root:

```bash
git clone https://github.com/mlauriere/mfrl-tutorial-code.git
cd mfrl-tutorial-code
```

Use Python 3.10 or 3.11. For example, create a Conda environment and install the
package in editable mode:

```bash
conda create -n mfrl-tutorial-oss python=3.11 pip
conda activate mfrl-tutorial-oss
python -m pip install -r requirements.txt
python -m pip install -e .
```

See [setup instructions](docs/local_setup.md) for a virtual-environment
alternative and environment checks. Jupyter, a GPU and HPC are not required.
If you do not install the package, set `PYTHONPATH="$PWD/src"`.

Check the installation and run small examples before full experiments:

```bash
PYTHONPATH=src python -m mfrl_tutorial.env_check
bash scripts/run_smoke_local.sh
```

The smoke script performs short training runs and writes results under
`outputs/smoke/`. These check that the code runs; they do not reproduce the
fully trained book results.

## Reproduce the book figures

| Examples | Step-by-step guide |
|---|---|
| Cybersecurity: tabular Q-learning and DDPG; distribution planning: DDPG | [Chapter 5: training and figure generation](docs/reproducing_chapter5.md) |
| Scalar and coupled 2D linear-quadratic control | [Chapter 6: training, analysis and figure generation](docs/reproducing_lq_revision.md) |

The guides specify all noise conditions, seeds, numerical settings, output
files and figure mappings. Full experiments take longer than smoke checks.

For example, train cybersecurity DDPG without common noise:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -B -m mfrl_tutorial.cybersecurity_ddpg --config configs/cyber_ddpg_without_cn.yaml
```

The command prints a directory such as
`outputs/cyber_ddpg_without_cn/20261027-120000`. Substitute your actual directory
in this plot-only command:

```bash
PYTHONPATH=src python -B -m mfrl_tutorial.chapter5_figures \
  --cyber-ddpg-nocn outputs/cyber_ddpg_without_cn/YOUR_TIMESTAMP \
  --output outputs/chapter5_figures
```

This produces PDF and PNG figures from local numerical exports. The Chapter 5
guide shows how to combine all six run directories into the eleven book assets.
For LQ, training is followed by exact policy analysis and a separate plot command.

Keep the generated histories, configurations and manifests so that figures can
be redrawn later. Outputs remain local and are excluded from Git. Numerical
results may vary slightly across library versions and platforms; PDF bytes
need not be identical. `RELEASE_MANIFEST.json` lists the distributed source files
and their hashes.

## Layout

- `src/mfrl_tutorial/`: importable package and config-driven entrypoints.
- `configs/`: frozen LQ JSON settings and other examples' YAML configs.
- `scripts/`: local run helpers.
- `docs/`: setup, numerical settings and figure-reproduction instructions.

## License

This code, its configurations and documentation are released under the
[MIT License](LICENSE).
