# Setup

Run from the repository root using Python 3.10 or 3.11. The commands below use
Python 3.11. Choose either Conda or a Python virtual environment.

## Conda

```bash
conda create -n mfrl-tutorial-oss python=3.11 pip
conda activate mfrl-tutorial-oss
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Python virtual environment

On macOS/Linux with Python 3.11 installed:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use one of these environments, or an existing compatible one, for all commands.
Jupyter is not required. The entrypoints are ordinary `python -m` commands.
Editable installation keeps the supplied configurations and optional C source
available from the checkout. Run reproduction commands from that checkout.

For only the LQ examples, `requirements-lq-revision.txt` supplies the smaller
NumPy/SciPy/Matplotlib dependency set; use `PYTHONPATH=src` instead of installing
the package with its full dependencies. See the LQ guide for commands.

## Check the runtime

```bash
PYTHONPATH=src python -m mfrl_tutorial.env_check
```

The full example set requires PyYAML (`import yaml`) as well as the numerical
and plotting dependencies. CPU is the configured device; a GPU is optional.

Architecture check:

```bash
uname -m
python -c "import platform; print(platform.machine())"
```

PyTorch check:

```bash
python - <<'PYTORCH'
import platform
import torch

print("machine", platform.machine())
print("torch", torch.__version__)
print("mps available", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
print("cuda available", torch.cuda.is_available())
PYTORCH
```

The run wrappers use headless plotting and writable local cache directories.
Results, logs, histories and generated figures are written under `outputs/`
and are ignored by Git.
