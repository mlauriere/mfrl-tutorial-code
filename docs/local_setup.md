# Local Mac Setup

This project targets local runs first.

```bash
conda create -n mfrl-tutorial-oss python=3.11 pip
source ~/opt/anaconda3/bin/activate mfrl-tutorial-oss
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you reuse an existing environment instead of creating this one, run `python -m mfrl_tutorial.env_check`
and make sure `yaml` is available. The package declares `pyyaml`; the small fallback parser is only for
basic local smoke runs.

Jupyter is intentionally excluded from the runtime environment. The public commands are plain
`python -m mfrl_tutorial...` module calls.

Architecture check:

```bash
uname -m
python -c "import platform; print(platform.machine())"
conda info | grep -E "platform|active env"
```

PyTorch check:

```bash
python - <<'PY'
import platform
import torch

print("machine", platform.machine())
print("torch", torch.__version__)
print("mps available", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)
print("cuda available", torch.cuda.is_available())
PY
```

For headless plotting, the run wrappers create writable matplotlib and XDG cache directories inside each output directory.
