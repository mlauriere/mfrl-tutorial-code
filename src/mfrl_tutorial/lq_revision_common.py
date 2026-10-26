"""Small standard-library helpers shared by revision commands."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_config(path):
    config = read_json(path)
    if config['schema_version'] != 1 or config['dimension'] not in (1, 2):
        raise ValueError('Unsupported revision configuration')
    if config['optimizer'] != 'plain constant-step SGD' or config['estimator'] != 'raw one-point independent-cost':
        raise ValueError('Only the frozen plain-SGD/raw-one-point method is supported')
    t = config['training']
    if not (t['eta'] > 0 and t['tau'] > 0 and t['M'] >= 2 and t['T'] >= 1):
        raise ValueError('Invalid training parameters')
    if len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Repeated seed IDs')
    # Initial draws are Uniform[-1,1]; changing this covariance alone would not
    # change that law. Reject inconsistent edits rather than silently using them.
    if config['model']['initial_covariance_per_component'] != 1 / 3:
        raise ValueError('Uniform[-1,1] initial coordinates have covariance1/3')
    expected_key = ['seed', 1, 'update', 'stream'] if config['dimension'] == 1 else ['seed', 'M', 'update', 'stream']
    if config['rng']['key'] != expected_key:
        raise ValueError('RNG protocol differs from the implemented benchmark')
    return config


def versions():
    import numpy
    import scipy
    import matplotlib
    return dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, platform=platform.platform(), architecture=platform.machine())


class Logger:
    def __init__(self, path, started=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic() if started is None else started

    def __call__(self, stage, message):
        line = f'[elapsed={time.monotonic()-self.started:10.3f}s] [{stage}] {message}'
        print(line, flush=True)
        with self.path.open('a') as stream:
            stream.write(line + '\n')
