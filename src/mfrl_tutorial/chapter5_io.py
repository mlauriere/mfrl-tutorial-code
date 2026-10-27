"""Small local numerical exports for Chapter 5; no plotting or simulation."""
from pathlib import Path
import hashlib
import json

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_info(folder, kind, config):
    """Record the actual resolved configuration, including its noise mode."""
    write_json(Path(folder)/'chapter5_run.json',
               dict(schema_version=1, kind=kind, config=config))


def save_snapshot(folder, kind, index, arrays, **metadata):
    """Save arrays without pickle and update a pointer to this checkpoint.

    Called only with values already produced by the training/evaluation code;
    saving performs no inference and consumes no random numbers.
    """
    folder = Path(folder)/'plot_data'
    folder.mkdir(parents=True, exist_ok=True)
    numeric = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in numeric.values()):
        raise ValueError('Plot exports must contain numeric arrays, not objects')
    path = folder/f'{kind}_ep{index:07d}.npz'
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **numeric)
    temporary.replace(path)
    descriptor = dict(schema_version=1, kind=kind, index=int(index),
        file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                for key, value in numeric.items()}, metadata=metadata)
    write_json(path.with_suffix('.json'), descriptor)
    write_json(folder/f'latest_{kind}.json', descriptor)


def save_training(folder, index, rewards, eval_rewards, eval_indices,
                  *, window=20, q_diffs=(), first_index=0):
    save_snapshot(folder, 'training', index, dict(
        rewards=np.asarray(rewards, dtype=float),
        indices=np.arange(first_index, first_index + len(rewards)),
        eval_rewards=np.asarray(eval_rewards, dtype=float),
        eval_indices=np.asarray(eval_indices, dtype=int),
        q_diffs=np.asarray(q_diffs, dtype=float)), moving_average_window=int(window),
        cost_convention='Negative undiscounted episode total reward')


def load_snapshot(folder, kind):
    folder = Path(folder)/'plot_data'
    pointer = folder/f'latest_{kind}.json'
    if not pointer.is_file():
        raise ValueError(f'Missing {pointer}; run the corresponding training command first')
    descriptor = json.loads(pointer.read_text())
    if descriptor['schema_version'] != 1 or descriptor['kind'] != kind:
        raise ValueError(f'Unsupported plotting descriptor: {pointer}')
    filename = descriptor['file']
    if Path(filename).name != filename:
        raise ValueError('Plot archive must be in the same directory as its descriptor')
    path = folder/filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor['sha256']:
        raise ValueError(f'Plot archive checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if arrays.keys() != descriptor['arrays'].keys():
        raise ValueError(f'Plot array keys do not match descriptor: {path}')
    for key, value in arrays.items():
        spec = descriptor['arrays'][key]
        if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
            raise ValueError(f'Plot array schema mismatch: {key}')
    return arrays, descriptor
