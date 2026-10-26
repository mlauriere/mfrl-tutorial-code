"""Fresh-start/resume plain SGD for the frozen LQ revision benchmarks.

Methods, seeds, update target and output must be explicitly selected. The default
wall budget is ten minutes; full runs require a deliberate larger budget.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import signal
import time

# Process-local numerical thread controls. No activation files or environments
# are edited. Callers can set these explicitly before launching Python.
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(name, '1')
import numpy as np
from .lq_revision_common import load_config, config_digest, read_json, write_json, digest, versions, Logger
from .lq_revision_model import make_model, exact_gradient, evaluate


class BudgetExpired(RuntimeError):
    pass


def kernel_hashes(backend='numpy'):
    here = Path(__file__).resolve().parent
    names = ['lq_revision_train.py', 'lq_revision_common.py', 'lq_revision_model.py', 'lq_revision_scalar.py',
             'lq_mfc_policy_gradient.py', 'lq_revision_matrix.py', 'lq_revision_matrix_reference.py', 'lq_revision_native.py']
    hashes = {n: digest(here/n) for n in names}
    if backend == 'fused':
        from .lq_revision_native import native_library_path
        library = native_library_path()
        hashes['native/rollout_fused.c'] = digest(here/'native/rollout_fused.c')
        hashes['native/'+library.name] = digest(library)
    return hashes


def train_case(config, method, seed, target, output, resume, backend, deadline, started, checkpoint_every=25):
    dimension = config['dimension']
    t = config['training']
    directory = Path(output)/method/('reference' if method == 'Exact' else str(seed))
    directory.mkdir(parents=True, exist_ok=True)
    log = Logger(directory/'run.log', started)
    checkpoint_path = directory/'checkpoint.json'
    signature = dict(config_sha256=config_digest(config), method=method, seed=seed,
                     backend=backend, kernels=kernel_hashes(backend))
    model = make_model(config)
    theta = np.asarray([t['K0'], t['L0']], dtype=float)
    rows, completed = [], 0
    if checkpoint_path.exists():
        if not resume:
            raise ValueError('Output already has a checkpoint; use --resume: '+str(directory))
        saved = read_json(checkpoint_path)
        if saved['signature'] != signature:
            raise ValueError('Checkpoint config, method, backend or numerical source differs')
        if saved['status'] == 'failed':
            raise ValueError('A failed trajectory needs review; it is not silently resumed')
        rows, completed, theta = saved['rows'], saved['completed_updates'], np.asarray(saved['theta'])
        if completed > target:
            raise ValueError('Requested target precedes checkpoint')
        if rows[-1]['k'] != completed or not np.array_equal(theta, rows[-1]['theta']):
            raise ValueError('Checkpoint/history mismatch')
    elif resume:
        # A sweep can resume completed cases and start missing cases. An absent
        # checkpoint always starts at0; it never imports another experiment.
        log('resume', f'{method} seed={seed}: no existing checkpoint, fresh start')
    N = int(method.removeprefix('Pop_N')) if method.startswith('Pop_N') else None
    if not rows:
        rows.append(dict(k=0, theta=theta.tolist(), **evaluate(model, theta, dimension, N)))
    if dimension == 1:
        from .lq_revision_scalar import estimate
        params = dict(t, gamma=config['model']['gamma'], cost_calls_per_update=t['M'], estimator='one_point_iid')
        gradient = lambda point, k: estimate(model, point, params, method, seed, 1, k)
    else:
        from .lq_revision_native import estimate, InputBuffer
        cfg = dict(config['model'], **t, seed=seed)
        buffer = None if method == 'Exact' else InputBuffer(cfg, t['M'], N or 1)
        gradient = lambda point, k: estimate(model, point, cfg, t['M'], method, k,
                    variant='fused' if backend == 'fused' else 'standard', input_buffer=buffer)

    def checkpoint(status, error=None):
        state = dict(schema_version=1, status=status, signature=signature, config=config,
                     completed_updates=completed, next_rng_update=completed, theta=theta.tolist(),
                     optimizer_state=dict(t=completed), rows=rows, versions=versions(), error=error)
        # The checkpoint is canonical and includes its complete history prefix;
        # interrupted writes to the convenience history export cannot corrupt resume.
        write_json(checkpoint_path, state)
        write_json(directory/'history.json', rows)

    def guard(*_):
        if time.monotonic() >= deadline:
            raise BudgetExpired('Wall-time budget reached')

    previous_handler = None
    if hasattr(signal, 'setitimer'):
        previous_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(BudgetExpired('Wall-time budget reached')))
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
    log('start', f'{method} seed={seed} dimension={dimension} M={t["M"]} T={t["T"]} backend={backend} from={completed} target={target}')
    status, error = 'completed', None
    try:
        for k in range(completed, target):
            guard()
            tick = time.monotonic()
            g = exact_gradient(model, theta, dimension) if method == 'Exact' else gradient(theta, k)
            guard()
            proposal = theta-t['eta']*g
            if not np.isfinite(g).all() or not np.isfinite(proposal).all():
                raise ValueError('Nonfinite gradient or policy; no clipping or replacement is applied')
            metrics = evaluate(model, proposal, dimension, N)
            theta, completed = proposal, k+1
            rows.append(dict(k=completed, theta=theta.tolist(), gradient=g.tolist(),
                             kernel_seconds=time.monotonic()-tick, **metrics))
            if completed % checkpoint_every == 0:
                checkpoint('running')
            if completed <= 3 or completed % 50 == 0 or completed == target:
                log('update', f'{method} seed={seed} k={completed}/{target} MF_error={metrics["MF_relative_error"]:.8g} step_seconds={rows[-1]["kernel_seconds"]:.4f}')
    except BudgetExpired as exc:
        status, error = 'budget_stopped', str(exc)
    except Exception as exc:
        status, error = 'failed', repr(exc)
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        checkpoint(status, error)
    log('done', f'{method} seed={seed} status={status} completed={completed} error={error}')
    return dict(method=method, seed=seed, status=status, completed_updates=completed,
                directory=str(directory), error=error)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--methods', nargs='+', required=True, help='Exact, MKV, Pop_N2 etc, or all')
    ap.add_argument('--seeds', nargs='+', required=True, help='Seed IDs in the config, or all; Exact runs only once')
    ap.add_argument('--updates', type=int, required=True, help='Total target, including any resumed prefix')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--backend', choices=['numpy', 'fused'], default='numpy')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--max-seconds', type=float, default=600)
    a = ap.parse_args()
    config = load_config(a.config)
    if not 0 <= a.updates <= config['training']['updates'] or a.workers < 1 or a.max_seconds <= 0:
        ap.error('Invalid update, worker or time budget')
    methods = ['Exact']+config['methods'] if a.methods == ['all'] else a.methods
    seeds = config['seeds'] if a.seeds == ['all'] else [int(s) for s in a.seeds]
    if len(set(methods)) != len(methods) or len(set(seeds)) != len(seeds):
        ap.error('Repeated methods or seed IDs')
    if not set(methods) <= set(config['methods']+['Exact']) or not set(seeds) <= set(config['seeds']):
        ap.error('Methods or seeds are outside the frozen configuration')
    if a.backend == 'fused':
        from .lq_revision_native import native_library_path
        if config['dimension'] != 2 or not native_library_path().is_file():
            ap.error('Fused backend needs dimension2 and a local native build; use numpy otherwise')
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started+a.max_seconds
    log = Logger(output/'run.log', started)
    log('setup', f'config={a.config} output={output} dimension={config["dimension"]} methods={methods} seeds={seeds} target={a.updates} workers={a.workers} budget={a.max_seconds}s')
    log('versions', str(versions()))
    resolved = output/'resolved_config.json'
    if resolved.exists() and read_json(resolved) != config:
        ap.error('Output belongs to a different configuration')
    write_json(resolved, config)
    cases = [(m, config['seeds'][0] if m == 'Exact' else s) for m in methods
             for s in (seeds[:1] if m == 'Exact' else seeds)]
    results = []
    if a.workers == 1:
        for m, s in cases:
            if time.monotonic() >= deadline:
                break
            results.append(train_case(config, m, s, a.updates, output, a.resume, a.backend, deadline, started))
    else:
        # Spawn prevents inherited BLAS state and keeps each method/seed's files separate.
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs = {pool.submit(train_case, config, m, s, a.updates, output, a.resume, a.backend, deadline, started): (m, s) for m, s in cases}
            for job in as_completed(jobs):
                results.append(job.result())
    status = 'completed' if len(results) == len(cases) and all(r['status'] == 'completed' for r in results) else 'incomplete'
    write_json(output/'run_manifest.json', dict(status=status, config=config, versions=versions(),
        cases=results, requested_cases=len(cases), wall_seconds=time.monotonic()-started, kernels=kernel_hashes(a.backend)))
    log('done', f'status={status} cases={len(results)}/{len(cases)} elapsed={time.monotonic()-started:.3f}s')
    if status != 'completed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
