"""Evaluate new training histories exactly and export revision plot-data arrays."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from .lq_revision_common import load_config, read_json, write_json, digest, config_digest, Logger
from .lq_revision_model import make_model, optimum, cost_parts


def statistics(value):
    value = np.asarray(value)
    sd = float(value.std(ddof=1)) if len(value) > 1 else None
    return dict(n=len(value), mean=float(value.mean()), SD=sd,
                SEM=None if sd is None else sd/np.sqrt(len(value)), per_seed=value.tolist())


def fields(model, theta, dimension, opt, N=None):
    parts = cost_parts(model, theta, dimension)
    star_parts = cost_parts(model, opt, dimension)
    mf, star = parts.sum(axis=-1), star_parts.sum()
    delta = theta-opt
    if dimension == 1:
        answer = dict(theta=theta, mf_cost=mf, mf_relative_gap=(mf-star)/star,
            absolute_K_error=np.abs(delta[..., 0]), absolute_L_error=np.abs(delta[..., 1]),
            relative_operator_error=np.max(np.abs(delta), axis=-1)/np.max(np.abs(opt)))
    else:
        norms = np.linalg.norm(delta, axis=(-2, -1))
        answer = dict(theta=theta, MF_relative_error=(mf-star)/star,
                      K_Frobenius_error=norms[..., 0], L_Frobenius_error=norms[..., 1])
    if N is not None:
        weights = np.array([1-1/N, 1+1/N])
        own, ownstar = parts@weights, star_parts@weights
        answer['own_relative_gap' if dimension == 1 else 'own_relative_error'] = (own-ownstar)/ownstar
        if dimension == 1:
            answer['own_cost'] = own
            rms = np.sqrt((1-1/N)*model.var_y0*delta[..., 0]**2+(model.var_z0+model.var_y0/N)*delta[..., 1]**2)
            optimal_rms = np.sqrt((1-1/N)*model.var_y0*opt[0]**2+(model.var_z0+model.var_y0/N)*opt[1]**2)
            answer['relative_RMS_control_error'] = rms/optimal_rms
    return answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--runs', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--allow-partial', action='store_true', help='Explicit smoke analysis; never substitutes for the book aggregate')
    a = ap.parse_args()
    started = time.monotonic()
    config = load_config(a.config)
    dimension = config['dimension']
    slug = f'{dimension}d'
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out/f'analysis_{slug}.log', started)
    log('start', f'dimension={dimension} runs={a.runs} output={out} partial={a.allow_partial}; exact policy evaluation, zero simulator calls')
    if read_json(a.runs/'resolved_config.json') != config:
        raise ValueError('Run directory has another config')
    model = make_model(config)
    opt = optimum(model, dimension)
    arrays, sources, records, missing = {}, {}, {}, []
    method_seeds = {}
    for method in config['methods']+['Exact']:
        wanted = config['seeds'] if method != 'Exact' else config['seeds'][:1]
        histories, kept = [], []
        for seed in wanted:
            directory = a.runs/method/('reference' if method == 'Exact' else str(seed))
            cp = directory/'checkpoint.json'
            if not cp.is_file():
                missing.append(dict(method=method, seed=seed, reason='absent'))
                continue
            state = read_json(cp)
            signature = state['signature']
            if signature['config_sha256'] != config_digest(config) or signature['method'] != method or signature['seed'] != seed:
                raise ValueError('Checkpoint identity mismatch: '+str(cp))
            if state['status'] == 'failed':
                raise ValueError('Failed trajectory cannot be aggregated: '+str(cp))
            if state['completed_updates'] != config['training']['updates']:
                missing.append(dict(method=method, seed=seed, reason='incomplete', completed=state['completed_updates']))
            rows = state['rows']
            if [r['k'] for r in rows] != list(range(state['completed_updates']+1)):
                raise ValueError('History index gap: '+str(cp))
            theta = np.asarray([r['theta'] for r in rows])
            np.testing.assert_array_equal(theta[0], np.asarray([config['training']['K0'], config['training']['L0']]))
            np.testing.assert_array_equal(theta[-1], np.asarray(state['theta']))
            if len(rows) > 1:
                np.testing.assert_array_equal(theta[1:], theta[:-1]-config['training']['eta']*np.asarray([r['gradient'] for r in rows[1:]]))
            # Re-evaluate saved scalar metrics rather than trusting history labels.
            N = int(method.removeprefix('Pop_N')) if method.startswith('Pop_N') else None
            calculated = fields(model, theta, dimension, opt, N)
            mf_key = 'mf_relative_gap' if dimension == 1 else 'MF_relative_error'
            np.testing.assert_allclose(calculated[mf_key], [r['MF_relative_error'] for r in rows], rtol=1e-8, atol=5e-14)
            if N is not None:
                own_key = 'own_relative_gap' if dimension == 1 else 'own_relative_error'
                np.testing.assert_allclose(calculated[own_key], [r['own_relative_error'] for r in rows], rtol=1e-8, atol=5e-14)
            histories.append(theta)
            kept.append(seed)
            sources[str(cp.relative_to(a.runs))] = digest(cp)
        if not histories:
            continue
        length = min(len(theta) for theta in histories)
        if not a.allow_partial and (missing or len(histories) != len(wanted)):
            raise ValueError('Missing/incomplete frozen runs; use --allow-partial only for explicit smoke analysis')
        theta = histories[0][:length] if method == 'Exact' else np.stack([v[:length] for v in histories])
        N = int(method.removeprefix('Pop_N')) if method.startswith('Pop_N') else None
        result = fields(model, theta, dimension, opt, N)
        if method == 'Exact' and dimension == 1:
            result = {k: result[k] for k in ['theta', 'mf_cost', 'mf_relative_gap']}
        arrays.update({method+'__'+key: value for key, value in result.items()})
        method_seeds[method] = kept
        records[method] = dict(seeds=kept, updates=length-1, raw_late_window=[601, 800], summaries={})
        if length >= 801 and method != 'Exact':
            records[method]['summaries'] = {k: statistics(v[:, 601:801].mean(axis=1)) for k, v in result.items() if k != 'theta'}
        log('method', f'{method} seeds={kept} updates={length-1}; SGD identities and exact cost checks passed')
    if missing and not a.allow_partial:
        raise ValueError('Missing frozen trajectories: '+str(missing))
    if not arrays:
        raise ValueError('No histories to analyze')
    archive = out/f'lq_revision_{slug}.npz'
    if archive.exists():
        raise ValueError('Analysis output already exists; select a new output directory to preserve it')
    np.savez_compressed(archive, **arrays)
    metadata_path = out/'lq_revision_metadata.json'
    metadata = read_json(metadata_path) if metadata_path.exists() else dict(schema_version=1, license='MIT; see LICENSE', datasets={}, figure_map={})
    metadata['reporting'] = config['evaluation']
    metadata['datasets'][slug] = dict(file=archive.name, sha256=digest(archive), bytes=archive.stat().st_size,
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()},
        seeds=config['seeds'], method_seeds=method_seeds, methods=[m for m in config['methods'] if m in records],
        optimum=opt.tolist(), MF_optimal_cost=float(cost_parts(model, opt, dimension).sum()), partial=bool(missing),
        source_checkpoint_hashes=sources)
    specs = [('6.1', 'fig:lq_costs', ['training_cost.pdf', 'training_K.pdf', 'training_L.pdf', 'training_legend.pdf']),
             ('6.2', 'fig:lq_params', ['chapter_population_size.pdf'])] if dimension == 1 else [
             ('6.3', 'fig:lq-2d', ['training_cost_2d.pdf', 'training_K_error_2d.pdf', 'training_L_error_2d.pdf', 'training_legend_2d.pdf']),
             ('6.4', 'fig:lq-2d-K', ['K_entries_2d.pdf']), ('6.5', 'fig:lq-2d-L', ['L_entries_2d.pdf'])]
    for number, label, files in specs:
        metadata['figure_map'][number] = dict(label=label, dataset=slug, files=files)
    write_json(metadata_path, metadata)
    write_json(out/f'analysis_summary_{slug}.json', dict(status='partial_smoke' if missing else 'complete',
        records=records, missing=missing, source_checkpoint_hashes=sources, wall_seconds=time.monotonic()-started,
        archive_sha256=digest(archive), simulator_calls=0))
    log('done', f'archive={archive.name} status={"partial_smoke" if missing else "complete"} elapsed={time.monotonic()-started:.3f}s')


if __name__ == '__main__':main()
