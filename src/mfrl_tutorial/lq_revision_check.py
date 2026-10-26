"""Check saved-policy evaluation and optionally a tiny native rollout batch.

No optimization updates are performed. Native checks use 19 auxiliary samples
and can run without saved datasets.
"""
from pathlib import Path
import argparse
import time
import numpy as np
from .lq_revision_common import load_config, read_json, write_json, versions
from .lq_revision_model import make_model, optimum
from .lq_revision_analyze import fields
from .lq_revision_plot import read_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--data', type=Path,
                    help='Local directory containing both complete analyzed datasets')
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.data is None and not a.native:
        ap.error('Provide --data for saved-policy checks or --native for a standalone native check')
    started = time.monotonic()
    checks = []
    if a.data is not None:
        metadata_path = a.data/'lq_revision_metadata.json'
        if not metadata_path.is_file():
            ap.error('No lq_revision_metadata.json in --data; run lq_revision_analyze first')
        metadata = read_json(metadata_path)
        for dimension in [1, 2]:
            slug = f'{dimension}d'
            config = load_config(a.root/f'configs/lq_revision_{slug}.json')
            data = read_dataset(a.data, metadata['datasets'][slug])
            model = make_model(config)
            opt = optimum(model, dimension)
            np.testing.assert_allclose(opt, metadata['datasets'][slug]['optimum'], rtol=1e-13, atol=2e-15)
            late_metrics = {}
            for method in config['methods']+['Exact']:
                N = int(method[5:]) if method.startswith('Pop_N') else None
                computed = fields(model, data[method+'__theta'], dimension, opt, N)
                for field, value in computed.items():
                    key = method+'__'+field
                    if key in data:
                        np.testing.assert_allclose(value, data[key], rtol=1e-8, atol=5e-13)
                if method != 'Exact':
                    field = 'mf_relative_gap' if dimension == 1 else 'MF_relative_error'
                    late = 100*computed[field][:, 601:801].mean(axis=1)
                    late_metrics[method] = dict(mean_MF_error_percent=float(late.mean()),
                        SEM_percent=float(late.std(ddof=1)/np.sqrt(20)))
                    # Historical exports may also supply independently recorded
                    # summaries; newly analyzed datasets need no historical values.
                    expected = metadata['datasets'][slug].get('late_metrics', {}).get(method)
                    if expected is not None:
                        np.testing.assert_allclose(late.mean(), expected['mean_MF_error_percent'], atol=5e-12, rtol=0)
                        np.testing.assert_allclose(late.std(ddof=1)/np.sqrt(20), expected['SEM_percent'], atol=5e-12, rtol=0)
            checks.append(dict(dataset=slug, hashes_schema_metrics='passed', late_metrics=late_metrics))
    if a.native:
        from . import lq_revision_matrix as matrix
        from . import lq_revision_matrix_reference as reference
        from . import lq_revision_native as native
        config = load_config(a.root/'configs/lq_revision_2d.json')
        flat = dict(config['model'], **config['training'], seed=config['seeds'][0])
        model = make_model(config)
        theta = np.asarray([flat['K0'], flat['L0']])
        for method in config['methods']:
            N = int(method[5:]) if method.startswith('Pop_N') else 1
            directions, draws = reference.inputs(flat, 19, N, 0)
            policies = theta[None]+directions
            numpy_costs = matrix.rollout(model, policies, draws, method)
            fused_costs = native.rollout_fused(model, policies, draws, method)
            direct_costs = reference.direct_rollout(model, policies, draws, method)
            np.testing.assert_allclose(numpy_costs, direct_costs, rtol=3e-13, atol=3e-13)
            np.testing.assert_allclose(fused_costs, numpy_costs, rtol=3e-13, atol=3e-13)
            checks.append(dict(method=method, auxiliary_samples=19, native_and_unbatched='passed',
                               max_native_cost_difference=float(np.max(np.abs(fused_costs-numpy_costs)))))
    result = dict(status='passed', checks=checks, optimization_updates=0, native_test=a.native,
                  versions=versions(), wall_seconds=time.monotonic()-started)
    write_json(a.output, result)
    print(f'Checks passed in{result["wall_seconds"]:.3f}s; zero optimization updates; report={a.output}')


if __name__ == '__main__':main()
