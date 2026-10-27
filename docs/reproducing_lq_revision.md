# Reproducing the revised LQ figures

This repository provides code to train the frozen benchmarks, analyze the
resulting policies and plot the book figures from those local results. No
datasets, training histories or generated figures are distributed. These
figures use plain constant-step SGD. The
`mfrl_tutorial.lq_mfc_policy_gradient` module provides the scalar LQ model
and exact benchmark routines used by this workflow. Record the source version
with `git rev-parse HEAD`.

## Dependencies and process settings

Run from the repository root with an existing Python interpreter that has the LQ
dependencies in `requirements-lq-revision.txt`. The verified stack is
Python 3.10.18, NumPy 1.26.4, SciPy 1.15.3 and Matplotlib 3.10.0, on x86_64 macOS.
Other Python/platform combinations may produce small numerical differences.

The LQ routes do not require Torch, Gymnasium, YAML or Jupyter. Those are used
by other examples retained from the public package. Package installation is
unnecessary when `PYTHONPATH` points to `src`:

```bash
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$PWD/outputs/matplotlib-cache"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

These settings affect only the current process/shell. They do not edit an
environment, activation script or system configuration. If a dependency is
missing, use an appropriate existing runtime or arrange its installation
separately; none of the commands below installs dependencies.

## Plot from local data (after training and analysis)

First run the training and exact analysis commands below. They create the
archives and metadata under `outputs/retrained_data`. The plot command requires
that directory explicitly; a fresh checkout contains no numerical results.

```bash
python -B -m mfrl_tutorial.lq_revision_plot --data outputs/retrained_data --output outputs/revised_figures
```

This performs zero training/simulator calls and does not import a model or the
native backend. It checks both archive hashes/schemas, then writes 11 PDFs,
matching PNG previews, displayed numerical curves and `figure_manifest.json`.
No C compiler or HPC is needed. To redraw one dimension only, add
`--dimension 1d` or `--dimension 2d`.

| Current book figure | Dataset | PDF assets |
|---|---|---|
| 6.1, `fig:lq_costs` | Local `lq_revision_1d.npz` | `training_cost.pdf`, `training_K.pdf`, `training_L.pdf`, `training_legend.pdf` |
| 6.2, `fig:lq_params` | Same scalar data | `chapter_population_size.pdf` |
| 6.3, `fig:lq-2d` | Local `lq_revision_2d.npz` | `training_cost_2d.pdf`, `training_K_error_2d.pdf`, `training_L_error_2d.pdf`, `training_legend_2d.pdf` |
| 6.4, `fig:lq-2d-K` | Same matrix data | `K_entries_2d.pdf` |
| 6.5, `fig:lq-2d-L` | Same matrix data | `L_entries_2d.pdf` |

Separate panels/legends preserve the book's LaTeX assembly. Figure numbers
refer to the revised book. Numerical and rendered-figure checks were performed
with the dependency versions above; byte identity is not promised across
different Matplotlib/font/library versions.

## Data, schemas and integrity

The locally generated `outputs/retrained_data/lq_revision_metadata.json`
contains ordered seeds, gain/cost optima,
complete array schemas and the figure map. Arrays load with `allow_pickle=False`.
Updates are 0 through 800 inclusive. Scalar stochastic gains have axes
`(seed, update, K/L)` and shape `(20,801,2)`. Matrix gains have axes
`(seed, update, K/L, row, column)` and shape `(20,801,2,2,2)`.
The exact reference has no seed axis. Gap arrays contain relative errors as
fractions; plotted percentages multiply them by 100.

Analysis records the generated archives' hashes and schemas in the local
metadata; plotting and saved-policy checks verify those hashes and schemas.
The local per-run arrays permit recomputation of aggregate statistics and
figure changes without repeating training. Keep results under `outputs/`;
generated datasets, histories, figures and caches are excluded from Git.
Source, configuration and documentation files intended for distribution are
listed with hashes in `RELEASE_MANIFEST.json`; local native libraries and
build metadata are also excluded.

## Frozen numerical settings

The complete machine-readable definitions are in
`configs/lq_revision_1d.json` and `configs/lq_revision_2d.json`.

| Setting | Scalar | Coupled 2D |
|---|---|---|
| State/control dimensions | 1/1 | 2/2 |
| Gains | Two scalar parameters | Unrestricted 2x2 K,L; eight free entries |
| Coefficients | All eight coefficients 0.5 | Eight matrices explicitly supplied in config |
| Discount | 0.9 | 0.9 |
| Initial gains | K=0, L=0.1 | K=0, L=0.1I2 |
| Optimizer | Plain constant-step SGD, eta=0.01 | Same |
| Updates | 800 | 800 |
| Raw one-point samples per update | M=1000 | M=24000 |
| Perturbation radius per gain | tau=0.1 | Frobenius radius tau=0.15 |
| Block normalization | 1/tau² | 4/tau² |
| Simulator horizon | 40 | 40 |
| Methods | MKV; N=2,5,10,20,50,100 | MKV; N=2,10,100 |
| Stochastic runs | 20 per method | 20 per method |

M counts joint perturbation/cost pairs, not 2M calls. Both use the original raw
one-point estimator, without Adam, a baseline, clipping, projection or seed
exclusion. Initial private/common coordinates are independent Uniform[-1,1];
innovations are independent Gaussian coordinates with variance 0.01. Matrix
initial/noise covariances are I2/3 and 0.01I2.

Scalar ordered seeds:

```text
20261028,20261029,20261030,20261101,20261102,20261103,
20261104,20261105,20261106,20261107,20261108,20261109,
20261110,20261111,20261112,20261113,20261114,20261115,
20261116,20261117
```

Matrix ordered seeds are 20261211 through 20261230 inclusive. Within each
dimension, matching seeds use common random numbers across methods and
private-agent prefixes across N. The two dimensions are not paired experiments.

The scalar RNG key is `SeedSequence([seed,1,update,stream])`; streams 0–4 are
directions, private initial state, private noise, common initial state, common
noise. Matrix keys are `SeedSequence([seed,M,update,stream])`; streams 0–4 are
directions, private initial state, common initial state, private noise, common
noise. Both use `default_rng`/PCG64. The scalar private draw shapes before
transposition are `(N,M)` and `(N,T,M)`; matrix shapes are `(N,M,2)` and
`(N,T,M,2)`. Directions are independent scalar signs or normalized Gaussian
matrices on each gain's Frobenius sphere. Noise includes the final unused draw,
as in the original implementation. Each update gets fresh keyed inputs; buffer
reuse does not reuse randomness. Resume retains the global update index.

## Training and exact analysis

Training is never launched by imports or plot commands. Methods, seeds,
update target and output directory must be explicit. `--updates` is the total
target, including an existing checkpoint prefix. `--methods all` includes one
deterministic exact-gradient reference and all stochastic methods. The exact
reference is run only once. Output is organized by method and seed.

### Bounded smoke commands

These perform at most three updates per trajectory. They are implementation
checks and do not provide the 20-seed book aggregate:

```bash
python -B -m mfrl_tutorial.lq_revision_train --config configs/lq_revision_1d.json --methods all --seeds 20261028 --updates 3 --output outputs/smoke_1d --max-seconds 120 --workers 2
python -B -m mfrl_tutorial.lq_revision_train --config configs/lq_revision_2d.json --methods all --seeds 20261211 --updates 3 --output outputs/smoke_2d --max-seconds 180 --workers 2
```

Use a new output directory for a fresh run. An existing checkpoint is rejected
unless `--resume` is supplied. To check restart equivalence, first run with
`--updates 1`, then repeat the same command with `--updates 3 --resume`.
Checkpoints include the full history prefix, numerical config/source hashes,
backend and next RNG update. For the fused backend, they also fingerprint the
C source and compiled library; rebuilding or changing either requires a fresh
run. Changing those identities is rejected. Failed
trajectories are not silently resumed. A sweep with `--resume` resumes existing
cases and starts missing cases at 0.

Partial analysis is explicit and labeled as a smoke result:

```bash
python -B -m mfrl_tutorial.lq_revision_analyze --config configs/lq_revision_1d.json --runs outputs/smoke_1d --output outputs/smoke_analysis_1d --allow-partial
```

The book plot command rejects partial datasets. Without `--allow-partial`,
analysis requires every configured method/seed and all 800 updates.

### Full reproduction commands (long runs)

The following launch long runs. They are supplied for a reader choosing to
repeat training, not as part of the bounded verification checks:

```bash
python -B -m mfrl_tutorial.lq_revision_train --config configs/lq_revision_1d.json --methods all --seeds all --updates 800 --output outputs/retrained_1d --workers 4 --max-seconds 14400
python -B -m mfrl_tutorial.lq_revision_train --config configs/lq_revision_2d.json --methods all --seeds all --updates 800 --output outputs/retrained_2d --workers 4 --max-seconds 28800
```

The default backend is NumPy. Use `--resume` with the same settings/output to
continue after a deliberate interruption or time-budget stop. Time limits
produce an incomplete manifest and a nonzero exit status rather than silently
reporting a completed sweep. Worker logs record dimensions, method/seed,
updates, metrics and timings, with structured manifests/checkpoints. A default
ten-minute budget prevents an unqualified long run; a larger explicit limit is
needed for full training. The available budget includes setup and all cases.

Analyze each dimension into a common new data directory:

```bash
python -B -m mfrl_tutorial.lq_revision_analyze --config configs/lq_revision_1d.json --runs outputs/retrained_1d --output outputs/retrained_data
python -B -m mfrl_tutorial.lq_revision_analyze --config configs/lq_revision_2d.json --runs outputs/retrained_2d --output outputs/retrained_data
python -B -m mfrl_tutorial.lq_revision_plot --data outputs/retrained_data --output outputs/retrained_figures
```

After both full analyses, recompute the saved policies' costs and gain errors
with zero optimization updates:

```bash
python -B -m mfrl_tutorial.lq_revision_check --data outputs/retrained_data --output outputs/saved_data_check.json
```

Analysis does not overwrite an existing dimension's archive. Use a new output
directory when reanalyzing. It verifies SGD update identities and recomputes
exact infinite-horizon MF/own-N costs and gain errors from saved policies. This
evaluation differs from the finite-horizon stochastic cost used for training.
Both benchmarks share MF/finite-N optimal gains, while their optimal costs
differ. Analysis writes the arrays, schemas and hashes required by plotting
and saved-policy verification.

### Optional accelerated 2D backend

NumPy works without compilation. To use the optional C population rollout with
an existing compiler:

```bash
python -B -m mfrl_tutorial.lq_revision_build_native
```

The builder uses macOS `-dynamiclib -arch <Python-process-architecture>` or
Linux `-shared -fPIC`, with `-O3 -ffp-contract=off` and no fast-math. It writes
only a local library/BUILD.json beside the included C source; it installs
nothing. `CC` may select an already installed compiler. On other platforms,
use NumPy. Compilation and execution were tested on x86_64 macOS only; Linux
build logic requires a check on Linux before claiming tested support.

Then add `--backend fused` to the 2D training command. The MKV path still uses
the checked NumPy recursion. Local compiled libraries are intentionally absent
from the distribution manifest. A missing library gives an explicit error;
there is no silent backend change during resume.

After building, check its costs against NumPy and the original unbatched
equations on 19 auxiliary samples per matrix method. This standalone check
requires no training histories or analyzed datasets:

```bash
python -B -m mfrl_tutorial.lq_revision_check --native --output outputs/native_check.json
```

## Aggregation and display conventions

All 20 stochastic runs are pooled. For training plots, apply a centered 21-point
arithmetic moving average separately to each run, with radius
`min(10,k,800-k)` at update k. This preserves both endpoints. Then compute
across-run means and sample SD (`ddof=1`). Costs show SEM=SD/sqrt20; gains and
matrix errors show one SD. Compute each run's matrix Frobenius error before
smoothing and averaging. The exact-gradient reference stays unsmoothed.

For Figure 6.2, average raw updates 601–800 inclusive within each run, then
summarize these 20 values. MF and own-N costs use their corresponding exact
optimal costs for normalization. Points sit at exact N, without jitter. No
asymptotic population-size rate is fitted. Saved errors are fractions; plots
show percent. Training plots use linear update axes and consistent colors and line styles
across dimensions. All configured seeds contribute to the displayed statistics.

## Verification, runtime and limitations

Implementation checks cover bounded training (at most three updates per
selected trajectory, including N=100 at M=24000), fresh/resumed equivalence,
batched/reference kernels and the optional native/NumPy comparison. Exact
evaluation of saved policies and figure generation require no simulator calls.
These bounded checks do not replace a complete 220-trajectory reproduction.

A reference 20-seed N=100 2D run took about 2 h 18 min using four workers and
the fused backend, with about 1.69 GiB peak per worker. This is not a runtime
guarantee for the NumPy fallback or another host.
Budget memory for full-horizon noise arrays when choosing worker count. The
three-update smoke is a check, not a reliable sustained-load runtime forecast.

The data illustrate finite-time learning, not exact stochastic convergence.
Different coefficients, M and radius prevent a controlled dimension-only
comparison. Local bitwise agreement does not establish bitwise agreement across
OS, NumPy, BLAS, compiler or font stacks. Compare numerical quantities within
appropriate tolerances and interpret stochastic results using their uncertainty.

The root MIT License covers the included revision code, configurations and
documentation; retain its copyright and permission notice.
