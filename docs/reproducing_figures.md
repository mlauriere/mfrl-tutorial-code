# Reproducing the figures

Run commands from the repository root after following [setup](local_setup.md).
No datasets, training histories, model checkpoints or precomputed figures are
included. Each workflow starts by generating numerical results locally.

| Examples | Workflow | Guide |
|---|---|---|
| Cybersecurity Q-learning, cybersecurity DDPG, distribution planning | Train the required noise conditions, then assemble PDF/PNG figures from the saved numerical exports. | [Chapter 5](reproducing_chapter5.md) |
| Scalar and coupled 2D linear-quadratic control | Train the configured methods/seeds, analyze the saved policies, then generate the separate panels and legends. | [Chapter 6](reproducing_lq_revision.md) |

Both guides give exact commands, numerical settings, generated filenames,
figure mappings and short smoke checks. Plotting does not train a policy or
simulate missing data. Saved local results can be replotted after changing
plot styles without running training again.

Use the full configurations for book-scale results. Smoke configurations are
short implementation checks and produce different numerical results.

Keep the generated histories, resolved configurations, logs and manifests under
`outputs/`. These are local results, excluded from Git. Record the source commit
and runtime versions when reporting a reproduction. Numerical results and PDF
bytes can differ across machines and library versions.
