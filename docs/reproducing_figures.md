# Reproducing Figures

For book Figures 6.1–6.5, use
[the LQ reproduction guide](reproducing_lq_revision.md). It provides the frozen
settings, exact figure mapping, independent plain-SGD training/analysis commands
and plotting commands for locally generated data. No datasets are distributed.
The commands below also cover the
cybersecurity and distribution-planning examples.

Run from this folder.

Environment check:

```bash
PYTHONPATH=src python -m mfrl_tutorial.env_check
```

First train and analyze the LQ experiments using the linked guide. Then plot
the locally generated results without further training:

```bash
PYTHONPATH=src python -B -m mfrl_tutorial.lq_revision_plot --data outputs/retrained_data --output outputs/revised_figures
```

Independent LQ training uses the frozen 1D and 2D configurations and explicit
update/time budgets; follow the linked guide for those commands.

Cybersecurity Q-learning:

```bash
PYTHONPATH=src python -m mfrl_tutorial.cybersecurity_qlearning --config configs/cyber_qlearning_without_cn.yaml
PYTHONPATH=src python -m mfrl_tutorial.cybersecurity_qlearning --config configs/cyber_qlearning_with_cn.yaml
```

Cybersecurity DDPG:

```bash
PYTHONPATH=src python -m mfrl_tutorial.cybersecurity_ddpg --config configs/cyber_ddpg_without_cn.yaml
PYTHONPATH=src python -m mfrl_tutorial.cybersecurity_ddpg --config configs/cyber_ddpg_with_cn.yaml
```

Distribution planning DDPG:

```bash
PYTHONPATH=src python -m mfrl_tutorial.distribution_planning_ddpg --config configs/distribution_planning_without_cn.yaml
PYTHONPATH=src python -m mfrl_tutorial.distribution_planning_ddpg --config configs/distribution_planning_with_cn.yaml
```

The long stochastic training runs may not reproduce byte-identical PDF files across machines,
but fixed seeds and saved run manifests should make the numerical setup clear enough for
comparison and extension.
