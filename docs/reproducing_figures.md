# Reproducing Figures

Run from this folder.

Environment check:

```bash
PYTHONPATH=src python -m mfrl_tutorial.env_check
```

LQ data-generation demo:

```bash
PYTHONPATH=src python -m mfrl_tutorial.lq_generate_data --config configs/lq_generate_demo_2seeds.yaml
```

The LQ data-generation demo is a lightweight from-scratch check with two seeds, so it should be
treated as an algorithmic reproducibility demo rather than a statistical reproduction of the book
aggregate. Full aggregate LQ data are not bundled in this first public version.

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
