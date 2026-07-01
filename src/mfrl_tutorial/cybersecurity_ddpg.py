"""Config-driven DDPG runs for the cybersecurity example."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np

from .utils import (
    RunLogger,
    configure_seed,
    create_run_dir,
    load_config,
    setup_matplotlib_cache,
    tee_output,
    write_manifest,
    write_resolved_config,
)


def _common_noise_sampler(cyber_core: Any, config: Dict[str, Any]):
    cn = config.get("common_noise", {})
    if not cn.get("enabled", False):
        return lambda x: x
    sigma = float(cn.get("sigma", 0.05))
    low = float(cn.get("clip_low_multiplier", 0.7)) * cyber_core.v_H_DEFAULT
    high = float(cn.get("clip_high_multiplier", 1.3)) * cyber_core.v_H_DEFAULT
    return lambda x: np.clip(x + np.random.normal(0.0, sigma), low, high)


def _initial_distribution_sampler(n_states: int):
    def sampler() -> np.ndarray:
        p = np.random.rand()
        if p < 0.4:
            v = np.random.dirichlet([1.0] * n_states)
        elif p < 0.8:
            v = np.random.dirichlet([0.1] * n_states)
        else:
            v = np.zeros(n_states)
            num_active = np.random.randint(1, 3)
            indices = np.random.choice(n_states, num_active, replace=False)
            v[indices] = np.random.uniform(0.1, 1.0, size=num_active)
        return v / (np.sum(v) + 1e-8)

    return sampler


def run(config: Dict[str, Any]) -> str:
    output_dir = create_run_dir(config)
    setup_matplotlib_cache(output_dir)

    from . import cybersecurity_ddpg_core as cyber_core

    import torch

    configure_seed(config.get("seed"))
    write_resolved_config(config, output_dir)
    run_log = output_dir / "run.log"
    logger = RunLogger(run_log)

    with tee_output(run_log):
        logger.log("setup", f"output_dir={output_dir}")
        params = config.get("model", {})
        train_cfg = config.get("training", {})
        net_cfg = config.get("network", {})

        device = torch.device(config.get("device", "cpu"))
        T = float(params.get("T", 10.0))
        dt = float(params.get("dt", 0.1))
        tau = float(train_cfg.get("tau", 0.005))
        discount_ref = float(params.get("discount_ref", 0.5))
        dt_ref = float(params.get("dt_ref", 0.1))
        rho = -np.log(discount_ref) / dt_ref
        gamma = float(np.exp(-rho * dt))
        dt_fine = float(params.get("dt_fine", 0.1))
        test_inits = params.get(
            "test_inits",
            [[0.25] * 4, [1, 0, 0, 0], [0, 0, 0, 1], [0.5, 0.2, 0.2, 0.1]],
        )

        env = cyber_core.CyberSecEnv(np.array(test_inits[0]), T, dt, _common_noise_sampler(cyber_core, config))
        solver = cyber_core.ForwardBackwardSolver(env, rho, dt_fine=dt_fine)
        true_sols = [solver.solve(np.array(init))[0] for init in test_inits]

        shared_tables = {
            "bench_slices": cyber_core.precompute_bench_values(
                solver,
                params.get("simplex_slices", [0.0, 0.25, 0.5, 0.75, 1.0]),
                int(params.get("simplex_grid_res", 10)),
            )
        }

        actor_hdims = net_cfg.get("actor_hidden_dims", [32, 32])
        critic_hdims = net_cfg.get("critic_hidden_dims", [32, 32])
        actor = cyber_core.Actor(cyber_core.N_STATES, cyber_core.N_STATES, hidden_dims=actor_hdims).to(device)
        critic = cyber_core.Critic(cyber_core.N_STATES, cyber_core.N_STATES, hidden_dims=critic_hdims).to(device)
        target_actor = cyber_core.Actor(cyber_core.N_STATES, cyber_core.N_STATES, hidden_dims=actor_hdims).to(device)
        target_critic = cyber_core.Critic(cyber_core.N_STATES, cyber_core.N_STATES, hidden_dims=critic_hdims).to(device)
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())

        actor_opt = cyber_core.optim.Adam(actor.parameters(), lr=float(train_cfg.get("actor_lr", 1e-3)))
        critic_opt = cyber_core.optim.Adam(critic.parameters(), lr=float(train_cfg.get("critic_lr", 2e-3)))

        logger.log(
            "training",
            f"episodes={train_cfg.get('episodes', 500)}, plot_freq={train_cfg.get('plot_freq', 20)}, device={device}",
        )
        cyber_core.train_policy(
            env,
            actor,
            critic,
            target_actor,
            target_critic,
            actor_opt,
            critic_opt,
            gamma,
            tau,
            cyber_core.ReplayBuffer(
                (cyber_core.N_STATES,),
                cyber_core.N_STATES,
                device,
                int(train_cfg.get("buffer_capacity", 50000)),
                int(train_cfg.get("batch_size", 16)),
            ),
            int(train_cfg.get("episodes", 500)),
            int(train_cfg.get("plot_freq", 20)),
            int(train_cfg.get("print_freq", 10)),
            _initial_distribution_sampler(cyber_core.N_STATES),
            [np.array(init) for init in test_inits],
            device,
            str(output_dir),
            shared_tables,
            true_sols,
            int(train_cfg.get("moving_average_window", 20)),
        )
        logger.log("done", "cybersecurity DDPG completed")

    write_manifest(output_dir, config, status="completed")
    return str(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output_dir = run(load_config(args.config))
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    main()
