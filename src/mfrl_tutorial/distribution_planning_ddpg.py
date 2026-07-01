"""Config-driven DDPG runs for the discrete distribution-planning example."""

from __future__ import annotations

import argparse
from typing import Dict

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


def _initial_distribution_sampler(n_states: int):
    def sampler() -> np.ndarray:
        p = np.random.rand()
        if p < 0.4:
            v = np.random.dirichlet([1.0] * n_states)
        elif p < 0.8:
            v = np.random.dirichlet([0.1] * n_states)
        else:
            v = np.zeros(n_states)
            num_active = np.random.randint(1, 4)
            indices = np.random.choice(n_states, num_active, replace=False)
            v[indices] = np.random.uniform(0.1, 1.0, size=num_active)
        return v / (np.sum(v) + 1e-8)

    return sampler


def _common_noise_sampler(config: Dict):
    mode = config.get("common_noise", {}).get("mode", "small")
    if mode == "none":
        return lambda: 0
    probabilities = config.get("common_noise", {}).get("probabilities", [0.05, 0.9, 0.05])
    return lambda: np.random.choice(np.arange(-1, 2), p=probabilities)


def run(config: Dict) -> str:
    output_dir = create_run_dir(config)
    setup_matplotlib_cache(output_dir)

    from . import distribution_planning_core as planning_core

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
        T = float(params.get("T", 50.0))
        dt = float(params.get("dt", 1.0))
        gamma = float(train_cfg.get("gamma", 0.99))
        tau = float(train_cfg.get("tau", 0.005))
        test_inits = params.get(
            "test_inits",
            [
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.4, 0.2, 0.1],
                [0.1, 0.2, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

        env = planning_core.DiscretePlanningEnv(np.array(test_inits[0]), T, dt, _common_noise_sampler(config))

        actor = planning_core.Actor(
            planning_core.N_STATES,
            planning_core.N_STATES,
            planning_core.N_ACTIONS_PER_STATE,
            hidden_dims=net_cfg.get("actor_hidden_dims", [128, 128, 128]),
        ).to(device)
        critic = planning_core.Critic(
            planning_core.N_STATES,
            planning_core.N_STATES * planning_core.N_ACTIONS_PER_STATE,
            state_hidden_dims=net_cfg.get("critic_state_hidden_dims", [128]),
            action_hidden_dims=net_cfg.get("critic_action_hidden_dims", [128]),
            combined_hidden_dims=net_cfg.get("critic_combined_hidden_dims", [128, 128]),
        ).to(device)
        target_actor = planning_core.Actor(
            planning_core.N_STATES,
            planning_core.N_STATES,
            planning_core.N_ACTIONS_PER_STATE,
            hidden_dims=net_cfg.get("actor_hidden_dims", [128, 128, 128]),
        ).to(device)
        target_critic = planning_core.Critic(
            planning_core.N_STATES,
            planning_core.N_STATES * planning_core.N_ACTIONS_PER_STATE,
            state_hidden_dims=net_cfg.get("critic_state_hidden_dims", [128]),
            action_hidden_dims=net_cfg.get("critic_action_hidden_dims", [128]),
            combined_hidden_dims=net_cfg.get("critic_combined_hidden_dims", [128, 128]),
        ).to(device)
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())

        actor_opt = planning_core.optim.Adam(actor.parameters(), lr=float(train_cfg.get("actor_lr", 5e-4)))
        critic_opt = planning_core.optim.Adam(critic.parameters(), lr=float(train_cfg.get("critic_lr", 1e-3)))

        logger.log(
            "training",
            f"episodes={train_cfg.get('episodes', 3000)}, plot_freq={train_cfg.get('plot_freq', 100)}, device={device}",
        )
        planning_core.train_policy(
            env,
            actor,
            critic,
            target_actor,
            target_critic,
            actor_opt,
            critic_opt,
            gamma,
            tau,
            planning_core.ReplayBuffer(
                planning_core.N_STATES,
                (planning_core.N_STATES, planning_core.N_ACTIONS_PER_STATE),
                device,
                int(train_cfg.get("buffer_capacity", 50000)),
                int(train_cfg.get("batch_size", 128)),
            ),
            int(train_cfg.get("episodes", 3000)),
            int(train_cfg.get("plot_freq", 100)),
            int(train_cfg.get("print_freq", 20)),
            _initial_distribution_sampler(planning_core.N_STATES),
            [np.array(init) for init in test_inits],
            device,
            str(output_dir),
            save_freq=int(train_cfg.get("save_freq", 500)),
            window_size=int(train_cfg.get("moving_average_window", 20)),
        )
        logger.log("done", "distribution-planning DDPG completed")

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
