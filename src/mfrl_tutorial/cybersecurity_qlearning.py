"""Config-driven tabular Q-learning runs for the cybersecurity example."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import numpy as np

from .utils import (
    configure_seed,
    create_run_dir,
    load_config,
    setup_matplotlib_cache,
    tee_output,
    write_manifest,
    write_resolved_config,
    RunLogger,
)


def _common_noise_sampler(cyber_core: Any, config: Dict[str, Any]):
    cn = config.get("common_noise", {})
    if not cn.get("enabled", False):
        return lambda x: x
    sigma = float(cn.get("sigma", 0.05))
    low = float(cn.get("clip_low_multiplier", 0.7)) * cyber_core.v_H_DEFAULT
    high = float(cn.get("clip_high_multiplier", 1.3)) * cyber_core.v_H_DEFAULT
    return lambda x: np.clip(x + np.random.normal(0.0, sigma), low, high)


def run(config: Dict[str, Any]) -> str:
    output_dir = create_run_dir(config)
    setup_matplotlib_cache(output_dir)

    from . import cybersecurity_qlearning_core as cyber_core

    configure_seed(config.get("seed"))
    write_resolved_config(config, output_dir)
    run_log = output_dir / "run.log"
    logger = RunLogger(run_log)

    with tee_output(run_log):
        logger.log("setup", f"output_dir={output_dir}")
        params = config.get("model", {})
        T = float(params.get("T", 10.0))
        dt = float(params.get("dt", 0.2))
        n_steps_state = int(params.get("n_steps_state", 30))
        n_actions_level = int(params.get("n_actions_level", 2))
        dt_fine = float(params.get("dt_fine", 0.1))
        discount_ref = float(params.get("discount_ref", 0.5))
        dt_ref = float(params.get("dt_ref", 0.1))

        env = cyber_core.CyberSecEnvContinuous(T, dt, _common_noise_sampler(cyber_core, config), n_actions_level)
        shared_tables = cyber_core.precompute_tables(env, {"n_steps_state": n_steps_state})

        rho = -np.log(discount_ref) / dt_ref
        discount_val = float(np.exp(-rho * dt))
        logger.log(
            "setup",
            f"T={T}, dt={dt}, n_steps_state={n_steps_state}, actions={n_actions_level}, discount={discount_val:.6f}",
        )

        ode_sol = cyber_core.ForwardBackwardSolver(env, rho, dt_fine=dt_fine)
        eval_inits = [
            np.array([0.25] * 4),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.5, 0.2, 0.2, 0.1]),
        ]
        ode_trajs = [ode_sol.solve(m0)[0] for m0 in eval_inits]
        shared_tables["bench_slices"] = cyber_core.precompute_bench_values(
            ode_sol,
            params.get("simplex_slices", [0.0, 0.25, 0.5, 0.75, 1.0]),
            int(params.get("simplex_grid_res", 10)),
        )

        solvers = []
        for solver_config in config.get("solvers", []):
            cfg = dict(solver_config)
            cfg.setdefault("discount", discount_val)
            cfg.setdefault("eval_freq", cfg.get("plot_freq", 1000))
            logger.log("solver", f"{cfg['name']} mode={cfg['mode']} iters={cfg['total_iters']}")
            solvers.append(cyber_core.QlearningSolver(cfg["name"], env, shared_tables, cfg, str(output_dir)))

        for solver in solvers:
            solver.train(ode_sol=ode_sol, ode_trajs=ode_trajs)

        logger.log("done", "cybersecurity Q-learning completed")

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
