"""Config-driven lightweight LQ-MFC data generation demo."""

from __future__ import annotations

import argparse
from pathlib import Path
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


def _task_name(task: Dict[str, Any]) -> str:
    solver_kwargs = task.get("solver_kwargs") or {}
    return str(task.get("name", task.get("type", "task"))).format(**solver_kwargs)


def _task_args(task: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "opt_kwargs": task.get("opt_kwargs") or {},
        "solver_kwargs": task.get("solver_kwargs") or {},
    }


def _style_map(task_names: list[str]) -> dict[str, dict[str, Any]]:
    colors = ["#1f77b4", "#9c27b0", "#2ca02c", "#d62728", "#ff7f0e", "#8c564b"]
    linestyles = ["-", "--", "-.", ":"]
    styles = {}
    for idx, name in enumerate(task_names):
        styles[name] = {
            "color": colors[idx % len(colors)],
            "ls": linestyles[idx % len(linestyles)],
            "linewidth": 2,
        }
        styles.setdefault(name.split(" (")[0], styles[name])
    return styles


def run(config: dict) -> str:
    output_dir = create_run_dir(config)
    setup_matplotlib_cache(output_dir)
    configure_seed(config.get("seed"))
    write_resolved_config(config, output_dir)
    run_log = output_dir / "run.log"
    logger = RunLogger(run_log)

    from . import lq_mfc_policy_gradient as lq_pg

    tasks = config.get("tasks") or []
    seeds = config.get("seeds") or []
    n_iter = int(config.get("n_iter", 1000))
    plot_every = int(config.get("plot_every", max(1, n_iter)))
    show_plots = bool(config.get("show_plots", False))
    model_params = config.get("model_params") or {
        "a": 0.5,
        "a_bar": 0.5,
        "b": 0.5,
        "b_bar": 0.5,
        "q": 0.5,
        "q_bar": 0.5,
        "r": 0.5,
        "r_bar": 0.5,
        "gamma": 0.9,
    }

    with tee_output(run_log):
        logger.log("setup", f"output_dir={output_dir}")
        logger.log("setup", f"n_iter={n_iter}, plot_every={plot_every}, seeds={seeds}")
        logger.log("setup", f"tasks={[_task_name(task) for task in tasks]}")

        model = lq_pg.LQMFCModel(model_params)
        k_star, l_star = model.solve_riccati()
        optimal_cost = model.compute_cost(k_star, l_star)
        styles = _style_map([_task_name(task) for task in tasks])

        all_results: dict[str, list[dict[str, Any]]] = {}
        data_dir = Path(output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            task_type = task["type"]
            task_name = _task_name(task)
            all_results[task_name] = []
            logger.log("task", f"{task_name} type={task_type}")

            if task_type == "Exact":
                solver_class = lq_pg.ExactPGSolver
                solver_seeds = [None]
            elif task_type == "MKV":
                solver_class = lq_pg.ModelFreeMKVSolver
                solver_seeds = seeds
            elif task_type == "Pop":
                solver_class = lq_pg.ModelFreePopSolver
                solver_seeds = seeds
            else:
                raise ValueError(f"Unknown LQ task type: {task_type}")

            for seed in solver_seeds:
                label = f"{task_name} (Seed {seed})" if seed is not None else task_name
                styles[label] = styles[task_name]
                logger.log("run", f"{label}")
                history = lq_pg.run_single_solver(
                    model,
                    solver_class,
                    _task_args(task),
                    n_iter,
                    seed,
                    str(output_dir),
                    label,
                    styles,
                    k_star,
                    l_star,
                    optimal_cost,
                    show_plots,
                    plot_every,
                )
                all_results[task_name].append(history)

            if len(all_results[task_name]) > 1:
                aggregate = _aggregate_histories(all_results[task_name])
                np.save(data_dir / f"aggregate_{_safe_name(task_name)}.npy", aggregate)
                lq_pg.plot_layout(
                    [aggregate],
                    [task_name],
                    styles,
                    k_star,
                    l_star,
                    optimal_cost,
                    f"Aggregated Result: {task_name} ({len(solver_seeds)} Seeds)",
                    output_dir=str(output_dir),
                    filename_prefix=f"agg_{_safe_name(task_name)}",
                    show_plots=show_plots,
                )

        comparison_histories = []
        comparison_labels = []
        for task_name, histories in all_results.items():
            if not histories:
                continue
            if len(histories) > 1:
                comparison_histories.append(_aggregate_histories(histories))
            else:
                comparison_histories.append(histories[0])
            comparison_labels.append(task_name)

        if comparison_histories:
            lq_pg.plot_layout(
                comparison_histories,
                comparison_labels,
                styles,
                k_star,
                l_star,
                optimal_cost,
                "Comparison of Solvers",
                output_dir=str(output_dir),
                filename_prefix="final_comparison",
                show_plots=show_plots,
            )
        logger.log("done", "LQ data generation demo completed")

    write_manifest(
        output_dir,
        config,
        status="completed",
        extra={"lq_optimal": {"K": k_star, "L": l_star, "cost": optimal_cost}},
    )
    return str(output_dir)


def _aggregate_histories(histories: list[dict[str, Any]]) -> dict[str, Any]:
    arrays = {key: np.array([history[key] for history in histories]) for key in histories[0].keys()}
    aggregate = {f"{key}_mean": np.mean(value, axis=0) for key, value in arrays.items()}
    aggregate.update({f"{key}_std": np.std(value, axis=0) for key, value in arrays.items()})
    return aggregate


def _safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output_dir = run(load_config(args.config))
    print(f"Output written to {output_dir}")


if __name__ == "__main__":
    main()
