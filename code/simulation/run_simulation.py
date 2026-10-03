#!/usr/bin/env python3
"""Run, archive, and plot the faithful Zorowitz weighted-extrema simulation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from analysis import analyze_results, compute_statistics, plot_summary
from self_efficacy_simulation import run_simulation


W_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]
N_RUNS = 20
N_EPISODES = 100
N_STATES = 10
ALPHA = 0.1
GAMMA = 0.8
REWARD_PROB = 0.85
EPSILON = 0.20
MAX_STEPS = 200
BETA = 3.0
SEED = 1


def _write_run_level_csv(path: Path, results: dict, analysis: dict) -> None:
    fields = [
        "w",
        "run",
        "raw_physical_midpoint",
        "generalization_breadth_mu",
        "logistic_r_squared",
        "mean_episode_reward",
        "mean_episode_length",
        "max_episode_length",
        "timeout_count",
        "positive_qmin_interior_states",
        "nonzero_action_gap_interior_states",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for w_index, weight in enumerate(results["w_values"]):
            for run_index in range(results["value_functions"].shape[1]):
                interior_q = results["action_values"][w_index, run_index, 1:-1]
                writer.writerow(
                    {
                        "w": f"{weight:.8g}",
                        "run": run_index + 1,
                        "raw_physical_midpoint": (
                            f"{analysis['mu_values'][w_index, run_index]:.12g}"
                        ),
                        "generalization_breadth_mu": (
                            f"{analysis['generalization_breadth_values'][w_index, run_index]:.12g}"
                        ),
                        "logistic_r_squared": (
                            f"{analysis['r_squared'][w_index, run_index]:.12g}"
                        ),
                        "mean_episode_reward": (
                            f"{results['episode_rewards'][w_index, run_index].mean():.12g}"
                        ),
                        "mean_episode_length": (
                            f"{results['episode_lengths'][w_index, run_index].mean():.12g}"
                        ),
                        "max_episode_length": int(
                            results["episode_lengths"][w_index, run_index].max()
                        ),
                        "timeout_count": int(
                            results["timeout_counts"][w_index, run_index]
                        ),
                        "positive_qmin_interior_states": int(
                            np.sum(np.min(interior_q, axis=1) > 0.0)
                        ),
                        "nonzero_action_gap_interior_states": int(
                            np.sum(np.ptp(interior_q, axis=1) > 0.0)
                        ),
                    }
                )


def _write_summary_csv(path: Path, stats: dict, analysis: dict) -> None:
    fields = [
        "w",
        "n_runs",
        "raw_physical_midpoint_mean",
        "raw_physical_midpoint_sem",
        "generalization_breadth_mean",
        "generalization_breadth_sem",
        "mean_logistic_r_squared",
        "condition_mean_correlation_r",
        "condition_mean_correlation_p",
        "pooled_run_correlation_r",
        "pooled_run_correlation_p",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, weight in enumerate(stats["w_values"]):
            writer.writerow(
                {
                    "w": f"{weight:.8g}",
                    "n_runs": analysis["mu_values"].shape[1],
                    "raw_physical_midpoint_mean": f"{stats['raw_mu_mean'][index]:.12g}",
                    "raw_physical_midpoint_sem": f"{stats['raw_mu_sem'][index]:.12g}",
                    "generalization_breadth_mean": f"{stats['mu_mean'][index]:.12g}",
                    "generalization_breadth_sem": f"{stats['mu_sem'][index]:.12g}",
                    "mean_logistic_r_squared": (
                        f"{np.nanmean(analysis['r_squared'][index]):.12g}"
                    ),
                    "condition_mean_correlation_r": (
                        f"{stats['condition_mean_correlation_r']:.12g}"
                    ),
                    "condition_mean_correlation_p": (
                        f"{stats['condition_mean_correlation_p']:.12g}"
                    ),
                    "pooled_run_correlation_r": f"{stats['correlation_r']:.12g}",
                    "pooled_run_correlation_p": f"{stats['correlation_p']:.12g}",
                }
            )


def main():
    script_dir = Path(__file__).resolve().parent
    figure_dir = script_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Original Zorowitz weighted-extrema simulation")
    print("=" * 72)
    print(f"w values: {W_VALUES}")
    print(f"runs x episodes: {N_RUNS} x {N_EPISODES}")
    print(
        f"alpha={ALPHA}, gamma={GAMMA}, epsilon={EPSILON}, "
        f"reward probability={REWARD_PROB}, beta={BETA}, seed={SEED}"
    )

    results = run_simulation(
        w_values=W_VALUES,
        n_runs=N_RUNS,
        n_episodes=N_EPISODES,
        n_states=N_STATES,
        alpha=ALPHA,
        gamma=GAMMA,
        reward_prob=REWARD_PROB,
        epsilon=EPSILON,
        max_steps=MAX_STEPS,
        seed=SEED,
    )
    analysis = analyze_results(results, beta=BETA, normalize=True)
    stats = compute_statistics(analysis)

    np.savez_compressed(
        script_dir / "simulation_results.npz",
        w_values=results["w_values"],
        positions=results["positions"],
        value_functions=results["value_functions"],
        action_values=results["action_values"],
        episode_rewards=results["episode_rewards"],
        episode_lengths=results["episode_lengths"],
        timeout_counts=results["timeout_counts"],
        propensities=analysis["probabilities"],
        logistic_parameters=analysis["all_params"],
        logistic_r_squared=analysis["r_squared"],
        generalization_breadth=analysis["generalization_breadth_values"],
    )
    _write_run_level_csv(script_dir / "simulation_run_level.csv", results, analysis)
    _write_summary_csv(script_dir / "simulation_summary.csv", stats, analysis)

    interior_q = results["action_values"][:, :, 1:-1]
    manifest = {
        "model": "Zorowitz weighted-extrema temporal-difference learner",
        "citation": "Zorowitz, Momennejad, and Daw (2020), Eq. 3",
        "operator": results["backup_operator"],
        "reported_state_value": results["reported_state_value"],
        "environment": {
            "states": N_STATES,
            "actions": ["left", "right"],
            "episode_start": "uniform over interior states 1..8",
            "nonrewarded_terminal": {"position": 0, "reward": 0.0},
            "rewarded_terminal": {
                "position": 90,
                "reward_distribution": f"Bernoulli({REWARD_PROB})",
            },
            "terminal_plot_anchors": [0.0, REWARD_PROB],
        },
        "configuration": {
            "w_values": W_VALUES,
            "n_runs": N_RUNS,
            "n_episodes": N_EPISODES,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon": EPSILON,
            "max_steps": MAX_STEPS,
            "beta": BETA,
            "seed": SEED,
        },
        "choice_mapping": (
            "binary softmax against zero, followed by within-run endpoint "
            "rescaling to [0,1] (retained from the prior plotting pipeline)"
        ),
        "curve_fit": {
            "function": "4-parameter logistic on physical positions 0..90",
            "bounds": {
                "lower_asymptote": [0.0, 0.01],
                "upper_asymptote": [0.99, 1.0],
                "midpoint": [0.0, 90.0],
                "scale": [0.1, 90.0],
            },
            "reporting_transform": (
                "mu_sim = 90 - physical-coordinate midpoint"
            ),
        },
        "sem_convention": "population SD / sqrt(20), ddof=0 (legacy replication)",
        "verification": {
            "valid_fits": stats["valid_fit_count"],
            "total_fits": stats["total_fit_count"],
            "timeouts": int(results["timeout_counts"].sum()),
            "maximum_episode_length": int(results["episode_lengths"].max()),
            "positive_qmin_interior_run_states": int(
                np.sum(np.min(interior_q, axis=-1) > 0.0)
            ),
            "nonzero_action_gap_interior_run_states": int(
                np.sum(np.ptp(interior_q, axis=-1) > 0.0)
            ),
        },
        "statistics": {
            "condition_mean_correlation_r": stats["condition_mean_correlation_r"],
            "condition_mean_correlation_p": stats["condition_mean_correlation_p"],
            "pooled_run_correlation_r": stats["correlation_r"],
            "pooled_run_correlation_p": stats["correlation_p"],
            "mean_logistic_r_squared": stats["r_squared_mean"],
        },
    }
    (script_dir / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_summary(results, analysis, stats, str(figure_dir))
    # Keep both the legacy folder filename and the manuscript-explicit alias;
    # they are byte-identical and use the reflected reporting coordinate.
    shutil.copy2(
        figure_dir / "simulation_combined_panel.png",
        figure_dir / "simulation_combined_panel_reflected.png",
    )
    shutil.copy2(
        figure_dir / "simulation_combined_panel.pdf",
        figure_dir / "simulation_combined_panel_reflected.pdf",
    )

    print("\nGeneralization breadth (mean +/- SEM):")
    for index, weight in enumerate(stats["w_values"]):
        print(
            f"  w={weight:.2f}: "
            f"{stats['mu_mean'][index]:.4f} +/- {stats['mu_sem'][index]:.4f}"
        )
    print(
        "\nAcross five condition means: "
        f"r={stats['condition_mean_correlation_r']:.6f}, "
        f"p={stats['condition_mean_correlation_p']:.8f}"
    )
    print(
        "Across 100 stochastic runs (descriptive): "
        f"r={stats['correlation_r']:.6f}, p={stats['correlation_p']:.6g}"
    )
    print(f"Mean logistic R^2={stats['r_squared_mean']:.6f}")
    print(f"Timeouts={int(results['timeout_counts'].sum())}")
    print(f"Outputs written to {script_dir}")
    return results, analysis, stats


if __name__ == "__main__":
    results, analysis, stats = main()
