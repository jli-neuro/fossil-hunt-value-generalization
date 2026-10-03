#!/usr/bin/env python3
"""Deterministic scientific checks for the archived simulation result."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from analysis import (
    analyze_results,
    compute_statistics,
    mean_run_level_logistic_curve,
)
from self_efficacy_simulation import ValueLearner, run_simulation


EXPECTED_BREADTH_MEAN = np.array(
    [15.3784073533, 18.4284949313, 23.3031601334, 32.4871337257, 46.8187897302]
)
EXPECTED_CONDITION_R = 0.958237315485
EXPECTED_RUN_R = 0.952523611892


def main() -> None:
    test_agent = ValueLearner(n_states=3, w=0.25)
    full_backup = test_agent.weighted_extrema(np.array([0.8, 0.2]))
    assert np.isclose(full_backup, 0.35)
    assert not np.isclose(full_backup, 0.25 * 0.8)

    results = run_simulation(
        [0.0, 0.25, 0.50, 0.75, 1.0],
        n_runs=20,
        n_episodes=100,
        n_states=10,
        alpha=0.1,
        gamma=0.8,
        reward_prob=0.85,
        epsilon=0.20,
        max_steps=200,
        seed=1,
    )
    analysis = analyze_results(results, beta=3.0, normalize=True)
    statistics = compute_statistics(analysis)

    # The learning runs are deterministic at seed 1; the logistic curve fits
    # that summarize them can differ in the last decimals between numerical
    # library builds, so archived values are checked at reporting precision.
    np.testing.assert_allclose(
        statistics["mu_mean"], EXPECTED_BREADTH_MEAN, rtol=0.0, atol=5e-3
    )
    assert np.isclose(
        statistics["condition_mean_correlation_r"], EXPECTED_CONDITION_R, atol=1e-3
    )
    assert np.isclose(statistics["correlation_r"], EXPECTED_RUN_R, atol=1e-3)
    assert statistics["valid_fit_count"] == statistics["total_fit_count"] == 100
    assert int(results["timeout_counts"].sum()) == 0

    # Panel A interpolation must preserve every archived condition-mean node.
    # Panel B must use the archived run-level 4PL fits and remain close to the
    # exact condition-mean propensity points.
    positions = np.asarray(results["positions"], dtype=float)
    distance = positions.max() - positions
    order = np.argsort(distance)
    distance = distance[order]
    smooth_distance = np.linspace(distance.min(), distance.max(), 1801)
    for index in range(len(results["w_values"])):
        mean_value = results["value_functions"][index].mean(axis=0)[order]
        interpolator = PchipInterpolator(distance, mean_value)
        guide = interpolator(smooth_distance)
        assert np.all(np.isfinite(guide))
        np.testing.assert_allclose(interpolator(distance), mean_value, atol=1e-12)
        assert np.isclose(guide[0], mean_value[0], atol=1e-12)
        assert np.isclose(guide[-1], mean_value[-1], atol=1e-12)
        assert guide.min() >= mean_value.min() - 1e-12
        assert guide.max() <= mean_value.max() + 1e-12

        choice_guide = mean_run_level_logistic_curve(
            analysis, index, smooth_distance
        )
        assert np.all(np.isfinite(choice_guide))
        assert np.all(np.diff(choice_guide) <= 1e-12)
        assert np.all((choice_guide >= 0.0) & (choice_guide <= 1.0))

        mean_propensity = np.mean(
            analysis["probabilities"][index], axis=0
        )[order]
        fitted_at_nodes = mean_run_level_logistic_curve(
            analysis, index, distance
        )
        node_rmse = np.sqrt(np.mean((fitted_at_nodes - mean_propensity) ** 2))
        assert node_rmse < 0.04

    interior_q = results["action_values"][:, :, 1:-1]
    assert np.sum(np.min(interior_q, axis=-1) > 0.0) > 0
    assert np.sum(np.ptp(interior_q, axis=-1) > 0.0) > 0
    print("All weighted-extrema simulation checks passed.")


if __name__ == "__main__":
    main()
