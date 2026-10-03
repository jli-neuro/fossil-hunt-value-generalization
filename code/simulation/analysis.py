"""Analysis and plotting for the Zorowitz weighted-extrema simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def softmax_probability(
    values: np.ndarray,
    beta: float = 3.0,
    normalize: bool = True,
) -> np.ndarray:
    """Map state values to a binary reward-choice propensity.

    Values are compared with a zero-valued reference through a logistic
    choice rule.  The legacy simulation then endpoint-rescaled each run to
    span 0--1.  That rescaling is retained for direct procedural replication,
    but the resulting quantity is a normalized choice propensity rather than
    a calibrated probability.
    """
    values = np.asarray(values, dtype=float)
    # Retain the prior implementation's algebraic form (rather than the
    # numerically equivalent expit form) for bit-level result replication.
    exp_value = np.exp(beta * values)
    probability = exp_value / (exp_value + 1.0)

    if normalize:
        span = float(np.max(probability) - np.min(probability))
        if not np.isfinite(span) or span <= np.finfo(float).eps:
            return np.full_like(probability, np.nan)
        probability = (probability - np.min(probability)) / span

    return probability


def logistic_4param(
    x: np.ndarray,
    gamma: float,
    lam: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Four-parameter increasing logistic on the physical coordinate."""
    return gamma + (lam - gamma) / (1.0 + np.exp(-(x - mu) / sigma))


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    bounds: Optional[Tuple] = None,
) -> Tuple[np.ndarray, float]:
    """Fit the four-parameter logistic and return parameters and R-squared."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_range = float(x.max() - x.min())
    if bounds is None:
        bounds = ([0.0, 0.5, x.min(), 0.1], [0.5, 1.0, x.max(), x_range])

    lower, upper = bounds
    p0 = [
        float(np.clip(0.0, lower[0], upper[0])),
        float(np.clip(1.0, lower[1], upper[1])),
        float(np.clip((x.min() + x.max()) / 2.0, lower[2], upper[2])),
        float(np.clip(x_range / 4.0, lower[3], upper[3])),
    ]

    if np.any(~np.isfinite(y)):
        return np.full(4, np.nan), np.nan

    try:
        parameters, _ = curve_fit(
            logistic_4param,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
    except (RuntimeError, ValueError):
        return np.full(4, np.nan), np.nan

    fitted = logistic_4param(x, *parameters)
    residual_ss = float(np.sum((y - fitted) ** 2))
    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan
    return parameters, r_squared


def analyze_results(
    results: Dict,
    beta: float = 3.0,
    normalize: bool = True,
) -> Dict:
    """Transform learned values and fit an individual 4PL to every run."""
    w_values = np.asarray(results["w_values"], dtype=float)
    value_functions = np.asarray(results["value_functions"], dtype=float)
    positions = np.asarray(results["positions"], dtype=float)
    n_w, n_runs, n_states = value_functions.shape

    analysis = {
        "w_values": w_values,
        "positions": positions,
        "mu_values": np.full((n_w, n_runs), np.nan),
        "all_params": np.full((n_w, n_runs, 4), np.nan),
        "r_squared": np.full((n_w, n_runs), np.nan),
        "probabilities": np.full((n_w, n_runs, n_states), np.nan),
        "beta": float(beta),
        "endpoint_rescaled": bool(normalize),
    }

    # Retain the prior simulation's nearly fixed endpoint-asymptote fit so the
    # only substantive change is the learning operator/environment.
    position_max = float(np.max(positions))
    bounds = ([0.0, 0.99, 0.0, 0.1], [0.01, 1.0, position_max, position_max])

    for w_index in range(n_w):
        for run_index in range(n_runs):
            propensity = softmax_probability(
                value_functions[w_index, run_index],
                beta=beta,
                normalize=normalize,
            )
            parameters, r_squared = fit_logistic(
                positions,
                propensity,
                bounds=bounds,
            )
            analysis["probabilities"][w_index, run_index] = propensity
            analysis["all_params"][w_index, run_index] = parameters
            analysis["mu_values"][w_index, run_index] = parameters[2]
            analysis["r_squared"][w_index, run_index] = r_squared

    # Physical position 90 is rewarded.  Report distance from reward so zero
    # is the rewarded endpoint and larger mu means broader generalization.
    analysis["generalization_breadth_values"] = (
        position_max - analysis["mu_values"]
    )
    return analysis


def compute_statistics(analysis: Dict) -> Dict:
    """Summarize breadth and its relation to the manipulated w levels."""
    w_values = np.asarray(analysis["w_values"], dtype=float)
    raw_mu = np.asarray(analysis["mu_values"], dtype=float)
    breadth = np.asarray(analysis["generalization_breadth_values"], dtype=float)

    breadth_mean = np.nanmean(breadth, axis=1)
    # ddof=0 intentionally reproduces the previous simulation's Monte Carlo
    # SEM convention; the convention is recorded in the output manifest.
    breadth_sem = np.nanstd(breadth, axis=1, ddof=0) / np.sqrt(breadth.shape[1])
    raw_mu_mean = np.nanmean(raw_mu, axis=1)
    raw_mu_sem = np.nanstd(raw_mu, axis=1, ddof=0) / np.sqrt(raw_mu.shape[1])

    flat_w = np.repeat(w_values, breadth.shape[1])
    flat_breadth = breadth.ravel()
    valid = np.isfinite(flat_breadth)
    run_result = pearsonr(flat_w[valid], flat_breadth[valid])
    mean_result = pearsonr(w_values, breadth_mean)

    return {
        "w_values": w_values,
        "mu_mean": breadth_mean,
        "mu_sem": breadth_sem,
        "raw_mu_mean": raw_mu_mean,
        "raw_mu_sem": raw_mu_sem,
        "correlation_r": float(run_result.statistic),
        "correlation_p": float(run_result.pvalue),
        "correlation_n": int(np.sum(valid)),
        "condition_mean_correlation_r": float(mean_result.statistic),
        "condition_mean_correlation_p": float(mean_result.pvalue),
        "condition_mean_correlation_n": int(len(w_values)),
        "r_squared_mean": float(np.nanmean(analysis["r_squared"])),
        "valid_fit_count": int(np.sum(np.isfinite(raw_mu))),
        "total_fit_count": int(raw_mu.size),
        "sem_ddof": 0,
    }


def _report_axis(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distance = float(np.max(positions)) - positions
    order = np.argsort(distance)
    return distance[order], order


def _save_figure(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def mean_run_level_logistic_curve(
    analysis: Dict,
    condition_index: int,
    distance: np.ndarray,
) -> np.ndarray:
    """Average the archived run-specific 4PL fits on the report coordinate."""
    parameters = np.asarray(analysis["all_params"])[condition_index]
    parameters = parameters[np.all(np.isfinite(parameters), axis=1)]
    if not len(parameters):
        raise ValueError("No valid run-level logistic fits for plotting.")

    physical_position = float(np.max(analysis["positions"])) - np.asarray(
        distance, dtype=float
    )
    with np.errstate(over="ignore"):
        curves = np.asarray(
            [logistic_4param(physical_position, *row) for row in parameters]
        )
    return np.mean(curves, axis=0)


def plot_value_landscapes(
    results: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 6),
):
    """Plot mean greedy state value in the distance-from-reward coordinate."""
    w_values = np.asarray(results["w_values"])
    positions = np.asarray(results["positions"])
    distance, order = _report_axis(positions)
    values = np.asarray(results["value_functions"])
    colors = plt.cm.viridis(np.linspace(0, 1, len(w_values)))
    fig, axis = plt.subplots(figsize=figsize)

    smooth_distance = np.linspace(distance.min(), distance.max(), 1801)
    for index, weight in enumerate(w_values):
        mean_value = values[index].mean(axis=0)[order]
        sem_value = values[index].std(axis=0, ddof=0)[order] / np.sqrt(values.shape[1])
        smooth_value = PchipInterpolator(distance, mean_value)(smooth_distance)
        axis.plot(smooth_distance, smooth_value, color=colors[index],
                  label=rf"$w={weight:g}$")
        axis.scatter(distance, mean_value, s=24,
                     color=colors[index], zorder=3)
        axis.fill_between(distance, mean_value - sem_value,
                          mean_value + sem_value,
                          color=colors[index], alpha=0.15)
    axis.scatter([distance.min(), distance.max()], [0.85, 0.0], marker="D",
                 s=34, color="black", zorder=4,
                 label="Expected terminal outcomes")

    axis.set_xlabel("Distance from rewarded endpoint (simulation units)")
    axis.set_ylabel(r"Learned state value $V(s)$")
    axis.set_title("Learned state values")
    axis.legend(title="Self-efficacy weight")
    _clean_axis(axis)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, axis


def plot_logistic_fits(
    analysis: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple = (10, 6),
):
    """Plot exact propensities with mean archived run-level 4PL fits."""
    w_values = np.asarray(analysis["w_values"])
    positions = np.asarray(analysis["positions"])
    distance, order = _report_axis(positions)
    smooth_distance = np.linspace(distance.min(), distance.max(), 1801)
    colors = plt.cm.viridis(np.linspace(0, 1, len(w_values)))
    fig, axis = plt.subplots(figsize=figsize)

    for index, weight in enumerate(w_values):
        mean_propensity = np.nanmean(analysis["probabilities"][index], axis=0)[order]
        sem_propensity = (
            np.nanstd(analysis["probabilities"][index], axis=0, ddof=0)[order]
            / np.sqrt(analysis["probabilities"].shape[1])
        )
        mean_breadth = np.nanmean(analysis["generalization_breadth_values"][index])
        mean_prediction = mean_run_level_logistic_curve(
            analysis, index, smooth_distance
        )

        axis.errorbar(distance, mean_propensity, yerr=sem_propensity, fmt="o",
                      color=colors[index], capsize=2, alpha=0.8)
        axis.plot(smooth_distance, mean_prediction, color=colors[index], linewidth=2,
                  label=rf"$w={weight:g}$ ($\mu_{{\mathrm{{sim}}}}={mean_breadth:.1f}$)")

    axis.axhline(0.5, color="0.65", linestyle="--", linewidth=1)
    axis.set_xlabel("Distance from rewarded endpoint (simulation units)")
    axis.set_ylabel("Endpoint-rescaled reward-choice propensity")
    axis.set_ylim(-0.05, 1.05)
    axis.set_title("Model-implied choice gradients")
    axis.legend(title="Self-efficacy weight", loc="upper right")
    _clean_axis(axis)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, axis


def plot_w_vs_mu(
    stats: Dict,
    analysis: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple = (8, 6),
):
    """Plot run-level breadth and the relationship across condition means."""
    w_values = np.asarray(stats["w_values"])
    breadth = np.asarray(analysis["generalization_breadth_values"])
    fig, axis = plt.subplots(figsize=figsize)
    jitter_rng = np.random.default_rng(20260819)

    for index, weight in enumerate(w_values):
        jitter = jitter_rng.normal(0.0, 0.012, breadth.shape[1])
        axis.scatter(weight + jitter, breadth[index], alpha=0.30,
                     color="steelblue", s=25)

    axis.plot(w_values, stats["mu_mean"], color="darkblue", linewidth=1.5,
              zorder=4)
    axis.errorbar(w_values, stats["mu_mean"], yerr=stats["mu_sem"], fmt="o",
                  color="darkblue", capsize=4, linewidth=1.5, zorder=5)
    axis.set_xlabel(r"Self-efficacy weight ($w$)")
    axis.set_ylabel(r"Simulated generalization breadth ($\mu_{\mathrm{sim}}$)")
    axis.set_ylim(bottom=0.0)
    axis.set_title("Self-efficacy and generalization breadth")
    _clean_axis(axis)
    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig, axis


def plot_combined_panel(
    results: Dict,
    analysis: Dict,
    stats: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple = (18, 5.5),
):
    """Create the three-panel manuscript simulation figure."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, facecolor="white")
    w_values = np.asarray(results["w_values"])
    positions = np.asarray(results["positions"])
    distance, order = _report_axis(positions)
    colors = plt.cm.viridis(np.linspace(0, 1, len(w_values)))

    axis = axes[0]
    smooth_distance = np.linspace(distance.min(), distance.max(), 1801)
    for index, weight in enumerate(w_values):
        mean_value = results["value_functions"][index].mean(axis=0)[order]
        smooth_value = PchipInterpolator(distance, mean_value)(smooth_distance)
        axis.plot(smooth_distance, smooth_value, color=colors[index],
                  linewidth=2, label=rf"$w={weight:g}$")
        axis.scatter(distance, mean_value, s=24,
                     color=colors[index], zorder=3)
    axis.scatter([distance.min(), distance.max()], [0.85, 0.0], marker="D",
                 s=34, color="black", zorder=4,
                 label="Expected terminal outcomes")
    axis.set_xlabel("Distance from rewarded endpoint\n(simulation units)")
    axis.set_ylabel(r"Learned state value $V(s)$")
    axis.legend(title="Self-efficacy", fontsize=9, title_fontsize=10)
    _clean_axis(axis)

    axis = axes[1]
    for index, weight in enumerate(w_values):
        mean_propensity = np.nanmean(analysis["probabilities"][index], axis=0)[order]
        mean_breadth = np.nanmean(analysis["generalization_breadth_values"][index])
        axis.scatter(distance, mean_propensity, color=colors[index], s=30, zorder=3)
        axis.plot(smooth_distance,
                  mean_run_level_logistic_curve(
                      analysis, index, smooth_distance
                  ),
                  color=colors[index],
                  linewidth=2,
                  label=rf"$w={weight:g}$ ($\mu_{{\mathrm{{sim}}}}={mean_breadth:.1f}$)")
    axis.axhline(0.5, color="0.65", linestyle="--", linewidth=1)
    axis.set_xlabel("Distance from rewarded endpoint\n(simulation units)")
    axis.set_ylabel("Endpoint-rescaled\nreward-choice propensity")
    axis.set_ylim(-0.05, 1.05)
    axis.legend(title="Self-efficacy", fontsize=8.5, title_fontsize=9.5,
                loc="upper right")
    _clean_axis(axis)

    axis = axes[2]
    breadth = analysis["generalization_breadth_values"]
    jitter_rng = np.random.default_rng(20260819)
    for index, weight in enumerate(w_values):
        jitter = jitter_rng.normal(0.0, 0.012, breadth.shape[1])
        axis.scatter(weight + jitter, breadth[index], alpha=0.30,
                     color="steelblue", s=25)
    axis.plot(w_values, stats["mu_mean"], color="darkblue", linewidth=1.5,
              zorder=4)
    axis.errorbar(w_values, stats["mu_mean"], yerr=stats["mu_sem"], fmt="o",
                  color="darkblue", markersize=8, capsize=4, linewidth=1.5,
                  zorder=5)
    axis.set_xlabel(r"Self-efficacy weight ($w$)")
    axis.set_ylabel(r"Simulated generalization breadth ($\mu_{\mathrm{sim}}$)")
    axis.set_ylim(bottom=0.0)
    _clean_axis(axis)

    for axis, label in zip(axes, "ABC"):
        axis.text(
            -0.15,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
        )

    fig.tight_layout(w_pad=2.5)
    _save_figure(fig, save_path)
    return fig, axes


def plot_summary(results: Dict, analysis: Dict, stats: Dict, save_dir: str) -> None:
    """Generate standalone and combined PNG/PDF outputs."""
    output = Path(save_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_value_landscapes(results, str(output / "simulation_value_landscapes.png"))
    plot_logistic_fits(analysis, str(output / "simulation_logistic_fits.png"))
    plot_w_vs_mu(stats, analysis, str(output / "simulation_w_vs_mu.png"))
    plot_combined_panel(
        results,
        analysis,
        stats,
        str(output / "simulation_combined_panel.png"),
    )
    plt.close("all")
