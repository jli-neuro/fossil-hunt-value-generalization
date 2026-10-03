"""Shared definitions for the Fossil Hunt / positive overgeneralization (POG) analyses.

All task quantities use the reporting coordinate of the manuscript: ``d`` is the
distance of a test stimulus from the participant's trained rewarded exemplar
(0 = rewarded endpoint, 90 = nonrewarded endpoint), in steps of 10.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("POG_DATA_DIR", ROOT / "data"))
DERIVED_DIR = Path(os.environ.get("POG_DERIVED_DIR", ROOT / "derived"))
RESULTS_DIR = Path(os.environ.get("POG_RESULTS_DIR", ROOT / "results"))

# Response-time analysis settings (Methods, "Response time analysis")
RT_MIN, RT_MAX = 0.2, 8.0          # seconds, inclusive analysis window
RT_MIN_VALID_TRIALS = 72           # of 90 test trials
RT_MIN_ADEQUATE_BINS = 8           # stimulus positions with enough valid RTs
RT_MIN_TRIALS_PER_BIN = 4

N_BOOT_SEED = 20260819             # seed for the participant-matched recovery

# Participant-level psychometric fits as reported in the manuscript (supplied
# with the data on request; see data/README.md).
REPORTED_FITS_FILE = "reported_psychometric_fits.csv"
PSYCHOMETRIC_PARAMS = ["mu", "sigma", "gamma", "lambda"]


def ensure_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def psychometric_fit_mode() -> str:
    """Which participant-level psychometric fits feed the downstream analyses.

    'reported': the fits reported in the manuscript (data/reported_psychometric_fits.csv).
    'refit'   : the fits estimated in this run by step 1.
    Set with the environment variable POG_PSYCHOMETRIC_FITS. The default is
    'reported' when the reported fits are present and 'refit' otherwise.
    """
    mode = os.environ.get("POG_PSYCHOMETRIC_FITS", "").strip().lower()
    if not mode:
        mode = "reported" if (DATA_DIR / REPORTED_FITS_FILE).exists() else "refit"
    if mode not in {"reported", "refit"}:
        raise ValueError("POG_PSYCHOMETRIC_FITS must be 'reported' or 'refit'")
    if mode == "reported" and not (DATA_DIR / REPORTED_FITS_FILE).exists():
        raise FileNotFoundError(f"POG_PSYCHOMETRIC_FITS=reported needs {DATA_DIR / REPORTED_FITS_FILE}")
    return mode


def write_run_info() -> dict:
    """Record the software environment and fit mode in results/run_info.json."""
    import importlib.metadata as metadata
    import json
    import platform

    info = {"python": platform.python_version(), "platform": platform.platform(),
            "psychometric_fits": psychometric_fit_mode()}
    for package in ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "factor_analyzer"]:
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = None
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def read_data(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the de-identified data files in {DATA_DIR} "
            "or set POG_DATA_DIR (see data/README.md)."
        )
    return pd.read_csv(path, dtype={"participant_id": str})


# ---------------------------------------------------------------------------
# Psychometric function
# ---------------------------------------------------------------------------
def logistic_4pl(d, mu, sigma, gamma, lam):
    """p(valuable | d) = gamma + (lambda - gamma) / (1 + exp((d - mu) / sigma))."""
    z = np.clip((np.asarray(d, dtype=float) - mu) / sigma, -700.0, 700.0)
    return gamma + (lam - gamma) / (1.0 + np.exp(z))


def _logistic_4pl_reward_normalized(r, mu_r, sigma, gamma, lam):
    """The same function written in the reward-normalized coordinate of the
    original analysis, r = 90 - d (r_bin in test_trials.csv), in which it
    increases with r."""
    with np.errstate(over="ignore"):
        return gamma + (lam - gamma) / (1.0 + np.exp(-(r - mu_r) / sigma))


def fit_4pl(levels, proportions) -> np.ndarray:
    """Unweighted nonlinear least squares fit of the four-parameter logistic.

    As in the original analysis, the optimization runs in the coordinate
    r = 90 - d (r_bin) and mu is converted back (mu_d = 90 - mu_r); the model
    is identical. Initial values: mu = median stimulus level, sigma = 1,
    gamma = min(y), lambda = max(y). Bounds: mu within the stimulus range,
    sigma >= 1e-6, gamma and lambda in [0, 1]. Returns [mu, sigma, gamma,
    lambda] in the reporting coordinate d, or NaNs if the fit fails.

    When a participant's choices switch completely between two adjacent
    stimulus levels, least squares determines mu only to within the 10-unit
    interval between them (sigma -> 0); where the optimizer stops inside that
    interval then depends on the numerical library build (see README).
    """
    r = 90.0 - np.asarray(levels, dtype=float)
    y = np.asarray(proportions, dtype=float)
    order = np.argsort(r, kind="stable")
    r, y = r[order], y[order]
    p0 = [float(np.median(r)), 1.0, float(y.min()), float(y.max())]
    try:
        params, _ = curve_fit(
            _logistic_4pl_reward_normalized,
            r,
            y,
            p0=p0,
            bounds=([r.min(), 1e-6, 0.0, 0.0], [r.max(), np.inf, 1.0, 1.0]),
            maxfev=20_000,
        )
    except Exception:
        return np.full(4, np.nan)
    mu_r, sigma, gamma, lam = (float(v) for v in params)
    return np.array([90.0 - mu_r, sigma, gamma, lam])


def binned_proportions(d, choices) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Proportion of 'valuable' predictions at each stimulus level."""
    levels, inverse = np.unique(np.asarray(d, dtype=float), return_inverse=True)
    counts = np.bincount(inverse)
    successes = np.bincount(inverse, weights=np.asarray(choices, dtype=float))
    return levels, successes / counts, counts


def auc_4pl(mu, sigma, gamma, lam) -> float:
    """Area under the fitted curve over d in [0, 90], divided by 90."""
    if not np.all(np.isfinite([mu, sigma, gamma, lam])):
        return np.nan
    grid = np.linspace(0.0, 90.0, 901)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(logistic_4pl(grid, mu, sigma, gamma, lam), grid) / 90.0)


def max_steepness(sigma, gamma, lam) -> float:
    """Maximum derivative of the fitted function, S = (lambda - gamma) / (4 sigma)."""
    return (lam - gamma) / (4.0 * sigma)


# ---------------------------------------------------------------------------
# Response-time model
# ---------------------------------------------------------------------------
def rt_design(d, mu) -> tuple[np.ndarray, np.ndarray]:
    """Design for log RT = b0 + b_D |z_R| + b_A z_R, with z_R = (mu - d) / 10."""
    z = (float(mu) - np.asarray(d, dtype=float)) / 10.0
    return np.column_stack([np.ones(len(z)), np.abs(z), z]), z


def ols_hc3(y, x) -> dict[str, np.ndarray]:
    """OLS coefficients with HC3 standard errors."""
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    inverse = np.linalg.pinv(x.T @ x)
    residual = y - x @ beta
    leverage = np.sum((x @ inverse) * x, axis=1)
    denominator = np.maximum(1.0 - leverage, 1e-8)
    weighted_x = x * (residual / denominator)[:, None]
    covariance = inverse @ (weighted_x.T @ weighted_x) @ inverse
    return {
        "beta": beta,
        "se": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
        "residual": residual,
        "leverage": leverage,
    }


# ---------------------------------------------------------------------------
# Questionnaire scoring
# ---------------------------------------------------------------------------
# Items in questionnaire_items.csv are already coded in scoring units (see
# data/README.md); each score is the sum of its items. POG items are numbered
# after removal of the embedded attention check (raw question 10).
SCALE_ITEMS: dict[str, list[str]] = {
    "pog_total": [f"pog_{i:02d}" for i in range(1, 17)],
    "pog_upward": [f"pog_{i:02d}" for i in (2, 4, 7, 9, 15)],
    "pog_social": [f"pog_{i:02d}" for i in (5, 10, 12, 13, 14)],
    "pog_lateral": [f"pog_{i:02d}" for i in (1, 3, 6, 8, 11, 16)],
    "7up7down_composite": [f"SU_{i:02d}" for i in range(1, 8)] + [f"SD_{i:02d}" for i in range(1, 8)],
    "7up": [f"SU_{i:02d}" for i in range(1, 8)],
    "7down": [f"SD_{i:02d}" for i in range(1, 8)],
    "pmq9": [f"pmq_{i:02d}" for i in range(1, 10)],
    "phq9": [f"phq_{i:02d}" for i in range(1, 10)],
    "bas_drive": [f"bisbas_{i:02d}" for i in (9, 10, 11, 12)],
    "bas_reward": [f"bisbas_{i:02d}" for i in (5, 6, 7, 8)],
    "bis": [f"bisbas_{i:02d}" for i in (1, 2, 3, 4)],
    "wassup": [f"wassup_{i:02d}" for i in range(1, 31)],
    "ngs": [f"ngs_{i:02d}" for i in range(1, 8)],
    "gse": [f"gse_{i:02d}" for i in range(1, 11)],
    "gad7": [f"gad7_{i:02d}" for i in range(1, 8)],
    "pswq": [f"pswq_{i:02d}" for i in range(1, 4)],
    "shaps": [f"shaps_{i:02d}" for i in range(1, 15)],
    "ie4": [f"ie4_{i:02d}" for i in range(1, 5)],
}

# Display order and labels used in Figure 5 and the correlation tables.
SCORE_LABELS: dict[str, str] = {
    "pog_total": "POG total",
    "7up7down_composite": "7Up/7Down composite",
    "pog_upward": "POG upward",
    "pog_social": "POG social",
    "pog_lateral": "POG lateral",
    "7up": "7-Up",
    "7down": "7-Down",
    "pmq9": "PMQ-9",
    "phq9": "PHQ-9",
    "bas_drive": "BAS Drive",
    "bas_reward": "BAS Reward",
    "bis": "BIS",
    "wassup": "WASSUP",
    "ngs": "NGS",
    "gse": "GSE",
    "gad7": "GAD-7",
    "pswq": "PSWQ",
    "shaps": "SHAPS",
    "ie4": "IE-4",
}
PRIMARY_SCORES = ("pog_total", "7up7down_composite")  # primary tests for mu only


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def pearson_with_ci(x, y, alpha: float = 0.05) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    r, p = stats.pearsonr(x, y)
    half = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n - 3)
    z = np.arctanh(r)
    return {"n": n, "r": float(r), "ci_low": float(np.tanh(z - half)),
            "ci_high": float(np.tanh(z + half)), "p": float(p)}


def bh_fdr(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q), computed at full precision."""
    pvalues = np.asarray(pvalues, dtype=float)
    return multipletests(pvalues, method="fdr_bh")[1]
