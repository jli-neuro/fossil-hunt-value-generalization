#!/usr/bin/env python3
"""Step 6: participant-matched parameter recovery (psychometric and RT models).

For each participant, 200 synthetic datasets are generated on that participant's
own trial sequence:
  1. choices are drawn from the participant's fitted psychometric function and
     the function is refitted (psychometric recovery);
  2. log RTs are generated from the participant's fitted RT model using
     leverage-adjusted (HC2) residuals with Rademacher signs;
  3. two-stage scheme: the RT model is refitted with distance recomputed from
     the *recovered* boundary; fixed-boundary scheme: distance uses the
     generating boundary, isolating the RT regression.
A two-stage design is treated as non-identifiable when its standardized
condition number exceeds 10.

Input : data/test_trials.csv, derived/psychometric_parameters.csv,
        derived/rt_parameters.csv, data/reported_psychometric_fits.csv
        (optional; supplies the original processing order)
Output: results/recovery_summary.csv, derived/recovery_replicates.csv
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from pog_common import (
    DATA_DIR, DERIVED_DIR, N_BOOT_SEED, REPORTED_FITS_FILE, RESULTS_DIR, RT_MAX,
    RT_MIN, auc_4pl, binned_proportions, ensure_dirs, fit_4pl, logistic_4pl,
    ols_hc3, read_data, rt_design,
)

N_REPLICATIONS = 200
MAX_CONDITION_NUMBER = 10.0


def standardized_condition_number(z: np.ndarray) -> float:
    a = np.abs(z)
    if a.std(ddof=0) <= 1e-12 or z.std(ddof=0) <= 1e-12:
        return np.inf
    x = np.column_stack([np.ones(len(z)), (a - a.mean()) / a.std(ddof=0), (z - z.mean()) / z.std(ddof=0)])
    return float(np.linalg.cond(x))


def run_participant(payload: dict) -> list[dict]:
    rng = np.random.default_rng(payload["seed"])
    theta = np.asarray(payload["theta"], dtype=float)          # mu, sigma, gamma, lambda
    choice_d, rt_d = payload["choice_d"], payload["rt_d"]
    p_true = np.clip(logistic_4pl(choice_d, *theta), 0.0, 1.0)
    x_true, _ = rt_design(rt_d, theta[0])
    beta, resid = payload["beta"], payload["residual_hc2"]
    rows = []
    for rep in range(1, N_REPLICATIONS + 1):
        choices = rng.binomial(1, p_true)
        levels, props, _ = binned_proportions(choice_d, choices)
        theta_hat = fit_4pl(levels, props)
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(rt_d), replace=True)
        log_rt = x_true @ beta + resid * signs
        fixed = ols_hc3(log_rt, x_true)
        full = {"beta": np.full(3, np.nan), "se": np.full(3, np.nan)}
        cond = np.nan
        if np.all(np.isfinite(theta_hat)):
            x_hat, z_hat = rt_design(rt_d, theta_hat[0])
            cond = standardized_condition_number(z_hat)
            if np.isfinite(cond) and cond <= MAX_CONDITION_NUMBER and np.linalg.matrix_rank(x_hat) == 3:
                full = ols_hc3(log_rt, x_hat)
        row = {"participant_id": payload["pid"], "replicate": rep,
               "mu_true": theta[0], "sigma_true": theta[1], "gamma_true": theta[2],
               "lambda_true": theta[3], "auc_true": payload["auc"],
               "mu_rec": theta_hat[0], "sigma_rec": theta_hat[1], "gamma_rec": theta_hat[2],
               "lambda_rec": theta_hat[3], "auc_rec": auc_4pl(*theta_hat),
               "condition_number": cond,
               "beta_d_true": beta[1], "beta_a_true": beta[2]}
        for scheme, fit in (("two_stage", full), ("fixed_mu", fixed)):
            for j, name in ((1, "beta_d"), (2, "beta_a")):
                est, se = fit["beta"][j], fit["se"][j]
                row[f"{name}_rec_{scheme}"] = est
                row[f"{name}_covered_{scheme}"] = bool(
                    np.isfinite(est) and np.isfinite(se) and est - 1.96 * se <= beta[j] <= est + 1.96 * se)
        rows.append(row)
    return rows


def build_payloads() -> list[dict]:
    """One simulation specification per participant in the RT sample."""
    trials = read_data("test_trials.csv")
    read = lambda name: pd.read_csv(DERIVED_DIR / name, dtype={"participant_id": str})
    params = read("psychometric_parameters.csv").merge(read("rt_parameters.csv"), on="participant_id")
    params = params.loc[params["rt_included"]]
    # Each participant gets its own random-number stream, assigned in the
    # order participants were processed in the original analysis when that
    # order is available (analysis_order in the reported fits), else by ID.
    order_path = DATA_DIR / REPORTED_FITS_FILE
    if order_path.exists():
        order = pd.read_csv(order_path, dtype={"participant_id": str})[["participant_id", "analysis_order"]]
        params = params.merge(order, on="participant_id", validate="one_to_one").sort_values("analysis_order")
    else:
        params = params.sort_values("participant_id")
    params = params.reset_index(drop=True)
    seeds = np.random.SeedSequence(N_BOOT_SEED).spawn(len(params))
    d_col = "distance_from_rewarded_endpoint"

    payloads = []
    for i, row in params.iterrows():
        g = trials.loc[trials["participant_id"] == row["participant_id"]]
        choice_d = g[d_col].to_numpy(float)
        rt = pd.to_numeric(g["rt"], errors="coerce").to_numpy(float)
        keep = np.isfinite(rt) & (rt >= RT_MIN) & (rt <= RT_MAX)
        rt_d = choice_d[keep]
        x, _ = rt_design(rt_d, row["mu"])
        obs = ols_hc3(np.log(rt[keep]), x)
        if not np.allclose(obs["beta"][1:], [row["beta_d"], row["beta_a"]], atol=1e-9):
            raise RuntimeError(f"RT refit mismatch for {row['participant_id']}")
        resid = obs["residual"] / np.sqrt(np.maximum(1.0 - obs["leverage"], 1e-8))
        payloads.append({"pid": row["participant_id"],
                         "seed": int(seeds[i].generate_state(1, dtype=np.uint64)[0]),
                         "theta": [row["mu"], row["sigma"], row["gamma"], row["lambda"]],
                         "auc": row["auc"], "choice_d": choice_d, "rt_d": rt_d,
                         "beta": obs["beta"], "residual_hc2": resid - resid.mean()})
    return payloads


def simulate(payloads: list[dict], workers: int | None = None) -> pd.DataFrame:
    """Run the replications; results do not depend on the number of workers."""
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame([r for rows in pool.map(run_participant, payloads) for r in rows])


def summarize(reps: pd.DataFrame) -> pd.DataFrame:
    summary = []
    means = reps.groupby("participant_id").mean(numeric_only=True)
    for name in ["mu", "sigma", "gamma", "lambda", "auc"]:
        ok = reps[f"{name}_rec"].notna()
        m = means[[f"{name}_true", f"{name}_rec"]].dropna()
        summary.append({
            "parameter": name, "scheme": "psychometric",
            "participant_mean_r": stats.pearsonr(m[f"{name}_true"], m[f"{name}_rec"])[0],
            "mean_bias": float((m[f"{name}_rec"] - m[f"{name}_true"]).mean()),
            "failure_rate": float(1 - ok.mean())})
    for scheme in ("two_stage", "fixed_mu"):
        for name in ("beta_d", "beta_a"):
            col = f"{name}_rec_{scheme}"
            per_rep = []
            for _, g in reps.groupby("replicate"):
                g = g.loc[g[col].notna()]
                per_rep.append(stats.pearsonr(g[f"{name}_true"], g[col])[0])
            per_rep = np.asarray(per_rep)
            m = means[[f"{name}_true", col]].dropna()
            summary.append({
                "parameter": name, "scheme": scheme,
                "participant_mean_r": stats.pearsonr(m[f"{name}_true"], m[col])[0],
                "replicate_r_median": float(np.median(per_rep)),
                "replicate_r_q025": float(np.quantile(per_rep, 0.025)),
                "replicate_r_q975": float(np.quantile(per_rep, 0.975)),
                "interval_coverage": float(reps.loc[reps[col].notna(), f"{name}_covered_{scheme}"].mean()),
                "mean_bias": float((reps[col] - reps[f"{name}_true"]).mean()),
                "failure_rate": float(reps[col].isna().mean())})
    return pd.DataFrame(summary)


def main() -> None:
    ensure_dirs()
    payloads = build_payloads()
    reps = simulate(payloads)
    reps.to_csv(DERIVED_DIR / "recovery_replicates.csv", index=False)
    summarize(reps).to_csv(RESULTS_DIR / "recovery_summary.csv", index=False)
    print(f"Recovery: {len(payloads)} participants x {N_REPLICATIONS} datasets")


if __name__ == "__main__":
    main()
