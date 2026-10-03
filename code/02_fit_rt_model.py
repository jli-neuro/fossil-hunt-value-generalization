#!/usr/bin/env python3
"""Step 2: participant-level log-RT model.

log RT = beta_0 + beta_D |z_R| + beta_A z_R + error,  z_R = (mu - d) / 10,
fitted by OLS to test trials with RT in [0.2, 8.0] s. z_R is positive toward
the rewarded endpoint, so beta_D + beta_A is the rewarded-direction slope
(beta_D_plus) and beta_D - beta_A the nonrewarded-direction slope.

Input : data/test_trials.csv, derived/psychometric_parameters.csv
Output: derived/rt_parameters.csv, results/rt_summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pog_common import (
    DERIVED_DIR, RESULTS_DIR, RT_MAX, RT_MIN, RT_MIN_ADEQUATE_BINS,
    RT_MIN_TRIALS_PER_BIN, RT_MIN_VALID_TRIALS, ensure_dirs, ols_hc3,
    read_data, rt_design,
)


def main() -> None:
    ensure_dirs()
    trials = read_data("test_trials.csv")
    psych = pd.read_csv(DERIVED_DIR / "psychometric_parameters.csv", dtype={"participant_id": str})
    d_col = "distance_from_rewarded_endpoint"

    rt = pd.to_numeric(trials["rt"], errors="coerce")
    trials["in_window"] = rt.between(RT_MIN, RT_MAX)
    if "rt_valid" in trials:
        mismatch = int((trials["in_window"] != trials["rt_valid"].astype(bool)).sum())
        if mismatch:
            raise RuntimeError(f"{mismatch} trials disagree with the rt_valid flag")

    mu_by_pid = psych.set_index("participant_id")["mu"]
    rows = []
    for pid, g in trials.groupby("participant_id", sort=True):
        valid = g.loc[g["in_window"]]
        bin_counts = valid[d_col].value_counts()
        n_adequate = int((bin_counts >= RT_MIN_TRIALS_PER_BIN).sum())
        included = len(valid) >= RT_MIN_VALID_TRIALS and n_adequate >= RT_MIN_ADEQUATE_BINS
        log_rt = np.log(valid["rt"].to_numpy(float))
        x, _ = rt_design(valid[d_col], mu_by_pid[pid])
        fit = ols_hc3(log_rt, x)
        b0, bd, ba = fit["beta"]
        at_rewarded = valid.loc[valid[d_col] == 0, "rt"]
        rows.append({
            "participant_id": pid,
            "n_test_trials": len(g),
            "n_valid_rt": len(valid),
            "n_adequate_bins": n_adequate,
            "rt_included": included,
            "beta_0": b0, "beta_d": bd, "beta_a": ba,
            "beta_d_plus": bd + ba, "beta_d_minus": bd - ba,
            "beta_d_se_hc3": fit["se"][1], "beta_a_se_hc3": fit["se"][2],
            "mean_log_rt": float(log_rt.mean()),
            "mean_rt": float(valid["rt"].mean()),
            "median_rt": float(valid["rt"].median()),
            "mean_log_rt_rewarded_exemplar": float(np.log(at_rewarded).mean()),
        })
    params = pd.DataFrame(rows)
    params.to_csv(DERIVED_DIR / "rt_parameters.csv", index=False)

    inc = params.loc[params["rt_included"]]
    summary = [
        {"quantity": "n_participants_included", "value": len(inc)},
        {"quantity": "n_trials_total", "value": len(trials)},
        {"quantity": "prop_trials_outside_window", "value": 1 - trials["in_window"].mean()},
        {"quantity": "skewness_raw_rt_retained",
         "value": stats.skew(trials.loc[trials["in_window"], "rt"], bias=False)},
        {"quantity": "skewness_log_rt_retained",
         "value": stats.skew(np.log(trials.loc[trials["in_window"], "rt"]), bias=False)},
    ]
    for name in ["beta_d", "beta_a", "beta_d_plus", "beta_d_minus", "beta_0"]:
        x = inc[name].to_numpy(float)
        t = stats.ttest_1samp(x, 0.0)
        summary += [
            {"quantity": f"{name}_mean", "value": x.mean()},
            {"quantity": f"{name}_sd", "value": x.std(ddof=1)},
            {"quantity": f"{name}_t_vs_0", "value": t.statistic, "df": len(x) - 1, "p": t.pvalue},
        ]
    medians = trials.loc[trials["in_window"]].groupby(d_col)["rt"].median()
    for level, value in medians.items():
        summary.append({"quantity": f"median_rt_at_d{int(level)}", "value": value})
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "rt_summary.csv", index=False)
    print(f"RT model: {len(inc)} of {len(params)} participants meet the inclusion rule")


if __name__ == "__main__":
    main()
