#!/usr/bin/env python3
"""Step 1: fit the four-parameter logistic psychometric function.

Every participant is refitted. The downstream analyses then use either the
fits reported in the manuscript or this run's fits (POG_PSYCHOMETRIC_FITS,
see pog_common.psychometric_fit_mode and the README). When the reported fits
are available, their agreement with this run's fits is summarized.
Skewness is the adjusted Fisher-Pearson (sample) coefficient.

Input : data/test_trials.csv (one row per test trial)
        data/reported_psychometric_fits.csv (optional)
Output: derived/psychometric_refit.csv       (participant level, this run)
        derived/psychometric_parameters.csv  (participant level, used downstream)
        results/psychometric_summary.csv     (descriptive, aggregate only)
        results/psychometric_refit_agreement.csv (aggregate; if reported fits present)
        results/run_info.json                (software versions, fit mode)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from pog_common import (
    DATA_DIR, DERIVED_DIR, PSYCHOMETRIC_PARAMS, REPORTED_FITS_FILE, RESULTS_DIR,
    auc_4pl, binned_proportions, ensure_dirs, fit_4pl, max_steepness,
    psychometric_fit_mode, read_data, write_run_info,
)

D_COL = "distance_from_rewarded_endpoint"


def agreement(refit: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    """Aggregate agreement between this run's fits and the reported fits."""
    both = refit.merge(reported, on="participant_id", suffixes=("_refit", "_reported"),
                       validate="one_to_one")
    rows = []
    for name in PSYCHOMETRIC_PARAMS:
        diff = (both[f"{name}_refit"] - both[f"{name}_reported"]).abs()
        rows.append({"parameter": name, "n": len(both),
                     "n_abs_diff_gt_0.001": int((diff > 1e-3).sum()),
                     "n_abs_diff_gt_0.01": int((diff > 1e-2).sum()),
                     "n_abs_diff_gt_0.1": int((diff > 0.1).sum()),
                     "max_abs_diff": float(diff.max())})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    trials = read_data("test_trials.csv")
    mode = psychometric_fit_mode()

    rows = []
    for pid, g in trials.groupby("participant_id", sort=True):
        levels, props, _ = binned_proportions(g[D_COL], g["pred_reward"])
        rows.append({"participant_id": pid, "n_test_trials": len(g),
                     **dict(zip(PSYCHOMETRIC_PARAMS, fit_4pl(levels, props)))})
    refit = pd.DataFrame(rows)
    refit.to_csv(DERIVED_DIR / "psychometric_refit.csv", index=False)

    reported_path = DATA_DIR / REPORTED_FITS_FILE
    if reported_path.exists():
        reported = pd.read_csv(reported_path, dtype={"participant_id": str})
        agreement(refit, reported[["participant_id", *PSYCHOMETRIC_PARAMS]]).to_csv(
            RESULTS_DIR / "psychometric_refit_agreement.csv", index=False)
    if mode == "reported":
        params = refit[["participant_id", "n_test_trials"]].merge(
            reported[["participant_id", *PSYCHOMETRIC_PARAMS]], on="participant_id", validate="one_to_one")
    else:
        params = refit.copy()
    params["fit_source"] = mode
    params["auc"] = [auc_4pl(*v) for v in params[PSYCHOMETRIC_PARAMS].to_numpy(float)]
    params["steepness"] = max_steepness(params["sigma"], params["gamma"], params["lambda"])
    params["log_steepness"] = np.log(params["steepness"])
    params.to_csv(DERIVED_DIR / "psychometric_parameters.csv", index=False)

    # Group-level curve: fit to the mean proportion at each stimulus level.
    per_level = (
        trials.groupby(["participant_id", D_COL])["pred_reward"].mean()
        .groupby(level=D_COL).mean()
    )
    group = fit_4pl(per_level.index.to_numpy(float), per_level.to_numpy(float))

    summary = []
    for name in ["mu", "sigma", "gamma", "lambda", "auc", "log_steepness"]:
        x = params[name].to_numpy(float)
        summary.append({
            "quantity": name, "n": int(np.isfinite(x).sum()),
            "mean": np.nanmean(x), "sd": np.nanstd(x, ddof=1),
            "median": np.nanmedian(x), "min": np.nanmin(x), "max": np.nanmax(x),
            "skewness": stats.skew(x[np.isfinite(x)], bias=False),
        })
    summary.append({"quantity": "steepness_raw", "n": len(params),
                    "skewness": stats.skew(params["steepness"], bias=False)})
    summary.append({"quantity": "n_sigma_below_1", "n": int((params["sigma"] < 1).sum())})
    summary.append({"quantity": "test_trials_per_participant",
                    "min": params["n_test_trials"].min(), "max": params["n_test_trials"].max()})
    for label, value in zip(PSYCHOMETRIC_PARAMS, group):
        summary.append({"quantity": f"group_curve_{label}", "mean": value})
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "psychometric_summary.csv", index=False)
    write_run_info()
    print(f"Fitted {len(refit)} participants (failures: {int(refit['mu'].isna().sum())}); "
          f"downstream analyses use the {mode} fits")


if __name__ == "__main__":
    main()
