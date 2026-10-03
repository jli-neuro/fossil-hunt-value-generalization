#!/usr/bin/env python3
"""Export the de-identified per-participant summary data posted in summary_data/.

Run by those holding the full data, after steps 1, 2, 3 and 5. Only
per-participant summaries are exported; trial-by-trial responses and
item-level questionnaire responses stay on request.

Input : data/test_trials.csv, derived/psychometric_parameters.csv,
        derived/rt_parameters.csv, derived/questionnaire_scores.csv,
        derived/factor_scores_k9.csv, results/factor_k9_table.csv
Output: summary_data/participant_summary.csv
        summary_data/choice_by_distance.csv
        summary_data/rt_by_distance.csv
        summary_data/factor_scores_k9.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pog_common import DERIVED_DIR, RESULTS_DIR, ROOT, RT_MAX, RT_MIN, SCORE_LABELS, read_data

SUMMARY_DIR = Path(os.environ.get("POG_SUMMARY_DIR", ROOT / "summary_data"))
D_COL = "distance_from_rewarded_endpoint"
# Column-name forms of the factor labels used in the paper.
FACTOR_NAMES = {"GAD-7": "Anxiety_Worry", "PSWQ": "Anxiety_Worry", "BIS/BAS": "BIS_BAS"}


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(DERIVED_DIR / name, dtype={"participant_id": str})

    psych = read("psychometric_parameters.csv")
    if not (psych["fit_source"] == "reported").all():
        raise RuntimeError("Export from the reported psychometric fits (POG_PSYCHOMETRIC_FITS=reported).")
    rt = read("rt_parameters.csv")
    scores = read("questionnaire_scores.csv")
    summary = (
        psych[["participant_id", "mu", "sigma", "gamma", "lambda", "auc", "log_steepness"]]
        .merge(rt[["participant_id", "n_valid_rt", "beta_0", "beta_d", "beta_a", "beta_d_plus",
                   "beta_d_minus", "mean_log_rt", "mean_rt", "median_rt",
                   "mean_log_rt_rewarded_exemplar"]], on="participant_id", validate="one_to_one")
        .merge(scores[["participant_id", *SCORE_LABELS]], on="participant_id", validate="one_to_one")
        .sort_values("participant_id")
    )
    summary.to_csv(SUMMARY_DIR / "participant_summary.csv", index=False)

    trials = read_data("test_trials.csv")
    choice = (trials.groupby(["participant_id", D_COL])["pred_reward"]
              .agg(n_trials="size", n_valuable="sum").reset_index()
              .rename(columns={D_COL: "distance"}))
    choice["p_valuable"] = choice["n_valuable"] / choice["n_trials"]
    choice.to_csv(SUMMARY_DIR / "choice_by_distance.csv", index=False)

    rt_trials = pd.to_numeric(trials["rt"], errors="coerce")
    valid = trials.loc[rt_trials.between(RT_MIN, RT_MAX)].assign(log_rt=lambda x: np.log(x["rt"]))
    rt_by = (valid.groupby(["participant_id", D_COL])
             .agg(n_valid_rt=("rt", "size"), mean_log_rt=("log_rt", "mean"), median_rt=("rt", "median"))
             .reset_index().rename(columns={D_COL: "distance"}))
    rt_by.to_csv(SUMMARY_DIR / "rt_by_distance.csv", index=False)

    factors = read("factor_scores_k9.csv")
    table = pd.read_csv(RESULTS_DIR / "factor_k9_table.csv")
    names = {row.factor: f"{row.factor}_{FACTOR_NAMES.get(row.label, row.label)}" for row in table.itertuples()}
    factors.rename(columns=names).sort_values("participant_id").to_csv(
        SUMMARY_DIR / "factor_scores_k9.csv", index=False)
    print(f"Exported summaries for {len(summary)} participants to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
