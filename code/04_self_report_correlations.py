#!/usr/bin/env python3
"""Step 4: correlations between task measures and the 19 questionnaire scores.

Correction families (Methods, "Statistical analyses"):
  * mu: POG total and the 7Up/7Down composite are the two primary tests and are
    reported unadjusted; BH-FDR is applied across the remaining 17 scores.
  * beta_D, beta_A, beta_D_plus (secondary): BH-FDR across all 19 scores,
    separately for each measure.
All q-values are computed from full-precision p-values.

Input : derived/psychometric_parameters.csv, derived/rt_parameters.csv,
        derived/questionnaire_scores.csv
Output: results/self_report_correlations.csv
        results/parameter_intercorrelations.csv
        results/rt_pog_checks.csv
        results/questionnaire_descriptives.csv
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from pog_common import (
    DERIVED_DIR, PRIMARY_SCORES, RESULTS_DIR, SCORE_LABELS, bh_fdr,
    ensure_dirs, pearson_with_ci,
)

MEASURES = {
    "mu": "Generalization breadth (mu)",
    "beta_d": "Log-RT distance slope (beta_D)",
    "beta_a": "Log-RT asymmetry (beta_A)",
    "beta_d_plus": "Rewarded-direction slope (beta_D_plus)",
}


def load() -> pd.DataFrame:
    read = lambda name: pd.read_csv(DERIVED_DIR / name, dtype={"participant_id": str})
    data = (read("psychometric_parameters.csv")
            .merge(read("rt_parameters.csv"), on="participant_id", suffixes=("", "_rt"))
            .merge(read("questionnaire_scores.csv"), on="participant_id"))
    return data.loc[data["rt_included"]].reset_index(drop=True)


def correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for measure, measure_label in MEASURES.items():
        block = []
        for score, score_label in SCORE_LABELS.items():
            res = pearson_with_ci(data[measure], data[score])
            primary = measure == "mu" and score in PRIMARY_SCORES
            block.append({"measure": measure, "measure_label": measure_label,
                          "score": score, "score_label": score_label,
                          "primary_test": primary, **res})
        block = pd.DataFrame(block)
        corrected = ~block["primary_test"]
        block["q"] = np.nan
        block.loc[corrected, "q"] = bh_fdr(block.loc[corrected, "p"])
        block["correction_family"] = np.where(
            block["primary_test"], "primary test, unadjusted",
            f"BH-FDR across {int(corrected.sum())} scores")
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ensure_dirs()
    data = load()
    table = correlation_table(data)
    table.to_csv(RESULTS_DIR / "self_report_correlations.csv", index=False)

    params = ["mu", "sigma", "gamma", "lambda", "auc", "beta_d", "beta_a"]
    inter = [{"x": a, "y": b, **pearson_with_ci(data[a], data[b])}
             for a, b in itertools.combinations(params, 2)]
    pd.DataFrame(inter).to_csv(RESULTS_DIR / "parameter_intercorrelations.csv", index=False)

    checks = ["mean_log_rt", "mean_rt", "median_rt", "beta_0", "mean_log_rt_rewarded_exemplar"]
    pd.DataFrame([{"x": "pog_total", "y": c, **pearson_with_ci(data["pog_total"], data[c])}
                  for c in checks]).to_csv(RESULTS_DIR / "rt_pog_checks.csv", index=False)

    desc = data[list(SCORE_LABELS)].agg(["mean", "std", "median", "min", "max"]).T
    desc.index.name = "score"
    desc.to_csv(RESULTS_DIR / "questionnaire_descriptives.csv")
    print(f"Correlations computed for N = {len(data)}")


if __name__ == "__main__":
    main()
