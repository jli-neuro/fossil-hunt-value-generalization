#!/usr/bin/env python3
"""Reproduce the main results from the public per-participant summary data.

Needs only summary_data/ (no trial- or item-level data):
  1. refits each participant's four-parameter logistic to the choice
     proportions at each stimulus distance and compares with the reported fits;
  2. recomputes beta_D and beta_A from the mean log RT at each distance,
     weighted by the number of valid trials. Because the model's predictors
     depend only on distance, this gives exactly the trial-level OLS estimates;
  3. recomputes the correlations of mu, beta_D, beta_A and beta_D_plus with the
     19 questionnaire scores, with the paper's FDR families;
  4. recomputes the correlations of mu with the nine k = 9 factor scores;
and checks 3 and 4 against the values reported in the paper, as step 7 does.

Output: results/from_summary_data/self_report_correlations.csv
        results/from_summary_data/factor_k9_correlations.csv
        results/from_summary_data/summary_data_check.md
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pog_common import PSYCHOMETRIC_PARAMS, RESULTS_DIR, ROOT, bh_fdr, fit_4pl, pearson_with_ci, rt_design

SUMMARY_DIR = Path(os.environ.get("POG_SUMMARY_DIR", ROOT / "summary_data"))
OUT_DIR = RESULTS_DIR / "from_summary_data"
HERE = Path(__file__).resolve().parent


def load_step(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(SUMMARY_DIR / name, dtype={"participant_id": str})
    summary = read("participant_summary.csv").set_index("participant_id")
    choice = read("choice_by_distance.csv")
    rt = read("rt_by_distance.csv")
    factors = read("factor_scores_k9.csv").set_index("participant_id")
    step4 = load_step("04_self_report_correlations.py", "step4")
    step7 = load_step("07_check_against_manuscript.py", "step7")
    report = ["# Results reproduced from the per-participant summary data", ""]

    # 1. Psychometric refits from the choice proportions.
    refit = pd.DataFrame.from_dict(
        {pid: fit_4pl(g["distance"].to_numpy(float), g["p_valuable"].to_numpy(float))
         for pid, g in choice.groupby("participant_id")},
        orient="index", columns=PSYCHOMETRIC_PARAMS)
    diff_mu = (refit["mu"] - summary.loc[refit.index, "mu"]).abs()
    report += [f"1. Psychometric refits: mu is within 0.01 of the reported fit for "
               f"{int((diff_mu <= 0.01).sum())} of {len(diff_mu)} participants (largest difference "
               f"{diff_mu.max():.2f}; see the README on participants whose choices switch completely "
               "between neighbouring stimuli).", ""]

    # 2. RT coefficients from the distance-level means.
    worst = 0.0
    for pid, g in rt.groupby("participant_id"):
        x, _ = rt_design(g["distance"], summary.at[pid, "mu"])
        w = g["n_valid_rt"].to_numpy(float)
        beta = np.linalg.solve(x.T @ (x * w[:, None]), x.T @ (w * g["mean_log_rt"].to_numpy(float)))
        worst = max(worst, float(np.abs(beta[1:] - summary.loc[pid, ["beta_d", "beta_a"]].to_numpy(float)).max()))
    report += [f"2. beta_D and beta_A recomputed from the distance-level means match "
               f"participant_summary.csv to within {worst:.1e}.", ""]

    # 3. Correlations with the 19 questionnaire scores.
    corr = step4.correlation_table(summary.reset_index())
    corr.to_csv(OUT_DIR / "self_report_correlations.csv", index=False)

    # 4. Correlations of mu with the k = 9 factor scores, FDR across the nine.
    k9 = pd.DataFrame([{"factor": c, **pearson_with_ci(summary.loc[factors.index, "mu"], factors[c])}
                       for c in factors.columns])
    k9["q"] = bh_fdr(k9["p"])
    k9.to_csv(OUT_DIR / "factor_k9_correlations.csv", index=False)

    # Compare 3 and 4 with the paper at reported precision.
    checks = []
    for measure, scores in step7.CORRELATIONS.items():
        for score, values in scores.items():
            row = corr.loc[(corr["measure"] == measure) & (corr["score"] == score)].iloc[0]
            for name, reported, dec in zip("rpq", values, (2, 3, 3)):
                if reported is None:
                    if name == "p":
                        checks.append((f"{measure} / {score} p < .001", "< .001", row["p"], row["p"] < .001))
                    continue
                checks.append((f"{measure} / {score} {name}", reported, row[name],
                               step7.check(row[name], reported, dec)))
    label_of = {c: c.split("_", 1)[1].replace("Anxiety_Worry", "Anxiety/Worry").replace("BIS_BAS", "BIS/BAS")
                for c in factors.columns}
    for _, row in k9.iterrows():
        label = label_of[row["factor"]]
        if label in step7.K9_TABLE:
            r, p, q, _ = step7.K9_TABLE[label]
            for name, reported, dec in (("r", r, 2), ("p", p, 3), ("q", q, 3)):
                checks.append((f"k = 9 factor {label} {name}", reported, row[name],
                               step7.check(row[name], reported, dec)))
    table = pd.DataFrame(checks, columns=["item", "reported", "from_summary_data", "match"])
    n_fail = int((~table["match"]).sum())
    report += [f"3-4. {len(table)} reported correlation values checked; {len(table) - n_fail} match at "
               f"reported precision; {n_fail} differ.", ""]
    if n_fail:
        report += [table.loc[~table["match"]].to_markdown(index=False), "",
                   "The factor-score values that differ lie on rounding boundaries; see the README.", ""]
    report += ["## All checks", "", table.to_markdown(index=False), ""]
    (OUT_DIR / "summary_data_check.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:8]))


if __name__ == "__main__":
    main()
