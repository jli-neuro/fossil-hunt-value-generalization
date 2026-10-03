#!/usr/bin/env python3
"""Step 7: compare regenerated results with the values reported in the manuscript.

Each reported value is checked at the precision it is reported with: a value
passes when the regenerated number rounds to it (tolerance of half a unit in
the last reported decimal). Writes results/manuscript_check.md.
"""
from __future__ import annotations

import json

import pandas as pd

from pog_common import RESULTS_DIR

# (measure, score) -> (r, p, q) as reported; q = None for the unadjusted primary tests.
CORRELATIONS = {
    "mu": {
        "pog_total": (.21, .008, None), "7up7down_composite": (.15, .050, None),
        "pog_upward": (.23, .003, .058), "pog_social": (.19, .015, .131), "pog_lateral": (.12, .121, .342),
        "7up": (.06, .426, .757), "7down": (.16, .041, .232), "pmq9": (.02, .757, .757),
        "phq9": (.03, .723, .757), "bas_drive": (.12, .115, .342), "bas_reward": (.04, .652, .757),
        "bis": (.11, .160, .389), "wassup": (.12, .115, .342), "ngs": (.05, .515, .757),
        "gse": (.07, .368, .757), "gad7": (.03, .742, .757), "pswq": (.03, .701, .757),
        "shaps": (.03, .679, .757), "ie4": (-.04, .602, .757)},
    "beta_d": {
        "pog_total": (.22, .005, .084), "7up7down_composite": (.13, .102, .242),
        "pog_upward": (.17, .031, .097), "pog_social": (.20, .009, .084), "pog_lateral": (.19, .017, .084),
        "7up": (.18, .022, .084), "7down": (.03, .682, .807), "pmq9": (.05, .512, .807),
        "phq9": (-.02, .830, .876), "bas_drive": (.18, .018, .084), "bas_reward": (.09, .251, .477),
        "bis": (.03, .722, .807), "wassup": (.15, .060, .162), "ngs": (.04, .605, .807),
        "gse": (.04, .591, .807), "gad7": (.00, .988, .988), "pswq": (.03, .713, .807),
        "shaps": (.05, .556, .807), "ie4": (.09, .238, .477)},
    "beta_a": {
        "pog_total": (.19, .014, .162), "7up7down_composite": (-.01, .909, .959),
        "pog_upward": (.14, .074, .351), "pog_social": (.17, .026, .162), "pog_lateral": (.18, .023, .162),
        "7up": (.01, .919, .959), "7down": (-.02, .812, .959), "pmq9": (.05, .521, .959),
        "phq9": (.01, .920, .959), "bas_drive": (.02, .756, .959), "bas_reward": (-.01, .878, .959),
        "bis": (-.01, .921, .959), "wassup": (.05, .531, .959), "ngs": (.10, .206, .781),
        "gse": (.03, .684, .959), "gad7": (.00, .959, .959), "pswq": (-.02, .837, .959),
        "shaps": (.01, .942, .959), "ie4": (-.02, .791, .959)},
    "beta_d_plus": {
        "pog_total": (.30, None, .002), "7up7down_composite": (.12, .136, .323),
        "pog_upward": (.23, .003, .014), "pog_social": (.28, None, .003), "pog_lateral": (.27, None, .004),
        "7up": (.17, .027, .084), "7down": (.02, .789, .902), "pmq9": (.07, .343, .544),
        "phq9": (-.01, .879, .928), "bas_drive": (.19, .016, .062), "bas_reward": (.08, .313, .544),
        "bis": (.02, .774, .902), "wassup": (.17, .035, .095), "ngs": (.09, .257, .542),
        "gse": (.06, .474, .693), "gad7": (.00, .968, .968), "pswq": (.02, .807, .902),
        "shaps": (.05, .552, .750), "ie4": (.08, .326, .544)},
}

# (file, row key column, row key, value column, reported value, decimals, description)
SCALARS = [
    ("psychometric_summary.csv", "quantity", "group_curve_mu", "mean", 48.00, 2, "Group curve mu"),
    ("psychometric_summary.csv", "quantity", "group_curve_sigma", "mean", 8.20, 2, "Group curve sigma"),
    ("psychometric_summary.csv", "quantity", "group_curve_gamma", "mean", .038, 3, "Group curve gamma"),
    ("psychometric_summary.csv", "quantity", "group_curve_lambda", "mean", .981, 3, "Group curve lambda"),
    ("psychometric_summary.csv", "quantity", "mu", "mean", 48.16, 2, "mu mean"),
    ("psychometric_summary.csv", "quantity", "mu", "sd", 9.48, 2, "mu SD"),
    ("psychometric_summary.csv", "quantity", "mu", "min", 20.29, 2, "mu minimum"),
    ("psychometric_summary.csv", "quantity", "mu", "max", 79.45, 2, "mu maximum"),
    ("psychometric_summary.csv", "quantity", "sigma", "mean", 5.07, 2, "sigma mean"),
    ("psychometric_summary.csv", "quantity", "sigma", "sd", 4.15, 2, "sigma SD"),
    ("psychometric_summary.csv", "quantity", "gamma", "mean", .047, 3, "gamma mean"),
    ("psychometric_summary.csv", "quantity", "gamma", "sd", .089, 3, "gamma SD"),
    ("psychometric_summary.csv", "quantity", "lambda", "mean", .971, 3, "lambda mean"),
    ("psychometric_summary.csv", "quantity", "lambda", "sd", .053, 3, "lambda SD"),
    ("psychometric_summary.csv", "quantity", "steepness_raw", "skewness", 3.33, 2, "Raw steepness skewness"),
    ("psychometric_summary.csv", "quantity", "log_steepness", "skewness", 1.26, 2, "Log steepness skewness"),
    ("psychometric_summary.csv", "quantity", "n_sigma_below_1", "n", 24, 0, "Participants with sigma < 1"),
    ("rt_summary.csv", "quantity", "beta_d_mean", "value", -.089, 3, "beta_D mean"),
    ("rt_summary.csv", "quantity", "beta_d_sd", "value", .053, 3, "beta_D SD"),
    ("rt_summary.csv", "quantity", "beta_d_t_vs_0", "value", -21.37, 2, "beta_D t(162)"),
    ("rt_summary.csv", "quantity", "beta_a_mean", "value", -.006, 3, "beta_A mean"),
    ("rt_summary.csv", "quantity", "beta_a_sd", "value", .028, 3, "beta_A SD"),
    ("rt_summary.csv", "quantity", "beta_a_t_vs_0", "value", -2.73, 2, "beta_A t(162)"),
    ("rt_summary.csv", "quantity", "beta_a_t_vs_0", "p", .007, 3, "beta_A p"),
    ("rt_summary.csv", "quantity", "beta_d_plus_mean", "value", -.095, 3, "Rewarded-direction slope mean"),
    ("rt_summary.csv", "quantity", "beta_d_minus_mean", "value", -.082, 3, "Nonrewarded-direction slope mean"),
    ("rt_summary.csv", "quantity", "prop_trials_outside_window", "value", .007, 3, "Trials outside 0.2-8 s"),
    ("rt_summary.csv", "quantity", "skewness_raw_rt_retained", "value", 3.57, 2, "Raw RT skewness"),
    ("rt_summary.csv", "quantity", "skewness_log_rt_retained", "value", 1.04, 2, "Log RT skewness"),
    ("rt_summary.csv", "quantity", "median_rt_at_d40", "value", .963, 3, "Median RT at d = 40"),
    ("rt_summary.csv", "quantity", "median_rt_at_d50", "value", 1.017, 3, "Median RT at d = 50"),
    ("rt_summary.csv", "quantity", "median_rt_at_d0", "value", .667, 3, "Median RT at d = 0"),
    ("rt_summary.csv", "quantity", "median_rt_at_d90", "value", .730, 3, "Median RT at d = 90"),
    ("questionnaire_descriptives.csv", "score", "pog_total", "median", 42, 0, "POG total median (median split)"),
    ("parallel_analysis.csv", "factor", 9, "retained", True, None, "Parallel analysis retains factor 9"),
    ("parallel_analysis.csv", "factor", 10, "retained", False, None, "Parallel analysis stops at 9"),
]
INTERCORRELATIONS = [("mu", "auc", .881), ("mu", "sigma", -.036), ("gamma", "lambda", -.291),
                     ("gamma", "auc", .265), ("mu", "beta_d", .027), ("mu", "beta_a", .346),
                     ("beta_d", "beta_a", -.162)]
RT_POG = [("mean_log_rt", -.06, .454), ("mean_rt", -.02, .791), ("median_rt", -.07, .384),
          ("beta_0", -.14, .076), ("mean_log_rt_rewarded_exemplar", .08, .324)]
K9_TABLE = {  # label: (r, p, q, purity)
    "POG": (.22, .006, .051, .82), "Depression": (.17, .026, .119, .49),
    "WASSUP": (.15, .052, .156, .91), "Hypomania": (.09, .277, .605, .65),
    "BIS/BAS": (.08, .336, .605, .62), "GSE": (.06, .432, .648, .64),
    "Anxiety/Worry": (-.04, .583, .749, .38), "SHAPS": (.02, .819, .841, .76),
    "NGS": (.02, .841, .841, .64)}
# Factor-analysis figure, panel A: |r(mu, factor score)| for the factor richest
# in each item set, k = 2..12, at the precision stored in the published figure.
# Factor signs are arbitrary in EFA, so absolute values are compared
# (tolerance .001).
FACTOR_SWEEP = {
    "pog": [0.153123811, 0.152782544, 0.127125778, 0.120855870, 0.127076446, 0.049449404,
            0.140501616, 0.215891179, 0.210888803, 0.222184341, 0.221335205],
    "depression": [0.074216364, 0.079405812, 0.079077360, 0.102494432, 0.100042678, 0.097794946,
                   0.092170002, 0.173835653, 0.178272943, 0.198093195, 0.197673179],
    "mania_approach": [0.153123811, 0.152782544, 0.151010721, 0.144750306, 0.151834860, 0.152171376,
                       0.151951168, 0.152391266, 0.157520866, 0.151788231, 0.159683259],
}
SIMULATION = {0.0: 15.4, 1.0: 46.8}
RECOVERY = [("mu", "psychometric", "participant_mean_r", .988, 3), ("sigma", "psychometric", "participant_mean_r", .920, 3),
            ("gamma", "psychometric", "participant_mean_r", .971, 3), ("lambda", "psychometric", "participant_mean_r", .918, 3),
            ("auc", "psychometric", "participant_mean_r", .998, 3), ("sigma", "psychometric", "mean_bias", -1.10, 2),
            ("beta_d", "two_stage", "replicate_r_median", .82, 2), ("beta_d", "two_stage", "replicate_r_q025", .77, 2),
            ("beta_d", "two_stage", "replicate_r_q975", .86, 2), ("beta_a", "two_stage", "replicate_r_median", .80, 2),
            ("beta_a", "two_stage", "replicate_r_q025", .71, 2), ("beta_a", "two_stage", "replicate_r_q975", .84, 2),
            ("beta_d", "fixed_mu", "replicate_r_median", .82, 2), ("beta_a", "fixed_mu", "replicate_r_median", .84, 2),
            ("beta_a", "two_stage", "interval_coverage", .90, 2), ("beta_a", "fixed_mu", "interval_coverage", .95, 2)]


def check(computed, reported, decimals) -> bool:
    if decimals is None:
        return bool(computed) == bool(reported)
    return abs(float(computed) - float(reported)) <= 0.5 * 10 ** (-decimals) + 1e-9


def decimals_of(value: float) -> int:
    text = f"{value:.3f}".rstrip("0")
    return max(len(text.split(".")[1]) if "." in text else 0, 2)


def main() -> None:
    rows, not_run = [], []
    add = lambda section, item, rep, comp, ok: rows.append(
        {"section": section, "item": item, "reported": rep, "regenerated": comp, "match": ok})

    def available(*names: str) -> bool:
        missing = [n for n in names if not (RESULTS_DIR / n).exists()]
        not_run.extend(missing)
        return not missing

    if available("self_report_correlations.csv"):
        check_correlations(add)
    for file, key_col, key, col, rep, dec, label in SCALARS:
        if not (RESULTS_DIR / file).exists():
            if file not in not_run:
                not_run.append(file)
            continue
        table = pd.read_csv(RESULTS_DIR / file)
        comp = table.loc[table[key_col] == key, col].iloc[0]
        add(file.replace(".csv", ""), label, rep, comp, check(comp, rep, dec))
    if available("parameter_intercorrelations.csv", "rt_pog_checks.csv"):
        check_intercorrelations(add)
    if available("factor_k9_table.csv"):
        check_k9(add)
    if available("factor_sweep.csv"):
        sweep = pd.read_csv(RESULTS_DIR / "factor_sweep.csv")
        for trajectory, values in FACTOR_SWEEP.items():
            for k, value in zip(range(2, 13), values):
                comp = abs(sweep.loc[(sweep["trajectory"] == trajectory) & (sweep["k"] == k), "r"].iloc[0])
                add("factor sweep (figure panel A)", f"{trajectory} k = {k} |r| (within .001)", round(value, 4),
                    comp, abs(comp - value) <= 1e-3)
    if available("simulation_summary.csv"):
        sim = pd.read_csv(RESULTS_DIR / "simulation_summary.csv")
        for w, value in SIMULATION.items():
            comp = sim.loc[sim["w"] == w, "generalization_breadth_mean"].iloc[0]
            add("simulation", f"mean breadth at w = {w:g}", value, comp, check(comp, value, 1))
    if available("recovery_summary.csv"):
        rec = pd.read_csv(RESULTS_DIR / "recovery_summary.csv")
        for par, scheme, col, rep, dec in RECOVERY:
            comp = rec.loc[(rec["parameter"] == par) & (rec["scheme"] == scheme), col].iloc[0]
            add("parameter recovery", f"{par} {scheme} {col}", rep, comp, check(comp, rep, dec))

    report = pd.DataFrame(rows)
    n_fail = int((~report["match"]).sum())
    info_path = RESULTS_DIR / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    lines = ["# Regenerated results vs. manuscript", ""]
    if info:
        lines += [f"Psychometric fits used: **{info.get('psychometric_fits')}**. "
                  f"Python {info.get('python')}, NumPy {info.get('numpy')}, SciPy {info.get('scipy')}, "
                  f"pandas {info.get('pandas')}, statsmodels {info.get('statsmodels')}, "
                  f"factor_analyzer {info.get('factor_analyzer')}, scikit-learn {info.get('scikit-learn')} "
                  f"({info.get('platform')}).", ""]
    lines += [f"{len(report)} reported values checked; {len(report) - n_fail} match at reported precision; "
              f"{n_fail} differ.", ""]
    if not_run:
        lines += ["Not checked (results not generated in this run): " + ", ".join(sorted(set(not_run))), ""]
    if n_fail:
        lines += ["## Values that differ", "", report.loc[~report["match"]].to_markdown(index=False), ""]
    lines += ["## All checks", "", report.to_markdown(index=False), ""]
    (RESULTS_DIR / "manuscript_check.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(report)} values checked, {n_fail} differ")


def check_correlations(add) -> None:
    corr = pd.read_csv(RESULTS_DIR / "self_report_correlations.csv")
    for measure, scores in CORRELATIONS.items():
        for score, (r, p, q) in scores.items():
            row = corr.loc[(corr["measure"] == measure) & (corr["score"] == score)].iloc[0]
            add(f"correlations: {measure}", f"{row['score_label']} r", r, row["r"], check(row["r"], r, 2))
            if p is None:
                add(f"correlations: {measure}", f"{row['score_label']} p < .001", "< .001", row["p"], row["p"] < .001)
            else:
                add(f"correlations: {measure}", f"{row['score_label']} p", p, row["p"], check(row["p"], p, 3))
            if q is not None:
                add(f"correlations: {measure}", f"{row['score_label']} q", q, row["q"], check(row["q"], q, 3))


def check_intercorrelations(add) -> None:
    inter = pd.read_csv(RESULTS_DIR / "parameter_intercorrelations.csv")
    for x, y, r in INTERCORRELATIONS:
        comp = inter.loc[(inter["x"] == x) & (inter["y"] == y), "r"].iloc[0]
        add("parameter intercorrelations", f"{x} - {y}", r, comp, check(comp, r, 3))
    checks = pd.read_csv(RESULTS_DIR / "rt_pog_checks.csv")
    for y, r, p in RT_POG:
        row = checks.loc[checks["y"] == y].iloc[0]
        add("RT and POG", f"POG total - {y} r", r, row["r"], check(row["r"], r, 2))
        add("RT and POG", f"POG total - {y} p", p, row["p"], check(row["p"], p, 3))


def check_k9(add) -> None:
    k9 = pd.read_csv(RESULTS_DIR / "factor_k9_table.csv")
    k9["table_label"] = k9["label"].replace({"GAD-7": "Anxiety/Worry", "PSWQ": "Anxiety/Worry"})
    for label, (r, p, q, pur) in K9_TABLE.items():
        match = k9.loc[k9["table_label"] == label]
        if match.empty:
            add("factor analysis k = 9", f"{label} factor present", "yes", "no", False)
            continue
        row = match.iloc[0]
        for name, rep, comp, dec in (("r", r, row["r"], 2), ("p", p, row["p"], 3),
                                     ("q", q, row["q"], 3), ("purity", pur, row["purity"], 2)):
            add("factor analysis k = 9", f"{label} {name}", rep, comp, check(comp, rep, dec))


if __name__ == "__main__":
    main()
