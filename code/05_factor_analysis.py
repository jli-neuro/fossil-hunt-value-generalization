#!/usr/bin/env python3
"""Step 5: exploratory factor analysis of the 135 substantive questionnaire items.

* Parallel analysis: eigenvalues of the item correlation matrix compared with
  the 95th percentile of eigenvalues from 100 random-normal datasets of the
  same size (seed 42). Retain the leading factors whose eigenvalue exceeds
  the threshold.
* For k = 2..12: maximum-likelihood EFA with oblimin rotation
  (factor_analyzer), regression factor scores, Pearson r with mu, and BH-FDR
  across the k factor-mu correlations within each solution.
* Factor labels: the item group with the highest mean |loading|. Purity: the
  share of a factor's summed squared loadings contributed by a named item set
  (Anxiety/Worry combines GAD-7 and PSWQ items). 7-Up and 7-Down items form
  separate groups, giving 13 groups for the 12 instruments.

Input : data/questionnaire_items.csv, derived/psychometric_parameters.csv
Output: results/parallel_analysis.csv, results/factor_sweep.csv,
        results/factor_k9_table.csv, results/factor_loadings_k3.csv,
        results/factor_loadings_k9.csv,
        derived/factor_scores_k9.csv (participant level)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# factor_analyzer calls sklearn's check_array with the pre-1.6 keyword
# force_all_finite; translate it for newer scikit-learn releases.
import sklearn.utils.validation as _skval
_original_check_array = _skval.check_array


def _check_array_compat(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _original_check_array(*args, **kwargs)


_skval.check_array = _check_array_compat
import factor_analyzer.factor_analyzer as _fa_module  # noqa: E402
_fa_module.check_array = _check_array_compat
from factor_analyzer import FactorAnalyzer  # noqa: E402

from pog_common import DERIVED_DIR, RESULTS_DIR, bh_fdr, ensure_dirs, pearson_with_ci, read_data  # noqa: E402

GROUP_ORDER = ["SU", "bisbas", "ngs", "SD", "shaps", "pswq", "gad7",
               "pog", "phq", "pmq", "gse", "ie4", "wassup"]
GROUP_LABELS = {"SU": "Hypomania", "SD": "Depression", "bisbas": "BIS/BAS",
                "gad7": "GAD-7", "pswq": "PSWQ", "shaps": "SHAPS", "ngs": "NGS",
                "pog": "POG", "phq": "PHQ-9", "pmq": "PMQ-9", "gse": "GSE",
                "ie4": "IE-4", "wassup": "WASSUP"}
PURITY_SETS = {"GAD-7": ["gad7", "pswq"], "PSWQ": ["gad7", "pswq"]}
TRAJECTORIES = {"pog": ["pog"], "depression": ["SD"],
                "mania_approach": ["SU", "bisbas", "ngs", "wassup"]}
K_RANGE = range(2, 13)


def group_of(item: str) -> str:
    return item.rsplit("_", 1)[0]


def parallel_analysis(data: np.ndarray, n_iter: int = 100, percentile: float = 95,
                      seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n, p = data.shape
    observed = np.sort(np.linalg.eigvalsh(np.corrcoef(data.T)))[::-1]
    rng = np.random.default_rng(seed)
    random_eigs = np.zeros((n_iter, p))
    for i in range(n_iter):
        random_eigs[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(rng.normal(size=(n, p)).T)))[::-1]
    return observed, np.percentile(random_eigs, percentile, axis=0)


def label_factor(loadings: pd.DataFrame, factor: str) -> str:
    means = {g: loadings.loc[[i for i in loadings.index if group_of(i) == g], factor].abs().mean()
             for g in GROUP_ORDER}
    return GROUP_LABELS[max(means, key=means.get)]


def purity(loadings: pd.DataFrame, factor: str, groups: list[str]) -> float:
    squared = loadings[factor] ** 2
    mask = [group_of(i) in groups for i in loadings.index]
    return float(squared[mask].sum() / squared.sum())


def richest(loadings: pd.DataFrame, groups: list[str]) -> str:
    items = [i for i in loadings.index if group_of(i) in groups]
    return max(loadings.columns, key=lambda f: loadings.loc[items, f].abs().mean())


def main() -> None:
    ensure_dirs()
    items = read_data("questionnaire_items.csv")
    psych = pd.read_csv(DERIVED_DIR / "psychometric_parameters.csv", dtype={"participant_id": str})
    item_cols = [c for g in GROUP_ORDER for c in sorted(i for i in items.columns if group_of(i) == g)]
    data = items[item_cols]
    mu = items[["participant_id"]].merge(psych[["participant_id", "mu"]], on="participant_id")["mu"]
    if len(mu) != len(items):
        raise RuntimeError("Questionnaire and task participants do not match")

    observed, threshold = parallel_analysis(data.to_numpy(float))
    exceeds = observed > threshold
    k_retained = int(np.argmin(exceeds)) if not exceeds.all() else len(exceeds)
    pd.DataFrame({"factor": np.arange(1, len(observed) + 1), "observed_eigenvalue": observed,
                  "random_95th_percentile": threshold,
                  "retained": np.arange(len(observed)) < k_retained}).to_csv(
        RESULTS_DIR / "parallel_analysis.csv", index=False)
    print(f"Parallel analysis retains k = {k_retained}; {len(item_cols)} items, N = {len(data)}")

    sweep, tables = [], {}
    for k in K_RANGE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fa = FactorAnalyzer(n_factors=k, rotation="oblimin", method="ml", is_corr_matrix=False)
            fa.fit(data)
        factors = [f"F{i + 1}" for i in range(k)]
        loadings = pd.DataFrame(fa.loadings_, index=item_cols, columns=factors)
        scores = pd.DataFrame(fa.transform(data), columns=factors)
        if k == 9:
            scores.insert(0, "participant_id", items["participant_id"].to_numpy())
            scores.to_csv(DERIVED_DIR / "factor_scores_k9.csv", index=False)
            scores = scores.drop(columns="participant_id")
        res = pd.DataFrame([{"factor": f, **pearson_with_ci(mu, scores[f])} for f in factors])
        res["q"] = bh_fdr(res["p"])
        res["label"] = [label_factor(loadings, f) for f in factors]
        res["purity"] = [purity(loadings, f, PURITY_SETS.get(lab, [g for g, n in GROUP_LABELS.items() if n == lab]))
                         for f, lab in zip(factors, res["label"])]
        res["k"] = k
        tables[k] = (res, loadings)
        best = res.loc[res["r"].abs().idxmax()]
        sweep.append({"k": k, "trajectory": "largest_abs_r", **best.to_dict()})
        for name, groups in TRAJECTORIES.items():
            f = richest(loadings, groups)
            row = res.set_index("factor").loc[f]
            sweep.append({"k": k, "trajectory": name, "factor": f, **row.to_dict(),
                          "purity_of_trajectory_set": purity(loadings, f, groups)})
    pd.DataFrame(sweep).to_csv(RESULTS_DIR / "factor_sweep.csv", index=False)

    res9, load9 = tables[9]
    res9 = res9.assign(abs_r=res9["r"].abs()).sort_values("abs_r", ascending=False).drop(columns="abs_r")
    res9.insert(0, "rank", np.arange(1, len(res9) + 1))
    res9.to_csv(RESULTS_DIR / "factor_k9_table.csv", index=False)
    load9.to_csv(RESULTS_DIR / "factor_loadings_k9.csv", index_label="item")
    tables[3][1].to_csv(RESULTS_DIR / "factor_loadings_k3.csv", index_label="item")


if __name__ == "__main__":
    main()
