# Per-participant summary data

De-identified summaries for the 163 participants in the analysis sample, made by
`code/export_summary_data.py`. They are enough to reproduce the psychometric
and response-time fits and the correlations with self-report without the
trial- or item-level data: run `bash run_all.sh summary` (or
`python code/reproduce_from_summary_data.py`). Trial-by-trial responses and
item-level questionnaire responses are available on request (see
`data/README.md`).

`participant_id` is a random release ID (POG001 to POG163), the same in every
file. Distances `d` are measured from each participant's rewarded exemplar
(0 = rewarded, 90 = nonrewarded, in steps of 10).

## participant_summary.csv

One row per participant.

| Column | Description |
|---|---|
| `mu`, `sigma`, `gamma`, `lambda` | Four-parameter logistic fit reported in the paper: boundary (generalization breadth), transition scale, lower and upper asymptotes |
| `auc` | Area under the fitted curve over d = 0 to 90, divided by 90 |
| `log_steepness` | Natural log of the maximum steepness S = (lambda - gamma) / (4 sigma) |
| `n_valid_rt` | Test trials with a response time between 0.2 and 8 s (of 90) |
| `beta_0`, `beta_d`, `beta_a` | Log-RT model: log RT = beta_0 + beta_d \|z\| + beta_a z, with z = (mu - d) / 10 |
| `beta_d_plus`, `beta_d_minus` | Rewarded- and nonrewarded-direction slopes (beta_d + beta_a, beta_d - beta_a) |
| `mean_log_rt`, `mean_rt`, `median_rt` | Mean log RT (log seconds), mean and median RT (seconds) over valid trials |
| `mean_log_rt_rewarded_exemplar` | Mean log RT at d = 0 |
| `pog_total`, `pog_upward`, `pog_social`, `pog_lateral` | POG scale: total and the upward, social and lateral subscales |
| `7up7down_composite`, `7up`, `7down` | 7 Up 7 Down Inventory: composite, 7-Up and 7-Down |
| `pmq9`, `phq9` | Patient Mania Questionnaire (PMQ-9), Patient Health Questionnaire (PHQ-9) |
| `bas_drive`, `bas_reward`, `bis` | BIS/BAS subscales |
| `wassup`, `ngs`, `gse`, `gad7`, `pswq`, `shaps`, `ie4` | WASSUP, NGS, GSE, GAD-7, PSWQ, SHAPS and IE-4 totals |

## choice_by_distance.csv

One row per participant and stimulus distance: `distance`, `n_trials`,
`n_valuable` (trials on which the stimulus was predicted to be valuable) and
`p_valuable` (their proportion). These are the proportions the psychometric
function is fitted to.

## rt_by_distance.csv

One row per participant and stimulus distance, over valid trials (0.2 to 8 s):
`distance`, `n_valid_rt`, `mean_log_rt` (log seconds) and `median_rt`
(seconds). Weighting `mean_log_rt` by `n_valid_rt` reproduces the trial-level
log-RT model exactly, because its predictors depend only on distance.

## factor_scores_k9.csv

Regression factor scores from the nine-factor solution (maximum likelihood,
oblimin rotation) of the 135 questionnaire items. Columns are named by factor
number and by the item set with the highest mean absolute loading, as in the
paper's k = 9 table. Factor signs are arbitrary; the loadings are in
`results/factor_loadings_k9.csv`.
