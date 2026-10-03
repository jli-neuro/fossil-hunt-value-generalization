# Data files (available on request)

The full pipeline (steps 1 to 7) reads three de-identified CSV files with
trial-by-trial and item-level responses. They are not posted here; they are
shared on reasonable request to the corresponding author, for research
consistent with the original consent. The per-participant summaries in
`summary_data/` are public, and the main results can be reproduced from them
alone (see the main README). To run the full pipeline, place the files in this
folder (or point `POG_DATA_DIR` at them) and run `run_all.sh`.

| File | Rows | Contents |
|---|---|---|
| `test_trials.csv` | 14,670 | One row per test trial (90 per participant, N = 163) |
| `questionnaire_items.csv` | 163 | 135 questionnaire items per participant, in scoring units |
| `reported_psychometric_fits.csv` | 163 | The four-parameter logistic fits used in the manuscript, and the order in which participants were processed in the original analysis |

`data_dictionary.csv` (in this folder) describes every column, with missing
counts and observed ranges.

## Conventions

- `participant_id` is a random release ID (POG001 to POG163), the same in all
  files. Prolific IDs, dates, device information and free text are not included.
- `distance_from_rewarded_endpoint` (d in the manuscript) is the test
  stimulus's distance from the participant's rewarded exemplar: 0 = rewarded,
  90 = nonrewarded, in steps of 10. `r_bin` is the reward-normalized
  coordinate of the original analysis (0 = nonrewarded, 90 = rewarded),
  `r_bin = 90 - d`.
- `pred_reward` is 1 when the participant predicted the stimulus was valuable.
- `rt` is the response time in seconds. `rt_valid` marks trials inside the
  0.2-8 s analysis window (14,571 trials).
- In `reported_psychometric_fits.csv`, `mu` is in the reporting coordinate d.
  `analysis_order` is used only to give each participant the same random-number
  stream as in the original parameter-recovery run.

## Questionnaire item coding

Items are numbered in substantive-item order, with the embedded attention
checks removed. 7-Up items are raw questions 1, 3, 4, 6, 7, 8 and 14 of the
combined 7-Up/7-Down form, and 7-Down items are raw questions 2, 5, 10, 11, 12,
13 and 15. GAD-7 omits raw question 5, NGS omits 4, POG omits 10, SHAPS omits
7 and GSE omits 8 (the attention checks); BIS/BAS uses raw questions 1 to 12
and omits the check at 13. Other item sequences are consecutive.

Items are stored in the units that are summed to form each score:

- 7-Up, 7-Down, GAD-7, SHAPS, PHQ-9 and PMQ-9: response options 1-4 coded 0-3.
- BIS/BAS and GSE: 1-4.
- PSWQ, IE-4 and WASSUP: 1-5.
- POG: response options 1-5 coded 5-1.
- NGS: the numeric responses as recorded.

Each score is the sum of its items; `code/pog_common.py` (`SCALE_ITEMS`) lists
the items in each of the 19 scores, including the POG subscales and the
BIS, BAS Reward and BAS Drive subscales.
