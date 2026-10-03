# Fossil Hunt: value generalization and positive overgeneralization

This repository contains the analysis code, per-participant summary data and summary results for the paper "Value Generalization along a Perceptual Continuum Tracks Self-Reported Positive Overgeneralization and Bipolar Traits". The code fits participants' reward predictions in the Fossil Hunt task, models their response times, scores the questionnaires, relates the task measures to self-reported traits, and runs the factor analyses, parameter recovery and learning simulation reported in the paper.

## Overview

Project components:
- Four-parameter logistic fits of each participant's reward predictions (generalization breadth µ)
- A response-time model relating log RT to distance from each participant's choice boundary (β_D, β_A, β_D⁺)
- Scoring of 19 questionnaire scores from 135 items, and their correlations with the task measures, with FDR correction
- Parallel analysis and exploratory factor analysis of the questionnaire items
- Participant-matched parameter recovery for the psychometric and RT models
- A weighted-extrema reinforcement-learning simulation (Zorowitz, Momennejad & Daw, 2020)
- A check that compares every regenerated value with the value reported in the paper
- De-identified per-participant summary data, with a script that reproduces the main results from them

## Reproducing the paper's results

`run_all.sh` runs the analysis in order. Each step writes aggregate results to `results/`, and the last step compares the regenerated values with those reported in the paper (`results/manuscript_check.md`).

| Step | Script | Paper content |
|---|---|---|
| 1 | `code/01_fit_psychometric.py` | Psychometric fits (µ, σ, γ, λ), AUC, steepness, group curve |
| 2 | `code/02_fit_rt_model.py` | Log-RT model (β_D, β_A, β_D⁺) and RT descriptives |
| 3 | `code/03_score_questionnaires.py` | The 19 questionnaire scores |
| 4 | `code/04_self_report_correlations.py` | Correlations of µ, β_D, β_A and β_D⁺ with the 19 scores; parameter intercorrelations; RT checks |
| 5 | `code/05_factor_analysis.py` | Parallel analysis; factor analyses for k = 2 to 12; the k = 9 table; factor-analysis figure, panels A-B |
| 6 | `code/06_parameter_recovery.py` | Parameter recovery for the psychometric and RT models |
| sim | `code/simulation/` | Weighted-extrema simulation and its figure |
| 7 | `code/07_check_against_manuscript.py` | Comparison of every regenerated value with the paper |
| summary | `code/reproduce_from_summary_data.py` | Fits and correlations reproduced from `summary_data/` alone |

Correction rules (step 4): for µ, the two primary tests (POG total and the 7Up/7Down composite) are reported without adjustment and Benjamini-Hochberg FDR runs over the other 17 scores; for β_D, β_A and β_D⁺, FDR runs over all 19 scores, separately for each measure. q-values are computed from unrounded p-values.

## Repository Structure

```
.
├── code/
│   ├── pog_common.py                   # Shared definitions: models, scoring keys, correction rules
│   ├── 01_fit_psychometric.py          # Psychometric fits and group curve
│   ├── 02_fit_rt_model.py              # Log-RT model
│   ├── 03_score_questionnaires.py      # Questionnaire scoring
│   ├── 04_self_report_correlations.py  # Correlations with self-report, FDR correction
│   ├── 05_factor_analysis.py           # Parallel analysis and factor analysis
│   ├── 06_parameter_recovery.py        # Parameter recovery
│   ├── 07_check_against_manuscript.py  # Comparison with the reported values
│   ├── export_summary_data.py          # Builds summary_data/ from the full data
│   ├── reproduce_from_summary_data.py  # Reproduces the main results from summary_data/
│   └── simulation/                     # Weighted-extrema learning simulation
├── summary_data/                       # De-identified per-participant summaries
│   ├── participant_summary.csv         # Task parameters, RT summaries, 19 questionnaire scores
│   ├── choice_by_distance.csv          # Choice proportions at each stimulus distance
│   ├── rt_by_distance.csv              # Log-RT summaries at each stimulus distance
│   ├── factor_scores_k9.csv            # Nine-factor solution scores
│   └── README.md                       # Column descriptions
├── data/
│   ├── README.md                       # Trial- and item-level files, available on request
│   └── data_dictionary.csv             # Every column of those files
├── results/                            # Summary results (aggregate)
│   ├── from_summary_data/              # Results reproduced from summary_data/
│   ├── simulation/                     # Simulation outputs and figures
│   └── sensitivity_refit/              # Comparison using refitted psychometric parameters
├── run_all.sh                          # Runs the full analysis
├── environment.yml                     # Conda environment
├── requirements.txt                    # Pinned Python packages
├── CITATION.cff
└── LICENSE
```

## Requirements

Python 3.12 with NumPy 2.1.3, SciPy 1.17.0, pandas 2.2.3, statsmodels 0.14.6, scikit-learn 1.9.1, factor_analyzer 0.5.1, matplotlib and tabulate. The NumPy, SciPy, pandas and statsmodels versions match the environment of the original analysis. factor_analyzer 0.5.1 predates scikit-learn 1.6, so `05_factor_analysis.py` includes a short compatibility shim.

## Installation

With conda:

```bash
conda env create -f environment.yml
conda activate fossil-hunt
```

Or with pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

To reproduce the main results from the public summary data (no data request needed):

```bash
bash run_all.sh summary
```

This refits the psychometric functions to `summary_data/choice_by_distance.csv`, recomputes the RT coefficients from `summary_data/rt_by_distance.csv`, recomputes the correlations with the 19 questionnaire scores and with the k = 9 factor scores, and checks them against the paper (`results/from_summary_data/summary_data_check.md`).

With the full data in `data/`:

```bash
bash run_all.sh          # every step, a few minutes on 4 cores
bash run_all.sh 5 7      # selected steps (1-7, sim, export, summary)
```

The simulation needs no participant data and can be run on its own:

```bash
cd code/simulation
python run_simulation.py
python verify_simulation.py
```

The data, intermediate and results folders can be changed with the environment variables `POG_DATA_DIR`, `POG_DERIVED_DIR` and `POG_RESULTS_DIR`.

## Data Requirements

The summary reproduction needs only `summary_data/`, which is included. Steps 1 to 7 also read three de-identified files placed in `data/`: `test_trials.csv`, `questionnaire_items.csv` and `reported_psychometric_fits.csv`. They are described in `data/README.md` and `data/data_dictionary.csv`. Intermediate files from the full pipeline are written to `derived/`. The data files in `data/` and the `derived/` folder are excluded from version control by `.gitignore`; do not commit or redistribute them.

## Data Availability

De-identified per-participant summary data are included in `summary_data/`. Trial-by-trial responses and item-level questionnaire responses are not posted; they are available from the corresponding author upon reasonable request. Everything in `results/` is aggregate.

## Psychometric Fits

Each participant's choices are fitted with a four-parameter logistic by unweighted least squares (`fit_4pl` in `code/pog_common.py`). For most participants the best fit is unique. For a few, choices switch completely between two neighbouring stimuli, so any boundary µ inside that 10-unit gap fits equally well and σ shrinks toward zero; where the optimizer stops then depends on the build of the numerical libraries rather than on the data. Refitting on a different computer moved µ by more than 0.01 for 4 of the 163 participants, by at most 1.2 units (`results/psychometric_refit_agreement.csv`).

So that the code reproduces the published numbers, the participant-level fits used in the paper are included in `summary_data/participant_summary.csv` and, for the full pipeline, in `reported_psychometric_fits.csv` with the data shared on request. Step 1 always refits every participant and records the agreement with the published fits, then by default passes the published fits to the later steps. Set `POG_PSYCHOMETRIC_FITS=refit` to use the new fits instead. Doing so left the primary results unchanged at reported precision (µ with POG total: r = .21, p = .008; with the 7Up/7Down composite: r = .15, p = .050), moved other correlations by less than .01 in r and .05 in q, and moved no p- or q-value across .05 (`results/sensitivity_refit/manuscript_check.md`).

## Output

All outputs in `results/` are aggregate:
- `from_summary_data/`: correlation tables and checks reproduced from `summary_data/`
- `psychometric_summary.csv`, `rt_summary.csv`: descriptive statistics and the group curve
- `self_report_correlations.csv`: r, 95% CI, p and q for each task measure and score
- `parameter_intercorrelations.csv`, `rt_pog_checks.csv`, `questionnaire_descriptives.csv`
- `parallel_analysis.csv`, `factor_sweep.csv`, `factor_k9_table.csv`, `factor_loadings_k3.csv`, `factor_loadings_k9.csv`
- `recovery_summary.csv`
- `simulation/` and `simulation_summary.csv`
- `run_info.json`: software versions and which psychometric fits were used
- `manuscript_check.md`: every regenerated value next to the reported one

`manuscript_check.md` compares 366 reported values at the precision each is reported; 358 match. The other 8:
- **Factor analysis, k = 9 table (4 values).** The factor solution matches the original to within 3 × 10⁻⁴ in factor scores, but four p- or q-values lie on a rounding boundary and round the other way (for example, NGS p = .84049 here and .84055 in the original run).
- **Psychometric parameter recovery (4 values).** The paper's recovery correlations for µ, σ and γ and the σ bias come from an earlier run of the recovery simulation. Step 6 reproduces the later run that supplies the RT recovery values; in it, µ r = .989, σ r = .921, γ r = .968 and σ bias = -1.11 (paper: .988, .920, .971 and -1.10).

The simulation's learning runs are identical to the archived run. Its logistic curve fits can differ in the last decimals between library builds, so `verify_simulation.py` checks the condition means to ±0.005.

## Citation

If you use this code, please cite the associated paper:

```
@article{li2026value,
  title={Value Generalization along a Perceptual Continuum Tracks Self-Reported Positive Overgeneralization and Bipolar Traits},
  author={Li, Jing and Malaviya, Maya and Bennett, Daniel and Radulescu, Angela},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.07.12.737635}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Jing Li
- Maya Malaviya
- Daniel Bennett
- Angela Radulescu

## Contact

For questions or issues with the code, please use GitHub issues. For data requests, contact the corresponding author.
