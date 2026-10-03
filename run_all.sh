#!/usr/bin/env bash
# Reproduce the analyses from the de-identified data files (see README.md).
#
#   bash run_all.sh          run every step (a few minutes on 4 cores)
#   bash run_all.sh 6 7      run only the listed steps (1-7, sim, export, summary)
#   bash run_all.sh summary  reproduce the main results from summary_data/ alone
#
# Folders (override with environment variables):
#   POG_DATA_DIR     test_trials.csv, questionnaire_items.csv,
#                    reported_psychometric_fits.csv          (default: ./data)
#   POG_DERIVED_DIR  participant-level outputs; do not share   (default: ./derived)
#   POG_RESULTS_DIR  aggregate results                          (default: ./results)
#   POG_PSYCHOMETRIC_FITS  'reported' (default when the reported fits are
#                    present) or 'refit'; see README.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export POG_DATA_DIR="${POG_DATA_DIR:-$ROOT/data}"
export POG_DERIVED_DIR="${POG_DERIVED_DIR:-$ROOT/derived}"
export POG_RESULTS_DIR="${POG_RESULTS_DIR:-$ROOT/results}"
export MPLBACKEND=Agg

STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(1 2 3 4 5 6 sim 7)
fi

cd "$ROOT/code"
for step in "${STEPS[@]}"; do
  echo "== step $step"
  case "$step" in
    1) python 01_fit_psychometric.py ;;
    2) python 02_fit_rt_model.py ;;
    3) python 03_score_questionnaires.py ;;
    4) python 04_self_report_correlations.py ;;
    5) python 05_factor_analysis.py ;;        # about 2 minutes
    6) python 06_parameter_recovery.py ;;     # about 1 minute on 4 cores
    sim)
      # Simulation (no participant data); deterministic at seed 1.
      ( cd simulation && python run_simulation.py && python verify_simulation.py )
      mkdir -p "$POG_RESULTS_DIR/simulation"
      for f in simulation_summary.csv simulation_run_level.csv simulation_manifest.json simulation_results.npz; do
        mv "simulation/$f" "$POG_RESULTS_DIR/simulation/$f"
      done
      rm -rf "$POG_RESULTS_DIR/simulation/figures"
      mv simulation/figures "$POG_RESULTS_DIR/simulation/figures"
      cp "$POG_RESULTS_DIR/simulation/simulation_summary.csv" "$POG_RESULTS_DIR/simulation_summary.csv"
      ;;
    7) python 07_check_against_manuscript.py ;;
    export) python export_summary_data.py ;;          # after steps 1-3 and 5; full data needed
    summary) python reproduce_from_summary_data.py ;; # needs only summary_data/
    *) echo "Unknown step: $step (use 1-7, sim, export or summary)" >&2; exit 1 ;;
  esac
done
