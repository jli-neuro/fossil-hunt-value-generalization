#!/usr/bin/env python3
"""Step 3: score the 19 questionnaire scores from item-level responses.

Input : data/questionnaire_items.csv (items already in scoring units)
Output: derived/questionnaire_scores.csv
"""
from __future__ import annotations

import pandas as pd

from pog_common import DERIVED_DIR, SCALE_ITEMS, ensure_dirs, read_data


def main() -> None:
    ensure_dirs()
    items = read_data("questionnaire_items.csv")
    missing = sorted({c for cols in SCALE_ITEMS.values() for c in cols} - set(items.columns))
    if missing:
        raise KeyError(f"Missing item columns: {missing}")
    scores = pd.DataFrame({"participant_id": items["participant_id"]})
    for name, cols in SCALE_ITEMS.items():
        scores[name] = items[cols].sum(axis=1, min_count=len(cols))
    scores.to_csv(DERIVED_DIR / "questionnaire_scores.csv", index=False)
    n_items = len({c for c in items.columns if c != "participant_id"})
    print(f"Scored {len(scores)} participants on {len(SCALE_ITEMS)} scores from {n_items} items")


if __name__ == "__main__":
    main()
