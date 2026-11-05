#!/usr/bin/env python3
"""Copy the committed (pre-specified) panel banks and the Schurch label map into $ADMIT_RUNS.

The drivers read their fixed panel banks from the output tree and only generate a bank when none exists. Run this
once before any experiment so that every driver uses exactly the banks of the paper:

  configs/panel_banks/orion_panel_bank.json      -> $ADMIT_RUNS/panel_bank.json
  configs/panel_banks/schurch_panel_bank.json    -> $ADMIT_RUNS/schurch/schurch_panel_bank.json
  configs/schurch_label_mapping.json             -> $ADMIT_RUNS/schurch/schurch_label_mapping.json
  configs/panel_banks/cohorts/<cohort>.json      -> $ADMIT_RUNS/cohorts/<cohort>/panel_bank.json

Existing files are never overwritten (a differing existing file is reported).
"""
from __future__ import annotations

import filecmp
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("ADMIT_RUNS", str(REPO / "runs")))
CFG = REPO / "configs"


def main():
    pairs = [(CFG / "panel_banks" / "orion_panel_bank.json", RUNS / "panel_bank.json"),
             (CFG / "panel_banks" / "schurch_panel_bank.json", RUNS / "schurch" / "schurch_panel_bank.json"),
             (CFG / "schurch_label_mapping.json", RUNS / "schurch" / "schurch_label_mapping.json")]
    for f in sorted((CFG / "panel_banks" / "cohorts").glob("*.json")):
        pairs.append((f, RUNS / "cohorts" / f.stem / "panel_bank.json"))
    for src, dst in pairs:
        if dst.exists():
            print(("same     " if filecmp.cmp(src, dst, shallow=False) else "DIFFERS  ") + str(dst))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print("copied   " + str(dst))


if __name__ == "__main__":
    main()
