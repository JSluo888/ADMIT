#!/usr/bin/env python3
"""Cohort eligibility from the blind `--selection-only` runs (region counts only; no label, no endpoint): a
within-cohort XGBoost cohort with region identifiers and coordinates is eligible if it has >= 20 DEV regions AND
>= 100 TEST regions (regions with >= 50 cells) in every seed.

  python experiments/regions/eligibility.py runs/regions/_selection > runs/regions/ELIGIBLE.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CANDIDATES = ["cords2024_nsclc_imc", "danenberg2022_metabric_imc", "fischer2023_breast_imc", "meyer2025_tnbc_imc",
              "neotrip_tnbc_imc", "quiche_tnbc_mibi_spain", "quiche_tnbc_mibi_stanford", "launonen2022_hgsc_tcycif",
              "schurch"]
MIN_DEV, MIN_TEST, SEEDS = 20, 100, 5


def main(seldir: Path):
    rows, elig = [], []
    for c in CANDIDATES:
        fs = sorted((seldir / c).glob("regions_xgb_seed*_selection.json"),
                    key=lambda f: int(re.search(r"seed(\d+)_", f.name).group(1)))
        if not fs:
            rows.append(f"| {c} | not run | | | | no |")
            continue
        ms = [json.loads(f.read_text()) for f in fs]
        dev = [m["regions"]["n_regions_kept"]["dev"] for m in ms]
        test = [m["regions"]["n_regions_kept"]["test"] for m in ms]
        xy = min(m["xy_finite_frac_test"] or 0 for m in ms)
        sel = [m["selection"]["L6"]["selected"] for m in ms]
        cnt = "; ".join("/".join(str(s[r]["0.20"]["test"]) if r in s else "-" for s in sel)
                        for r in ("uncert", "voi", "density"))
        ok = (len(ms) == SEEDS and min(dev) >= MIN_DEV and min(test) >= MIN_TEST and xy > 0.99)
        if ok and c != "schurch":
            elig.append(c)
        rows.append(f"| {c} | {len(ms)} | {'/'.join(map(str, dev))} | {'/'.join(map(str, test))} | {xy:.3f} | "
                    f"{cnt} | {'**yes**' if ok else 'no'} |")
    print("# Eligible cohorts\n")
    print("Determined from the blind `--selection-only` runs (`runs/regions/_selection/`; region counts only, "
          "no label used, no endpoint computed), before any full run. Rule: within-cohort XGBoost cohort with "
          f"region identifiers and coordinates, >= {MIN_DEV} DEV regions AND >= {MIN_TEST} TEST regions in every seed "
          "(regions with >= 50 cells). Schurch is secondary (listed for completeness); Orion is qualitative "
          "only.\n")
    print("| cohort | seeds | DEV regions per seed | TEST regions per seed | finite xy (TEST, min) | selected TEST "
          "regions at b = 0.20, L6: uncert; voi; density (per seed) | eligible |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(r)
    print(f"\n**Eligible ({len(elig)}):** {', '.join(elig) if elig else 'none'}\n")
    print("Secondary (runnable, not eligible): " + ", ".join(
        c for c in CANDIDATES if c not in elig) + ". Orion: qualitative panel only.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
