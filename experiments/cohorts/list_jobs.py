#!/usr/bin/env python3
"""Print one command per run of the new-cohort experiments, as fixed in configs/cohort_registry.json.

  python experiments/cohorts/list_jobs.py within   # within-cohort paired deletion (run_within.py)
  python experiments/cohorts/list_jobs.py pairs    # real panel change / new cohort / site shift (run_pair.py)

Each line is independent; run them in any order or in parallel, e.g.
  python experiments/cohorts/list_jobs.py within | xargs -P 4 -I{} sh -c '{}'
Cohorts or pairs whose data are not present under $ADMIT_DATA_ROOT (no READY file) fail fast and can be skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import std  # noqa: E402

PAIR_HEADS = ["xgb", "mlp"]


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "within"
    reg = std.registry()
    here = "experiments/cohorts"
    if what == "within":
        for name, r in reg.items():
            if name.startswith("_") or name == "pairs" or "splits" not in r or "heads" not in r:
                continue
            for seed in r["splits"]["seeds"]:
                for head in r["heads"]:
                    print(f"python {here}/run_within.py --cohort {name} --head {head} --seed {seed}")
    elif what == "pairs":
        for pair, pr in reg["pairs"].items():
            for direction in pr.get("directions", ["ab", "ba"]):
                for seed in pr["splits"]["seeds"]:
                    for head in PAIR_HEADS:
                        print(f"python {here}/run_pair.py --pair {pair} --direction {direction} --head {head} "
                              f"--seed {seed}")
    else:
        sys.exit("usage: list_jobs.py within|pairs")


if __name__ == "__main__":
    main()
