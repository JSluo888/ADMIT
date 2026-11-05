#!/usr/bin/env python3
"""Write the prespecified Fig 4 analysis plan (generic + per cohort). Never overwrites an existing plan with
different content: a changed plan must be written as a dated amendment (--amend "<reason>")."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_common import OUT, PLAN_GENERIC, COHORTS, plan_for, plan_hash, write_json  # noqa: E402


def put(path: Path, obj: dict, amend: str | None):
    h = plan_hash(obj)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('plan_hash') == h:
            print('unchanged', path, h)
            return
        if not amend:
            sys.exit(f'{path} exists with a different hash ({old.get("plan_hash")} vs {h}); use --amend "<reason>"')
        hist = old.get('amendments', []) + [{'at': datetime.now().isoformat(timespec='seconds'), 'reason': amend,
                                             'previous_hash': old.get('plan_hash')}]
    else:
        hist = []
    write_json({'written_at': datetime.now().isoformat(timespec='seconds'), 'plan_hash': h, 'amendments': hist,
                'plan': obj}, path)
    print('wrote', path, h)


ap = argparse.ArgumentParser()
ap.add_argument('--cohorts', nargs='*', default=list(COHORTS))
ap.add_argument('--amend', default=None)
a = ap.parse_args()
put(OUT / 'analysis_plan.json', PLAN_GENERIC, a.amend)
for c in a.cohorts:
    put(OUT / c / 'analysis_plan.json', plan_for(c), a.amend)
