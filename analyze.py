"""Print summary tables for all runs in results/. Read-only; no API calls."""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).parent
DIMS = ["purpose_focus_organization", "evidence_elaboration", "conventions", "total"]


def fmt(x):
    if x is None:
        return "  -- "
    return f"{x:>5}"


def main() -> None:
    files = sorted(glob.glob(str(ROOT / "results" / "*.json")))
    if not files:
        print("No result files in results/. Run run_experiment.py first.")
        return

    for f in files:
        d = json.load(open(f))
        cfg = d["config"]
        n = cfg["trials_per_condition"]
        print(f"\n=== {Path(f).name} ===")
        print(f"  model={cfg['model']} | essay={cfg.get('essay_file','?')} | "
              f"temperature={cfg['temperature']} | trials/cond={n}")
        header = f"  {'condition':>22s}  {'dim':>30s}  {'mean':>5}  {'stdev':>5}  {'range':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cond in ("prompt_only", "prompt_plus_simple", "prompt_plus_complex"):
            s = d["summary"].get(cond)
            if not s:
                continue
            for dim in DIMS:
                st = s[dim]
                rng = (f"{st['min']}-{st['max']}"
                       if st["min"] is not None else "  -- ")
                tag = cond if dim == DIMS[0] else ""
                print(f"  {tag:>22s}  {dim:>30s}  "
                      f"{fmt(st['mean'])}  {fmt(st['stdev'])}  {rng:>8}")
            print()


if __name__ == "__main__":
    main()
