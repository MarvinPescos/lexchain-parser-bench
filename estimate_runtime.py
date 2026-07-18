#!/usr/bin/env python3
"""Project full-run wall time from smoke-test results.

Reads per-doc meta files under --results-dir, computes s/page per tool, and
projects the makespan for the full Law set on a 2-GPU schedule (the longest
tool runs alone on one GPU; the other two share the second).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOOLS_DEFAULT = ["docling", "mineru", "marker"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_kaggle = Path("/kaggle").exists()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/kaggle/working/results" if on_kaggle else "results"))
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/kaggle/tmp/ohr_data" if on_kaggle else "data"))
    ap.add_argument("--tools", default=",".join(TOOLS_DEFAULT))
    ap.add_argument("--label", default="", help="tag printed in the header, e.g. 'scanned'")
    args = ap.parse_args()

    gt_files = list((args.data_dir / "gt" / "law").glob("*.json"))
    total_pages = sum(len(json.loads(p.read_text())) for p in gt_files)
    tag = f" [{args.label}]" if args.label else ""
    print(f"=== Full-run estimate{tag}: {total_pages} pages, {len(gt_files)} docs ===")

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    totals = {}
    for tool in tools:
        metas = [json.loads(p.read_text())
                 for p in (args.results_dir / tool / "meta").glob("*.json")]
        ok = [m for m in metas
              if m.get("status") == "success" and m.get("wall_s") and m.get("pages")]
        if not ok:
            print(f"{tool}: NO successful docs yet -- check "
                  f"{args.results_dir}/{tool}/worker.log before a full run")
            continue
        rate = sum(m["wall_s"] for m in ok) / sum(m["pages"] for m in ok)
        lt = args.results_dir / tool / "load_time.json"
        load = json.loads(lt.read_text()).get("model_load_s", 0) if lt.exists() else 0
        totals[tool] = rate * total_pages + load
        print(f"{tool}: {rate:.1f} s/page over {len(ok)} doc(s) "
              f"(model load {load:.0f}s) -> ~{totals[tool] / 3600:.1f} h alone")

    if len(totals) == len(tools) and len(totals) >= 3:
        t = sorted(totals.values(), reverse=True)
        makespan = max(t[0], sum(t[1:]))
        print(f"\nProjected wall time on 2 GPUs: ~{makespan / 3600:.1f} h")
        print("Kaggle GPU sessions cap at ~12 h. If the estimate exceeds that, run in "
              "chunks -- every document is checkpointed; rerunning the run cell resumes.")
    elif len(totals) == 2:
        print(f"\nProjected wall time on 2 GPUs: ~{max(totals.values()) / 3600:.1f} h")


if __name__ == "__main__":
    main()
