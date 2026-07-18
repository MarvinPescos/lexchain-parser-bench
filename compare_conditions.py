#!/usr/bin/env python3
"""Combined condition-aware evaluation: one paper-ready table with three slices.

  digital-born       original run (results/),        docs from digital_docs.txt
  natively-scanned   original run (results/),        docs from scanned_docs.txt
  simulated-scanned  OCR-forced run (results_scanned/), all docs

Filter files are produced automatically (evaluate.detect_scanned) if missing.
Outputs comparison.csv + comparison.md into the digital results dir, and flags
whether the tool ranking (by NED, TEDS, s/page) differs between conditions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate import (
    PAPER_COLS,
    TOOLS_DEFAULT,
    detect_scanned,
    evaluate_slice,
    load_doc_filter,
    load_gt,
)

RANKING_METRICS = [  # (column, higher_is_better)
    ("ned", False),
    ("teds", True),
    ("s_per_page", False),
]


def ranking(rows: list[dict], metric: str, higher_better: bool):
    scored = [r for r in rows if r.get(metric) is not None]
    if len(scored) < 2:
        return None
    return tuple(r["tool"] for r in sorted(scored, key=lambda r: r[metric],
                                           reverse=higher_better))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_kaggle = Path("/kaggle").exists()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/kaggle/tmp/ohr_data" if on_kaggle else "data"))
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/kaggle/working/results" if on_kaggle else "results"),
                    help="digital-condition results (also receives the outputs)")
    ap.add_argument("--scanned-results-dir", type=Path,
                    default=Path("/kaggle/working/results_scanned" if on_kaggle
                                 else "results_scanned"))
    ap.add_argument("--tools", default=",".join(TOOLS_DEFAULT))
    args = ap.parse_args()
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]

    gt = load_gt(args.data_dir)
    if not gt:
        raise SystemExit(f"No ground truth under {args.data_dir}/gt/law")

    digital_f = args.results_dir / "digital_docs.txt"
    scanned_f = args.results_dir / "scanned_docs.txt"
    if not (digital_f.exists() and scanned_f.exists()):
        detect_scanned(args.data_dir / "pdfs" / "law", args.results_dir)

    slices = [
        ("digital-born", args.results_dir, load_doc_filter(digital_f)),
        ("natively-scanned", args.results_dir, load_doc_filter(scanned_f)),
        ("simulated-scanned", args.scanned_results_dir, None),
    ]

    all_rows = []
    rankings: dict[str, dict[str, tuple]] = {m: {} for m, _ in RANKING_METRICS}
    for label, results_dir, doc_filter in slices:
        n = len(doc_filter) if doc_filter is not None else len(gt)
        print(f"\n--- {label} (n={n}, {results_dir}) ---")
        if not results_dir.exists():
            print(f"  {results_dir} missing -- skipping this slice")
            continue
        _, summaries = evaluate_slice(tools, results_dir, gt, doc_filter)
        for s in summaries:
            all_rows.append({"condition": label, **s})
        for metric, hb in RANKING_METRICS:
            r = ranking(summaries, metric, hb)
            if r:
                rankings[metric][label] = r

    if not all_rows:
        raise SystemExit("No results found in any slice.")

    df = pd.DataFrame(all_rows)
    cols = [("condition", "Condition"), ("docs", "n")] + PAPER_COLS
    paper = df[[c for c, _ in cols]].rename(columns=dict(cols))
    md_table = paper.to_markdown(index=False, floatfmt=".4f")

    lines = ["## Ranking stability across conditions", ""]
    any_diff = False
    for metric, _ in RANKING_METRICS:
        per_cond = rankings[metric]
        if len(per_cond) < 2:
            continue
        orders = set(per_cond.values())
        if len(orders) > 1:
            any_diff = True
            lines.append(f"- **{metric}: ranking DIFFERS between conditions**")
        else:
            lines.append(f"- {metric}: ranking consistent "
                         f"({' > '.join(next(iter(orders)))})")
        for cond, order in per_cond.items():
            if len(orders) > 1:
                lines.append(f"    - {cond}: {' > '.join(order)}")
    if not any_diff:
        lines.append("\nTool ranking is stable across all conditions.")
    ranking_md = "\n".join(lines)

    out_md = md_table + "\n\n" + ranking_md + "\n"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_dir / "comparison.csv", index=False)
    (args.results_dir / "comparison.md").write_text(out_md, encoding="utf-8")
    print("\n" + out_md)
    print(f"Wrote {args.results_dir}/comparison.csv and comparison.md")


if __name__ == "__main__":
    main()
