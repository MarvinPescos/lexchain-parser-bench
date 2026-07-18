#!/usr/bin/env python3
"""Evaluate parsed outputs against OHR-Bench ground truth (Law domain).

Per tool, computes:
- NED  (normalized edit distance, per page, lower better)  -- OHR-Bench's OCR-quality metric
- CER  (character error rate, per page, lower better)      -- supplementary
- TEDS / TEDS-S (table similarity, higher better)          -- OmniDocBench-style
- Reading-order edit distance (lower better)               -- OmniDocBench-style
- mean runtime per page, success rate

Inputs:
  --data-dir/gt/law/*.json           ground truth: [{"page_idx": N, "text": "..."}]
  --results-dir/{tool}/meta/*.json   per-doc status + timing (written by run_benchmark.py)
  --results-dir/{tool}/pages/*.json  per-page parsed text
  --results-dir/{tool}/markdown/*.md full parsed markdown

Outputs (in --results-dir):
  results_per_doc.csv, results_summary.csv, results.md

CPU-only. Runs on Kaggle or a laptop (pip install -r requirements-eval.txt).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from teds import teds

TOOLS_DEFAULT = ["docling", "mineru", "marker"]

# ---------------------------------------------------------------- text metrics

_IMG_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_WS = re.compile(r"\s+")


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _IMG_MD.sub(" ", text)
    text = _HTML_COMMENT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def ned(pred: str, gt: str) -> float:
    """Levenshtein / max(len) in [0, 1]. 0 = identical."""
    if not gt and not pred:
        return 0.0
    return Levenshtein.normalized_distance(pred, gt)


def cer(pred: str, gt: str) -> float | None:
    """Levenshtein / len(gt). Can exceed 1 when pred hallucinates extra text."""
    if not gt:
        return None
    return Levenshtein.distance(pred, gt) / len(gt)


# ------------------------------------------------------------- reading order

_SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+|\n+")

RO_MIN_SEG_LEN = 30
RO_MAX_SEGMENTS = 40
RO_MATCH_THRESHOLD = 75.0
RO_MIN_MATCHED = 3


def _segments(gt_text: str) -> list[str]:
    segs, buf = [], ""
    for piece in _SENT_SPLIT.split(gt_text):
        buf = (buf + " " + piece).strip() if buf else piece.strip()
        if len(buf) >= RO_MIN_SEG_LEN:
            segs.append(buf)
            buf = ""
    if len(buf) >= RO_MIN_SEG_LEN:
        segs.append(buf)
    if len(segs) > RO_MAX_SEGMENTS:  # subsample evenly to bound cost
        step = len(segs) / RO_MAX_SEGMENTS
        segs = [segs[int(i * step)] for i in range(RO_MAX_SEGMENTS)]
    return segs


def reading_order_distance(gt_text: str, pred_text: str) -> float | None:
    """OmniDocBench-style reading-order score.

    Fuzzy-locate each GT segment in the prediction; the segment start offsets
    induce a permutation. Score = edit distance between that permutation and
    the identity, / number of matched segments. 0 = perfect order.
    Unmatched (dropped) segments are excluded -- NED already penalizes them.
    """
    positions = []
    for seg in _segments(gt_text):
        aln = fuzz.partial_ratio_alignment(seg, pred_text)
        if aln is not None and aln.score >= RO_MATCH_THRESHOLD:
            positions.append(aln.dest_start)
    if len(positions) < RO_MIN_MATCHED:
        return None
    rank_of = {pos: i for i, pos in enumerate(sorted(range(len(positions)), key=positions.__getitem__))}
    observed = [rank_of[i] for i in range(len(positions))]
    identity = list(range(len(positions)))
    return Levenshtein.distance(observed, identity) / len(positions)


# ------------------------------------------------------------------- tables

_HTML_TABLE = re.compile(r"<table\b.*?</table\s*>", re.DOTALL | re.IGNORECASE)
_LATEX_TABULAR = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.DOTALL)

TEDS_MATCH_MIN_SIM = 30.0


def _md_table_to_html(block: str) -> str:
    rows = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not (line.startswith("|") or line.count("|") >= 2):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue  # separator row
        rows.append(cells)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table>{body}</table>"


def _latex_tabular_to_html(block: str) -> str:
    inner = re.sub(r"\\begin\{tabular\}(\{[^}]*\})?|\\end\{tabular\}", "", block)
    inner = re.sub(r"\\[hc]line|\\(top|mid|bottom)rule", "", inner)
    rows = [r for r in re.split(r"\\\\", inner) if r.strip()]
    body = "".join(
        "<tr>" + "".join(f"<td>{c.strip()}</td>" for c in row.split("&")) + "</tr>"
        for row in rows
    )
    return f"<table>{body}</table>"


def _md_table_blocks(text: str) -> list[str]:
    blocks, current = [], []
    for line in text.splitlines():
        if line.strip().startswith("|") and line.count("|") >= 2:
            current.append(line)
        else:
            if len(current) >= 2:
                blocks.append("\n".join(current))
            current = []
    if len(current) >= 2:
        blocks.append("\n".join(current))
    return blocks


def extract_tables(text: str) -> list[str]:
    """All tables in a text, converted to normalized HTML strings."""
    tables = [m.group(0) for m in _HTML_TABLE.finditer(text)]
    without_html = _HTML_TABLE.sub(" ", text)
    tables += [_latex_tabular_to_html(m.group(0)) for m in _LATEX_TABULAR.finditer(without_html)]
    without_latex = _LATEX_TABULAR.sub(" ", without_html)
    tables += [_md_table_to_html(b) for b in _md_table_blocks(without_latex)]
    return tables


def _table_text(html_str: str) -> str:
    return norm_text(re.sub(r"<[^>]+>", " ", html_str))


def match_and_score_tables(gt_tables: list[str], pred_tables: list[str]):
    """Greedy-match GT tables to predicted tables by content similarity.

    Returns (teds_scores, teds_s_scores), one entry per GT table.
    Unmatched GT tables score 0 (the parser missed them).
    """
    gt_texts = [_table_text(t) for t in gt_tables]
    pred_texts = [_table_text(t) for t in pred_tables]
    pairs = sorted(
        (
            (fuzz.ratio(gt_texts[i], pred_texts[j]), i, j)
            for i in range(len(gt_tables))
            for j in range(len(pred_tables))
        ),
        reverse=True,
    )
    match: dict[int, int] = {}
    used_pred = set()
    for sim, i, j in pairs:
        if sim < TEDS_MATCH_MIN_SIM:
            break
        if i in match or j in used_pred:
            continue
        match[i] = j
        used_pred.add(j)
    scores, scores_s = [], []
    for i, gt_html in enumerate(gt_tables):
        pred_html = pred_tables[match[i]] if i in match else ""
        t = teds(pred_html, gt_html, structure_only=False)
        ts = teds(pred_html, gt_html, structure_only=True)
        if t is not None:
            scores.append(t)
        if ts is not None:
            scores_s.append(ts)
    return scores, scores_s


# ----------------------------------------------------------------- data I/O


def load_gt(data_dir: Path) -> dict[str, list[str]]:
    """doc stem -> list of page texts ordered by page_idx."""
    gt = {}
    for path in sorted((data_dir / "gt" / "law").glob("*.json")):
        pages = json.loads(path.read_text(encoding="utf-8"))
        pages = sorted(pages, key=lambda p: p.get("page_idx", 0))
        gt[path.stem] = [p.get("text", "") for p in pages]
    return gt


def load_pred_pages(tool_dir: Path, stem: str):
    """Returns (paginated: bool, pages: list[str] ordered)  or (False, [full_md])."""
    pages_path = tool_dir / "pages" / f"{stem}.json"
    if pages_path.exists():
        data = json.loads(pages_path.read_text(encoding="utf-8"))
        if data.get("paginated") and data.get("pages"):
            ordered = [
                data["pages"][k]
                for k in sorted(data["pages"], key=lambda x: int(x))
            ]
            return True, ordered
    md_path = tool_dir / "markdown" / f"{stem}.md"
    if md_path.exists():
        return False, [md_path.read_text(encoding="utf-8")]
    return False, []


# --------------------------------------------------------------- evaluation


def align_pages(gt_pages: list[str], pred_pages: list[str]):
    """Pair GT pages with predicted pages positionally.

    If counts differ (tool merged/dropped pages), fall back to doc-level:
    a single pair of concatenated texts.
    """
    if len(pred_pages) == len(gt_pages):
        return list(zip(gt_pages, pred_pages)), True
    return [("\n".join(gt_pages), "\n".join(pred_pages))], False


def evaluate_doc(gt_pages: list[str], pred_pages: list[str], paginated: bool):
    if paginated:
        pairs, page_aligned = align_pages(gt_pages, pred_pages)
    else:
        pairs, page_aligned = [("\n".join(gt_pages), pred_pages[0] if pred_pages else "")], False

    neds, cers, ros = [], [], []
    for gt_raw, pred_raw in pairs:
        gt_n, pred_n = norm_text(gt_raw), norm_text(pred_raw)
        neds.append(ned(pred_n, gt_n))
        c = cer(pred_n, gt_n)
        if c is not None:
            cers.append(c)
        ro = reading_order_distance(gt_n, pred_n)
        if ro is not None:
            ros.append(ro)

    gt_tables = extract_tables("\n".join(gt_pages))
    pred_tables = extract_tables("\n".join(pred_pages))
    teds_scores, teds_s_scores = match_and_score_tables(gt_tables, pred_tables)

    return {
        "ned": statistics.mean(neds) if neds else None,
        "cer": statistics.mean(cers) if cers else None,
        "reading_order": statistics.mean(ros) if ros else None,
        "ro_pages": len(ros),
        "teds_scores": teds_scores,
        "teds_s_scores": teds_s_scores,
        "gt_tables": len(gt_tables),
        "pred_tables": len(pred_tables),
        "page_aligned": page_aligned,
    }


def evaluate_tool(tool: str, results_dir: Path, gt: dict[str, list[str]]):
    tool_dir = results_dir / tool
    meta_dir = tool_dir / "meta"
    rows = []
    all_teds, all_teds_s = [], []
    if not meta_dir.exists():
        return rows, all_teds, all_teds_s
    for meta_path in sorted(meta_dir.glob("*.json")):
        stem = meta_path.stem
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gt_pages = gt.get(stem)
        row = {
            "tool": tool,
            "doc": stem,
            "status": meta.get("status"),
            "wall_s": meta.get("wall_s"),
            "pages": meta.get("pages") or (len(gt_pages) if gt_pages else None),
            "error": (meta.get("error") or "")[:200],
        }
        if meta.get("status") == "success" and gt_pages:
            paginated, pred_pages = load_pred_pages(tool_dir, stem)
            m = evaluate_doc(gt_pages, pred_pages, paginated)
            all_teds += m.pop("teds_scores")
            all_teds_s += m.pop("teds_s_scores")
            row.update(m)
        rows.append(row)
    return rows, all_teds, all_teds_s


def summarize(tool: str, rows: list[dict], all_teds, all_teds_s):
    n = len(rows)
    ok = [r for r in rows if r["status"] == "success"]
    scored = [r for r in ok if r.get("ned") is not None]
    timed = [r for r in ok if r.get("wall_s") and r.get("pages")]
    total_wall = sum(r["wall_s"] for r in timed)
    total_pages = sum(r["pages"] for r in timed)
    mean = lambda vals: statistics.mean(vals) if vals else None
    return {
        "tool": tool,
        "docs": n,
        "success_rate": len(ok) / n if n else None,
        "ned": mean([r["ned"] for r in scored]),
        "cer": mean([r["cer"] for r in scored if r.get("cer") is not None]),
        "teds": mean(all_teds),
        "teds_s": mean(all_teds_s),
        "gt_tables_scored": len(all_teds),
        "reading_order": mean(
            [r["reading_order"] for r in scored if r.get("reading_order") is not None]
        ),
        "s_per_page": (total_wall / total_pages) if total_pages else None,
        "failures": n - len(ok),
    }


PAPER_COLS = [
    ("tool", "Tool"),
    ("ned", "NED ↓"),
    ("cer", "CER ↓"),
    ("teds", "TEDS ↑"),
    ("teds_s", "TEDS-S ↑"),
    ("reading_order", "Read. order ↓"),
    ("s_per_page", "s/page ↓"),
    ("success_rate", "Success ↑"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--tools", default=",".join(TOOLS_DEFAULT))
    args = ap.parse_args()

    on_kaggle = Path("/kaggle").exists()
    data_dir = args.data_dir or Path("/kaggle/tmp/ohr_data" if on_kaggle else "data")
    results_dir = args.results_dir or Path("/kaggle/working/results" if on_kaggle else "results")
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]

    gt = load_gt(data_dir)
    if not gt:
        raise SystemExit(f"No ground truth found under {data_dir}/gt/law")
    print(f"Ground truth: {len(gt)} docs, {sum(len(p) for p in gt.values())} pages")

    per_doc, summaries = [], []
    for tool in tools:
        rows, all_teds, all_teds_s = evaluate_tool(tool, results_dir, gt)
        if not rows:
            print(f"[{tool}] no results found, skipping")
            continue
        per_doc += rows
        s = summarize(tool, rows, all_teds, all_teds_s)
        summaries.append(s)
        print(
            f"[{tool}] {s['docs']} docs, success {s['success_rate']:.0%}, "
            f"NED {s['ned']:.4f}" if s["ned"] is not None else f"[{tool}] no scored docs"
        )

    if not summaries:
        raise SystemExit("Nothing to evaluate.")

    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_doc).to_csv(results_dir / "results_per_doc.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(results_dir / "results_summary.csv", index=False)

    paper = summary_df[[c for c, _ in PAPER_COLS]].rename(columns=dict(PAPER_COLS))
    md_table = paper.to_markdown(index=False, floatfmt=".4f")
    (results_dir / "results.md").write_text(md_table + "\n", encoding="utf-8")
    print("\n" + md_table)
    print(f"\nWrote {results_dir}/results_per_doc.csv, results_summary.csv, results.md")
    print("Note: quality metrics are computed over successful docs only; "
          "TEDS covers docs whose GT contains tables (see gt_tables_scored).")


if __name__ == "__main__":
    main()
