#!/usr/bin/env python3
"""Local tests (no GPU, no parser installs): metric unit tests + fake-parser
end-to-end test of the orchestrator (checkpointing, resume, watchdog, failures).

Run:  python tests/run_tests.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluate import (  # noqa: E402
    cer,
    extract_tables,
    ned,
    norm_text,
    reading_order_distance,
)
from teds import teds  # noqa: E402

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"FAIL: {name} {detail}"
    PASS += 1
    print(f"  ok: {name}")


def test_text_metrics():
    print("text metrics")
    check("ned identical", ned("abc", "abc") == 0.0)
    check("ned 1/3", abs(ned("axc", "abc") - 1 / 3) < 1e-9)
    check("ned empty pred", ned("", "abc") == 1.0)
    check("cer insertion>1", cer("abcdef", "abc") == 1.0)  # 3 insertions / len 3
    check("norm collapses ws", norm_text("a\n\n  b\tc") == "a b c")
    check("norm drops images", "png" not in norm_text("x ![img](a.png) y"))


def test_reading_order():
    segs = [f"This is sentence number {i} with enough characters to be a segment." for i in range(8)]
    gt = " ".join(segs)
    check("ro identity", reading_order_distance(gt, gt) == 0.0)
    rev = " ".join(reversed(segs))
    ro_rev = reading_order_distance(gt, rev)
    check("ro reversed high", ro_rev is not None and ro_rev > 0.5, f"got {ro_rev}")
    check("ro unmatchable -> None", reading_order_distance(gt, "zzz completely different") is None)


def test_tables():
    print("tables")
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"
    check("teds identical", teds(html, html) == 1.0)
    check("teds structure-only identical",
          teds("<table><tr><td>x</td><td>y</td></tr><tr><td>3</td><td>4</td></tr></table>",
               html, structure_only=True) == 1.0)
    worse = "<table><tr><td>a</td></tr></table>"
    score = teds(worse, html)
    check("teds degraded", 0 <= score < 1, f"got {score}")
    check("teds empty pred", teds("", html) == 0.0)

    md = "| a | b |\n|---|---|\n| 1 | 2 |"
    tabs = extract_tables(md)
    check("md table extracted", len(tabs) == 1)
    check("md->html teds vs html gt", teds(tabs[0], html) > 0.9, f"got {teds(tabs[0], html)}")

    latex = r"\begin{tabular}{cc} a & b \\ 1 & 2 \end{tabular}"
    ltabs = extract_tables(latex)
    check("latex table extracted", len(ltabs) == 1)
    check("latex->html teds", teds(ltabs[0], html) > 0.9)

    both = md + "\n\ntext\n" + html
    check("mixed extraction", len(extract_tables(both)) == 2)


def run_bench(tmp, extra):
    return subprocess.run(
        [sys.executable, str(REPO / "run_benchmark.py"),
         "--data-dir", str(tmp / "data"), "--results-dir", str(tmp / "results"),
         "--envs-dir", str(tmp / "envs"), "--tools", "fake"] + extra,
        capture_output=True, text=True, timeout=300,
    )


def test_fake_e2e():
    print("fake-parser end-to-end (checkpoint/resume/watchdog)")
    tmp = Path(tempfile.mkdtemp(prefix="benchtest_"))
    try:
        pdf_dir = tmp / "data" / "pdfs" / "law"
        gt_dir = tmp / "data" / "gt" / "law"
        pdf_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        for stem in ["doc_a", "doc_b", "fail_doc", "hang_doc"]:
            (pdf_dir / f"{stem}.pdf").write_bytes(b"%PDF-fake" + stem.encode())
            (gt_dir / f"{stem}.json").write_text(json.dumps(
                [{"page_idx": 0, "text": f"fake parsed content of {stem}."}]))

        r = run_bench(tmp, ["--timeout-per-doc", "8", "--max-workers", "1"])
        check("orchestrator exit 0", r.returncode == 0, r.stderr[-500:])
        meta = tmp / "results" / "fake" / "meta"
        statuses = {p.stem: json.loads(p.read_text())["status"] for p in meta.glob("*.json")}
        check("all docs have meta", len(statuses) == 4, str(statuses))
        check("success recorded", statuses["doc_a"] == "success")
        check("crash recorded", statuses["fail_doc"] == "failed")
        check("watchdog timeout recorded", statuses["hang_doc"] == "timeout", str(statuses))
        check("markdown written", (tmp / "results" / "fake" / "markdown" / "doc_a.md").exists())

        # resume: nothing left pending -> no reprocessing (mtimes unchanged)
        mtime_before = (meta / "doc_a.json").stat().st_mtime_ns
        r2 = run_bench(tmp, ["--timeout-per-doc", "8"])
        check("resume exit 0", r2.returncode == 0, r2.stderr[-500:])
        check("resume skips done docs",
              (meta / "doc_a.json").stat().st_mtime_ns == mtime_before)
        check("resume reports nothing pending", "nothing pending" in r2.stdout, r2.stdout[-500:])

        # restore-from: results copied into a fresh working dir count as done
        moved = tmp / "old_results"
        shutil.move(tmp / "results", moved)
        r3 = run_bench(tmp, ["--timeout-per-doc", "8", "--restore-from", str(moved)])
        check("restore-from exit 0", r3.returncode == 0, r3.stderr[-500:])
        check("restore-from skips done docs", "nothing pending" in r3.stdout, r3.stdout[-500:])

        # evaluate the fake results end-to-end
        r4 = subprocess.run(
            [sys.executable, str(REPO / "evaluate.py"),
             "--data-dir", str(tmp / "data"), "--results-dir", str(tmp / "results"),
             "--tools", "fake"],
            capture_output=True, text=True, timeout=120,
        )
        check("evaluate exit 0", r4.returncode == 0, r4.stderr[-800:])
        import csv
        with open(tmp / "results" / "results_summary.csv") as f:
            row = next(csv.DictReader(f))
        check("success rate 2/4", abs(float(row["success_rate"]) - 0.5) < 1e-9, str(row))
        check("perfect NED on fake docs", float(row["ned"]) < 0.01, str(row))
        check("results.md written", (tmp / "results" / "results.md").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mid_run_kill_resume():
    print("mid-run kill + resume")
    tmp = Path(tempfile.mkdtemp(prefix="benchtest_"))
    try:
        pdf_dir = tmp / "data" / "pdfs" / "law"
        pdf_dir.mkdir(parents=True)
        (tmp / "data" / "gt" / "law").mkdir(parents=True)
        for i in range(30):
            (pdf_dir / f"doc_{i:02d}.pdf").write_bytes(b"%PDF-fake" + bytes([i]))

        proc = subprocess.Popen(
            [sys.executable, str(REPO / "run_benchmark.py"),
             "--data-dir", str(tmp / "data"), "--results-dir", str(tmp / "results"),
             "--envs-dir", str(tmp / "envs"), "--tools", "fake"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        meta = tmp / "results" / "fake" / "meta"
        deadline = time.time() + 60
        while time.time() < deadline and len(list(meta.glob("*.json"))) < 3:
            time.sleep(0.3)
        proc.kill()
        proc.wait()
        done_before = len(list(meta.glob("*.json")))
        check("some docs done before kill", 3 <= done_before < 30, str(done_before))

        r = run_bench(tmp, [])
        check("resume-after-kill exit 0", r.returncode == 0, r.stderr[-500:])
        check("all docs done after resume", len(list(meta.glob("*.json"))) == 30)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _make_text_pdf(path: Path, n_pages=3):
    import fitz

    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Hello searchable text on page {i}. " * 5)
    doc.save(path)
    doc.close()


def test_make_scanned():
    print("make_scanned (rasterize + verify + resume)")
    import fitz

    tmp = Path(tempfile.mkdtemp(prefix="benchtest_"))
    try:
        src_dir, out_dir = tmp / "src", tmp / "out"
        src_dir.mkdir()
        _make_text_pdf(src_dir / "doc_a.pdf", 3)
        _make_text_pdf(src_dir / "doc_b.pdf", 2)
        (out_dir / "x").mkdir(parents=True)  # ensure out dir may pre-exist
        (out_dir / "x").rmdir()
        (out_dir / "stale.pdf.tmp").write_bytes(b"partial")  # must be cleaned up

        cmd = [sys.executable, str(REPO / "make_scanned.py"),
               "--src-dir", str(src_dir), "--out-dir", str(out_dir), "--dpi", "72"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        check("make_scanned exit 0", r.returncode == 0, r.stderr[-500:])
        check("tmp file cleaned", not (out_dir / "stale.pdf.tmp").exists())
        for name, n in [("doc_a.pdf", 3), ("doc_b.pdf", 2)]:
            with fitz.open(out_dir / name) as d:
                check(f"{name} page count", len(d) == n)
                check(f"{name} no text layer",
                      all(len(p.get_text().strip()) == 0 for p in d))

        mtime = (out_dir / "doc_a.pdf").stat().st_mtime_ns
        r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        check("make_scanned resume exit 0", r2.returncode == 0, r2.stderr[-500:])
        check("resume skips existing", (out_dir / "doc_a.pdf").stat().st_mtime_ns == mtime)
        check("resume reports skipped", "2 skipped" in r2.stdout, r2.stdout[-300:])

        from evaluate import is_scanned_pdf
        check("text pdf -> digital", not is_scanned_pdf(src_dir / "doc_a.pdf"))
        check("rasterized pdf -> scanned", is_scanned_pdf(out_dir / "doc_a.pdf"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scanned_condition():
    print("--condition scanned (fake e2e)")
    tmp = Path(tempfile.mkdtemp(prefix="benchtest_"))
    try:
        for sub in ("law", "law_scanned"):
            d = tmp / "data" / "pdfs" / sub
            d.mkdir(parents=True)
            (d / "doc_x.pdf").write_bytes(b"%PDF-fake " + sub.encode())
        (tmp / "data" / "gt" / "law").mkdir(parents=True)

        def run(extra):
            return subprocess.run(
                [sys.executable, str(REPO / "run_benchmark.py"),
                 "--data-dir", str(tmp / "data"), "--envs-dir", str(tmp / "envs"),
                 "--tools", "fake"] + extra,
                capture_output=True, text=True, timeout=120, cwd=str(tmp),
            )

        r = run(["--condition", "scanned"])
        check("scanned run exit 0", r.returncode == 0, r.stderr[-500:])
        check("scanned results dir used",
              (tmp / "results_scanned" / "fake" / "meta" / "doc_x.json").exists())
        check("digital results untouched", not (tmp / "results").exists())

        shutil.rmtree(tmp / "data" / "pdfs" / "law_scanned")
        (tmp / "data" / "pdfs" / "law_scanned").mkdir()
        r2 = run(["--condition", "scanned", "--results-dir", str(tmp / "r2")])
        check("scanned guard errors when set missing",
              r2.returncode != 0 and "make_scanned" in (r2.stdout + r2.stderr))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_fake_result(results_dir: Path, tool: str, stem: str, text: str):
    d = results_dir / tool
    for sub in ("meta", "pages", "markdown"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "markdown" / f"{stem}.md").write_text(text)
    (d / "pages" / f"{stem}.json").write_text(
        json.dumps({"paginated": True, "pages": {"0": text}}))
    (d / "meta" / f"{stem}.json").write_text(json.dumps(
        {"doc": stem, "tool": tool, "status": "success", "wall_s": 1.0, "pages": 1}))


def test_compare_conditions():
    print("compare_conditions (3 slices + ranking flag)")
    tmp = Path(tempfile.mkdtemp(prefix="benchtest_"))
    try:
        gt_dir = tmp / "data" / "gt" / "law"
        gt_dir.mkdir(parents=True)
        (tmp / "data" / "pdfs" / "law").mkdir(parents=True)
        texts = {"doc_1": "Alpha bravo charlie delta echo foxtrot golf hotel india juliett.",
                 "doc_2": "Kilo lima mike november oscar papa quebec romeo sierra tango."}
        for stem, text in texts.items():
            (gt_dir / f"{stem}.json").write_text(
                json.dumps([{"page_idx": 0, "text": text}]))

        res, res_sc = tmp / "results", tmp / "results_scanned"
        for stem, text in texts.items():
            _write_fake_result(res, "toolA", stem, text)          # digital: both perfect
            _write_fake_result(res, "toolB", stem, text)
            _write_fake_result(res_sc, "toolA", stem, "zz " * 10)  # scanned: A garbled
            _write_fake_result(res_sc, "toolB", stem, text)        # scanned: B perfect
        res.mkdir(exist_ok=True)
        (res / "digital_docs.txt").write_text("doc_1\n")
        (res / "scanned_docs.txt").write_text("doc_2\n")

        r = subprocess.run(
            [sys.executable, str(REPO / "compare_conditions.py"),
             "--data-dir", str(tmp / "data"), "--results-dir", str(res),
             "--scanned-results-dir", str(res_sc), "--tools", "toolA,toolB"],
            capture_output=True, text=True, timeout=120,
        )
        check("compare exit 0", r.returncode == 0, r.stderr[-800:])
        md = (res / "comparison.md").read_text()
        for cond in ("digital-born", "natively-scanned", "simulated-scanned"):
            check(f"slice present: {cond}", cond in md)
        check("ranking difference flagged", "DIFFERS" in md, md[-600:])
        import csv
        with open(res / "comparison.csv") as f:
            rows = list(csv.DictReader(f))
        check("6 condition x tool rows", len(rows) == 6, str(len(rows)))
        by_cond_tool = {(r["condition"], r["tool"]): r for r in rows}
        check("digital slice filtered to n=1",
              by_cond_tool[("digital-born", "toolA")]["docs"] == "1")
        check("scanned toolA garbled NED high",
              float(by_cond_tool[("simulated-scanned", "toolA")]["ned"]) > 0.5)

        # evaluate.py CLI: --filter-docs + --out-prefix
        r2 = subprocess.run(
            [sys.executable, str(REPO / "evaluate.py"),
             "--data-dir", str(tmp / "data"), "--results-dir", str(res),
             "--tools", "toolA", "--filter-docs", str(res / "digital_docs.txt"),
             "--out-prefix", "results_digital"],
            capture_output=True, text=True, timeout=120,
        )
        check("evaluate --filter-docs exit 0", r2.returncode == 0, r2.stderr[-500:])
        check("out-prefix respected", (res / "results_digital.md").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_text_metrics()
    test_reading_order()
    test_tables()
    test_fake_e2e()
    test_mid_run_kill_resume()
    test_make_scanned()
    test_scanned_condition()
    test_compare_conditions()
    print(f"\nALL {PASS} CHECKS PASSED")
