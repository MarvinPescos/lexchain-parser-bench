# lexchain-parser-bench

Benchmark of three PDF document-parsing tools — **[Docling](https://github.com/docling-project/docling)**, **[MinerU](https://github.com/opendatalab/MinerU)** (successor of magic-pdf, pipeline backend), **[Marker](https://github.com/datalab-to/marker)** — on the **Law domain of [OHR-Bench](https://github.com/opendatalab/OHR-Bench)** (95 PDFs, 1,187 pages, human-verified page-level ground truth). Built for the LexChain capstone; designed to run on a **Kaggle T4×2** notebook.

## Metrics

| Metric | Meaning | Method |
|---|---|---|
| NED ↓ | text fidelity | per-page normalized Levenshtein vs GT (OHR-Bench's OCR-quality metric) |
| CER ↓ | char error rate | Levenshtein ÷ GT length (penalizes hallucinated insertions) |
| TEDS / TEDS-S ↑ | table structure+content / structure only | tables extracted from GT & prediction, matched, scored with APTED tree edit distance (PubTabNet implementation, OmniDocBench-style) |
| Reading order ↓ | content sequence correctness | GT segments fuzzy-located in prediction; permutation edit distance vs identity (OmniDocBench-style) |
| s/page ↓ | throughput | wall time ÷ pages, model load excluded |
| Success ↑ | reliability | clean exit + non-empty markdown; crashes/timeouts/empty logged |

Note: OHR-Bench's released eval code covers the RAG stages (retrieval/generation) only; its OCR-quality edit distance is reimplemented here, and table/reading-order metrics follow OmniDocBench (same lab). State this in the paper's methodology.

## Why three venvs

`mineru` requires `pillow>=11.0.0` while `marker-pdf` requires `Pillow<11.0.0` (hard conflict, verified on PyPI 2026-07), with further divergent `torch`/`transformers`/`surya-ocr` pins. `setup.sh` creates one `uv` venv per tool; a shared wheel cache hardlinks the common heavy wheels.

## Usage (Kaggle)

Open `benchmark.ipynb` on Kaggle (GPU T4×2, internet ON), set `REPO_URL`, run:

1. **Cell 1** – clone + `setup.sh` (venvs, Law-domain data, model weights)
2. **Cell 2** – 3-document smoke test → prints s/page per tool + projected full-run wall time. **Approve before continuing.**
3. **Cell 3** – full run. Two long-lived workers (one per GPU), third tool starts when a GPU frees. Checkpoint after every doc to `/kaggle/working/results`; rerunning resumes. Watchdog kills hung docs (`--timeout-per-doc`, default 20 min).
4. **Cell 4** – `evaluate.py` → `results_summary.csv`, `results_per_doc.csv`, `results.md` (paper table).

**Session died?** Attach the previous run's output as an input dataset, then
`python run_benchmark.py --restore-from /kaggle/input/<slug>/results`.

Useful flags: `--limit N` (N smallest docs), `--docs a,b` (specific stems), `--tools docling,marker`, `--retry-failed`.

## Local development (no GPU)

The GPU tools never run locally. The orchestrator + metrics are testable on CPU:

```bash
pip install -r requirements-eval.txt
python tests/run_tests.py        # metric unit tests + fake-parser e2e (resume, watchdog)
```

## Layout

```
setup.sh                 Kaggle setup: 3 uv venvs, OHR-Bench Law data, model prefetch
run_benchmark.py         orchestrator (stdlib-only): scheduling, checkpoints, watchdog
workers/parse_worker.py  per-tool adapter, runs inside the tool's venv
evaluate.py              metrics -> CSV + markdown table
teds.py                  TEDS (adapted from IBM PubTabNet, Apache-2.0)
benchmark.ipynb          Kaggle driver notebook
tests/run_tests.py       local tests (no GPU needed)
```
