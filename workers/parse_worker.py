#!/usr/bin/env python3
"""Per-tool parsing worker. Executed inside the tool's own venv by run_benchmark.py.

Loads the tool's models ONCE, then processes a list of PDFs sequentially,
checkpointing after every document:
    {out}/markdown/{doc}.md    full markdown
    {out}/pages/{doc}.json     {"paginated": bool, "pages": {"0": text, ...}}
    {out}/meta/{doc}.json      status/wall_s/pages/error  (atomic write)
    {out}/load_time.json       one-time model load duration
A heartbeat file ({doc}, timestamp) is refreshed before each document so the
orchestrator can detect hangs and kill/restart this worker.

The only tool importable here is the one selected by --tool (each venv has
exactly one of docling/mineru/marker). The 'fake' tool has no ML deps and is
used for local orchestrator testing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path


def atomic_write_text(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1))


def pdf_page_count(path: str):
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return None


# ------------------------------------------------------------------ adapters
# Each adapter: load() -> engine description str; parse(pdf_path) -> (markdown, pages|None)
# pages: dict {page_idx(int, 0-based): text} or None when per-page split unavailable.


class DoclingAdapter:
    name = "docling"

    def load(self):
        from importlib.metadata import version

        from docling.document_converter import DocumentConverter

        self.converter = DocumentConverter()
        try:  # trigger model load now so parse timing excludes it
            from docling.datamodel.base_models import InputFormat

            self.converter.initialize_pipeline(InputFormat.PDF)
        except Exception:
            pass
        return f"docling {version('docling')}"

    def parse(self, pdf_path):
        doc = self.converter.convert(pdf_path).document
        md = doc.export_to_markdown()
        pages = None
        try:
            n = doc.num_pages() if callable(doc.num_pages) else int(doc.num_pages)
            pages = {p - 1: doc.export_to_markdown(page_no=p) for p in range(1, n + 1)}
        except Exception:
            pages = None
        return md, pages


class MineruAdapter:
    name = "mineru"

    def load(self):
        from importlib.metadata import version

        os.environ.setdefault("MINERU_DEVICE_MODE", "cuda")
        os.environ.setdefault("MINERU_MODEL_SOURCE", "huggingface")
        self.do_parse = None
        try:
            from mineru.cli.common import do_parse, read_fn

            self.do_parse = do_parse
            self.read_fn = read_fn
            engine = "api"
        except Exception:
            engine = "cli"  # fallback: mineru CLI per doc (reloads models, slower)
        self.workdir = Path(tempfile.mkdtemp(prefix="mineru_work_"))
        return f"mineru {version('mineru')} ({engine} mode)"

    def _run_api(self, pdf_path, name, out_dir):
        pdf_bytes = self.read_fn(Path(pdf_path))
        kwargs = dict(
            output_dir=str(out_dir),
            pdf_file_names=[name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["en"],
            backend="pipeline",
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
        )
        try:
            self.do_parse(**kwargs)
        except TypeError:  # older/newer signature: fall back to minimal call
            self.do_parse(str(out_dir), [name], [pdf_bytes], ["en"], backend="pipeline")

    def _run_cli(self, pdf_path, out_dir):
        import subprocess

        subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(out_dir), "-b", "pipeline"],
            check=True,
            capture_output=True,
            text=True,
        )

    def parse(self, pdf_path):
        name = Path(pdf_path).stem
        out_dir = self.workdir / name
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        if self.do_parse is not None:
            self._run_api(pdf_path, name, out_dir)
        else:
            self._run_cli(pdf_path, out_dir)

        md_files = list(out_dir.rglob(f"{name}.md")) or list(out_dir.rglob("*.md"))
        if not md_files:
            raise RuntimeError(f"mineru produced no markdown in {out_dir}")
        md = md_files[0].read_text(encoding="utf-8")

        pages = None
        cl_files = list(out_dir.rglob(f"{name}_content_list.json")) or list(
            out_dir.rglob("*content_list.json")
        )
        if cl_files:
            content = json.loads(cl_files[0].read_text(encoding="utf-8"))
            pages = {}
            for item in content:
                idx = item.get("page_idx")
                if idx is None:
                    continue
                parts = []
                t = item.get("type")
                if t == "table":
                    parts.append(item.get("table_body") or item.get("text") or "")
                    parts += item.get("table_caption") or []
                elif t == "equation":
                    parts.append(item.get("text") or item.get("latex") or "")
                elif t == "image":
                    parts += item.get("image_caption") or []
                else:
                    parts.append(item.get("text") or "")
                chunk = "\n".join(p for p in parts if p)
                if chunk:
                    pages[idx] = (pages.get(idx, "") + "\n\n" + chunk).strip()
            pages = pages or None
        shutil.rmtree(out_dir, ignore_errors=True)
        return md, pages


class MarkerAdapter:
    name = "marker"

    # marker's paginate_output separates pages with "{page_id}" + 48 dashes
    PAGE_SEP = re.compile(r"\n?\{(\d+)\}-{30,}\n?")

    def load(self):
        from importlib.metadata import version

        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        self._text_from_rendered = text_from_rendered
        cp = ConfigParser({"output_format": "markdown", "paginate_output": True})
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config=cp.generate_config_dict(),
            processor_list=cp.get_processors(),
            renderer=cp.get_renderer(),
        )
        return f"marker-pdf {version('marker-pdf')}"

    def parse(self, pdf_path):
        rendered = self.converter(str(pdf_path))
        md, _, _ = self._text_from_rendered(rendered)
        pages = None
        parts = self.PAGE_SEP.split(md)
        if len(parts) >= 3:  # [prefix, id, text, id, text, ...]
            pages = {}
            for i in range(1, len(parts) - 1, 2):
                pages[int(parts[i])] = parts[i + 1]
            if parts[0].strip() and pages:
                first = min(pages)
                pages[first] = parts[0] + pages[first]
        return md, pages


class FakeAdapter:
    """No-dependency adapter for local orchestrator tests.

    Filename triggers: 'hang' -> sleeps (tests the watchdog),
    'fail' -> raises, 'empty' -> empty output.
    """

    name = "fake"

    def load(self):
        time.sleep(0.2)
        return "fake 0.0"

    def parse(self, pdf_path):
        stem = Path(pdf_path).stem
        if "hang" in stem:
            time.sleep(600)
        if "fail" in stem:
            raise RuntimeError("simulated parser crash")
        if "empty" in stem:
            return "", {}
        time.sleep(0.1)
        return f"# {stem}\n\nfake parsed content of {stem}.", {0: f"fake parsed content of {stem}."}


ADAPTERS = {
    "docling": DoclingAdapter,
    "mineru": MineruAdapter,
    "marker": MarkerAdapter,
    "fake": FakeAdapter,
}


# ---------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--pdfs-json", required=True, help="JSON file: list of PDF paths")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heartbeat", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    for sub in ("markdown", "pages", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    heartbeat = Path(args.heartbeat)
    pdfs = json.loads(Path(args.pdfs_json).read_text())

    atomic_write_json(heartbeat, {"doc": "__model_load__", "ts": time.time()})
    t0 = time.time()
    adapter = ADAPTERS[args.tool]()
    engine = adapter.load()
    load_s = time.time() - t0
    if not (out / "load_time.json").exists():
        atomic_write_json(out / "load_time.json", {"model_load_s": load_s, "engine": engine})
    print(f"[{args.tool}] loaded ({engine}) in {load_s:.1f}s; {len(pdfs)} docs queued", flush=True)

    for pdf in pdfs:
        stem = Path(pdf).stem
        meta_path = out / "meta" / f"{stem}.json"
        if meta_path.exists():  # already done (or recorded as failed/timeout)
            continue
        atomic_write_json(heartbeat, {"doc": stem, "pdf": pdf, "ts": time.time()})
        meta = {"doc": stem, "tool": args.tool, "engine": engine, "ts": time.time()}
        t0 = time.time()
        try:
            md, pages = adapter.parse(pdf)
            meta["wall_s"] = round(time.time() - t0, 3)
            meta["pages"] = pdf_page_count(pdf) or (len(pages) if pages else None)
            if md and md.strip():
                atomic_write_text(out / "markdown" / f"{stem}.md", md)
                atomic_write_json(
                    out / "pages" / f"{stem}.json",
                    {
                        "paginated": pages is not None,
                        "pages": {str(k): v for k, v in pages.items()} if pages else {},
                    },
                )
                meta["status"] = "success"
                meta["paginated"] = pages is not None
            else:
                meta["status"] = "empty"
                meta["error"] = "tool returned empty output"
        except Exception:
            meta["wall_s"] = round(time.time() - t0, 3)
            meta["pages"] = pdf_page_count(pdf)
            meta["status"] = "failed"
            meta["error"] = traceback.format_exc()[-4000:]
        atomic_write_json(meta_path, meta)
        print(f"[{args.tool}] {stem}: {meta['status']} ({meta.get('wall_s', 0):.1f}s)", flush=True)

    atomic_write_json(heartbeat, {"doc": "__done__", "ts": time.time()})
    print(f"[{args.tool}] worker finished", flush=True)


if __name__ == "__main__":
    main()
