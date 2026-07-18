#!/usr/bin/env python3
"""Build the simulated-scanned condition: rasterize every Law PDF into an
image-only PDF (no text layer) so every tool is forced through its OCR path.

Each page is rendered at --dpi (default 200) with pymupdf, re-embedded as a
JPEG on a fresh page of identical dimensions, and written to
pdfs/law_scanned/<same-filename>.pdf so ground truth maps 1:1.

Resumable: existing outputs are skipped; files are written to a .tmp path and
renamed only when complete, so a killed session never leaves a half-written
PDF that would be skipped later. Ends with a verification pass: every output
must match the source page count and have (near-)zero extractable text.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import fitz  # pymupdf

VERIFY_MAX_CHARS_PER_PAGE = 5  # rasterized pages should have exactly 0


def rasterize(src: Path, dest: Path, dpi: int) -> int:
    tmp = dest.with_name(dest.name + ".tmp")
    with fitz.open(src) as doc, fitz.open() as out:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            try:
                img = pix.tobytes("jpeg", jpg_quality=85)
            except Exception:
                img = pix.tobytes("png")
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img)
        n = len(out)
        out.save(tmp, garbage=3, deflate=True)
    os.replace(tmp, dest)
    return n


def verify(src: Path, dest: Path) -> str | None:
    """Returns an error string, or None if dest is a faithful image-only copy."""
    with fitz.open(src) as sdoc, fitz.open(dest) as ddoc:
        if len(sdoc) != len(ddoc):
            return f"page count {len(ddoc)} != source {len(sdoc)}"
        for i, page in enumerate(ddoc):
            chars = len(page.get_text().strip())
            if chars > VERIFY_MAX_CHARS_PER_PAGE:
                return f"page {i} has {chars} extractable chars"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_kaggle = Path("/kaggle").exists()
    data_dir = Path("/kaggle/tmp/ohr_data" if on_kaggle else "data")
    ap.add_argument("--src-dir", type=Path, default=data_dir / "pdfs" / "law")
    ap.add_argument("--out-dir", type=Path, default=data_dir / "pdfs" / "law_scanned")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    sources = sorted(args.src_dir.glob("*.pdf"))
    if not sources:
        raise SystemExit(f"No PDFs in {args.src_dir} (run setup.sh first)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.out_dir.glob("*.tmp"):
        stale.unlink()

    converted = skipped = failed = 0
    for src in sources:
        dest = args.out_dir / src.name
        if dest.exists():
            skipped += 1
            continue
        try:
            n = rasterize(src, dest, args.dpi)
            converted += 1
            print(f"  {src.name}: {n} pages rasterized @ {args.dpi} DPI", flush=True)
        except Exception as e:
            failed += 1
            print(f"  ERROR {src.name}: {e}", file=sys.stderr, flush=True)

    print("\nVerifying outputs are image-only ...")
    bad = 0
    for src in sources:
        dest = args.out_dir / src.name
        if not dest.exists():
            continue
        err = verify(src, dest)
        if err:
            bad += 1
            dest.unlink()  # force rebuild on next run
            print(f"  BAD {dest.name}: {err} -- deleted, rerun to rebuild")
    total = list(args.out_dir.glob("*.pdf"))
    size_mb = sum(p.stat().st_size for p in total) / 1e6
    print(
        f"\n{converted} converted, {skipped} skipped (already done), {failed} failed, "
        f"{bad} failed verification\n"
        f"{len(total)}/{len(sources)} scanned PDFs ready in {args.out_dir} ({size_mb:.0f} MB)"
    )
    if failed or bad or len(total) != len(sources):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
