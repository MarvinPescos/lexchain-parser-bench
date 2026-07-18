#!/usr/bin/env bash
# Kaggle environment setup for the LexChain parser benchmark.
#
# 1. Creates THREE ISOLATED venvs (uv) -- required, not optional:
#    mineru needs pillow>=11 while marker-pdf needs Pillow<11 (verified on PyPI,
#    2026-07), plus divergent torch/transformers/surya pins. One env per tool.
#    A shared uv cache hardlinks common wheels (torch), so disk cost stays sane.
# 2. Downloads OHR-Bench data from HuggingFace and extracts the LAW domain only:
#    pdfs.zip -> data/pdfs/law/*.pdf ; retrieval.zip -> data/gt/law/*.json
# 3. Pre-downloads each tool's model weights so parse timing excludes downloads.
#
# Idempotent: safe to re-run; finished steps are skipped.
set -euo pipefail

if [ -d /kaggle ]; then TMP_ROOT=/kaggle/tmp; else TMP_ROOT="$(pwd)/.bench_tmp"; fi
ENVS_DIR="${BENCH_ENVS_DIR:-$TMP_ROOT/envs}"
DATA_DIR="${BENCH_DATA_DIR:-$TMP_ROOT/ohr_data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TMP_ROOT/uv-cache}"
export HF_HOME="${HF_HOME:-$TMP_ROOT/hf}"
mkdir -p "$ENVS_DIR" "$DATA_DIR" "$UV_CACHE_DIR" "$HF_HOME"
echo "== envs: $ENVS_DIR | data: $DATA_DIR | hf cache: $HF_HOME"

command -v uv >/dev/null 2>&1 || pip install -q uv

create_env() {  # create_env <name> <pip-spec...>
  local name="$1"; shift
  local env="$ENVS_DIR/$name"
  if [ -f "$env/.done" ]; then echo "== env $name: already set up"; return; fi
  echo "== env $name: installing $*"
  rm -rf "$env"
  uv venv "$env" --python 3.11 >/dev/null
  uv pip install --python "$env/bin/python" -q "$@"
  touch "$env/.done"
}

create_env docling docling
create_env mineru "mineru[core]"
create_env marker marker-pdf

echo "== sanity checks"
"$ENVS_DIR/docling/bin/python" - <<'EOF'
from importlib.metadata import version
import docling.document_converter  # noqa: F401
print("  docling", version("docling"), "OK")
EOF
"$ENVS_DIR/mineru/bin/python" - <<'EOF'
from importlib.metadata import version
import mineru  # noqa: F401
print("  mineru", version("mineru"), "OK")
EOF
"$ENVS_DIR/marker/bin/python" - <<'EOF'
from importlib.metadata import version
import marker  # noqa: F401
print("  marker-pdf", version("marker-pdf"), "OK")
EOF

echo "== eval dependencies (base python)"
pip install -q -r "$(dirname "$0")/requirements-eval.txt" huggingface_hub

echo "== OHR-Bench data (Law domain)"
DATA_DIR="$DATA_DIR" python3 - <<'EOF'
import os, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

data_dir = Path(os.environ["DATA_DIR"])
pdf_dir = data_dir / "pdfs" / "law"
gt_dir = data_dir / "gt" / "law"

def extract_law(zip_name, want_suffix, need_parts, dest):
    if dest.exists() and any(dest.iterdir()):
        print(f"  {dest} already populated, skipping {zip_name}")
        return
    print(f"  downloading {zip_name} ...")
    zpath = hf_hub_download("opendatalab/OHR-Bench", zip_name, repo_type="dataset")
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            if member.endswith(want_suffix) and all(p in parts for p in need_parts):
                target = dest / Path(member).name
                with zf.open(member) as src, open(target, "wb") as out:
                    out.write(src.read())
                n += 1
    print(f"  extracted {n} files -> {dest}")

extract_law("pdfs.zip", ".pdf", ["law"], pdf_dir)
extract_law("retrieval.zip", ".json", ["gt", "law"], gt_dir)

pdfs = {p.stem for p in pdf_dir.glob("*.pdf")}
gts = {p.stem for p in gt_dir.glob("*.json")}
print(f"  law PDFs: {len(pdfs)}, GT files: {len(gts)}, matched: {len(pdfs & gts)}")
if pdfs != gts:
    print(f"  WARNING pdf-only: {sorted(pdfs - gts)[:5]} gt-only: {sorted(gts - pdfs)[:5]}")
assert pdfs & gts, "no matching pdf/gt pairs -- zip layout changed? inspect the zips"
EOF

echo "== pre-downloading model weights (best effort; tools also download on first use)"
"$ENVS_DIR/docling/bin/docling-tools" models download \
  || echo "  WARN: docling model prefetch failed (first doc will include download time)"
"$ENVS_DIR/mineru/bin/mineru-models-download" -s huggingface -m pipeline \
  || echo "  WARN: mineru model prefetch failed (first doc will include download time)"
"$ENVS_DIR/marker/bin/python" -c \
  "from marker.models import create_model_dict; create_model_dict(); print('  marker models OK')" \
  || echo "  WARN: marker model prefetch failed (first doc will include download time)"

echo "== setup complete"
