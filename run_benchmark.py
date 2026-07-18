#!/usr/bin/env python3
"""Resumable benchmark orchestrator (stdlib only; runs on the system python).

Schedules one long-lived worker per tool (workers/parse_worker.py inside that
tool's venv), pinning each worker to a GPU via CUDA_VISIBLE_DEVICES. With 2
GPUs and 3 tools, the third tool starts as soon as a GPU frees up.

Crash-safety:
- every document is checkpointed by the worker (meta/*.json written atomically);
  on restart, docs with a meta file are skipped (--retry-failed to redo failures)
- a watchdog kills a worker whose heartbeat is older than --timeout-per-doc,
  records the stuck doc as a timeout, and restarts the worker
- --restore-from copies results from a previous (dead) Kaggle session's output
  dataset into the working results dir before running

Typical use:
  python run_benchmark.py --limit 3          # smoke test (3 smallest PDFs)
  python run_benchmark.py                    # full run
  python run_benchmark.py --restore-from /kaggle/input/<prev-output>/results
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

TOOLS_DEFAULT = ["docling", "mineru", "marker"]
POLL_S = 5
MAX_RESTARTS_PER_TOOL = 8


def log(msg):
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        ).stdout
        return [str(i) for i in range(len([l for l in out.splitlines() if l.strip()]))]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def atomic_write_json(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def restore_from(src: Path, results_dir: Path):
    """Copy a previous session's results in, without clobbering newer files."""
    n = 0
    for f in src.rglob("*"):
        if not f.is_file() or f.suffix == ".tmp":
            continue
        dest = results_dir / f.relative_to(src)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        n += 1
    log(f"restored {n} files from {src}")


class ToolRun:
    def __init__(self, tool, venv_python, pdfs, results_dir, timeout_s):
        self.tool = tool
        self.venv_python = venv_python
        self.pdfs = pdfs
        self.dir = results_dir / tool
        self.timeout_s = timeout_s
        self.heartbeat = self.dir / "heartbeat.json"
        self.restarts = 0
        self.proc = None
        self.gpu = None
        self.log_file = None

    def pending(self):
        meta = self.dir / "meta"
        return [p for p in self.pdfs if not (meta / f"{Path(p).stem}.json").exists()]

    def start(self, gpu):
        pending = self.pending()
        if not pending:
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        pdfs_json = self.dir / "queue.json"
        atomic_write_json(pdfs_json, pending)
        self.heartbeat.unlink(missing_ok=True)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        self.gpu = gpu
        self.log_file = open(self.dir / "worker.log", "a")
        worker = Path(__file__).parent / "workers" / "parse_worker.py"
        self.proc = subprocess.Popen(
            [
                self.venv_python,
                str(worker),
                "--tool", self.tool,
                "--pdfs-json", str(pdfs_json),
                "--out-dir", str(self.dir),
                "--heartbeat", str(self.heartbeat),
            ],
            env=env,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log(f"{self.tool}: worker started on GPU {gpu} ({len(pending)} docs pending, "
            f"restart {self.restarts})")
        return True

    def read_heartbeat(self):
        try:
            return json.loads(self.heartbeat.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def kill(self):
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def record_incident(self, status):
        """Write a failure meta for the doc named in the heartbeat."""
        hb = self.read_heartbeat()
        if not hb or hb["doc"].startswith("__"):
            return None
        stem = hb["doc"]
        meta_path = self.dir / "meta" / f"{stem}.json"
        if meta_path.exists():
            return None
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            meta_path,
            {
                "doc": stem,
                "tool": self.tool,
                "status": status,
                "wall_s": None,
                "pages": None,
                "error": f"{status}: killed by orchestrator "
                         f"(timeout={self.timeout_s}s)" if status == "timeout"
                         else f"{status}: worker died while processing this doc",
                "ts": time.time(),
            },
        )
        return stem

    def check(self):
        """Returns 'running' | 'finished' | 'aborted'. Restarts on crash/hang."""
        exit_code = self.proc.poll()
        if exit_code is None:
            hb = self.read_heartbeat()
            if hb and hb["doc"] != "__done__":
                # model load gets 3x the doc budget (downloads, first-run compile)
                budget = self.timeout_s * (3 if hb["doc"] == "__model_load__" else 1)
                if time.time() - hb["ts"] > budget:
                    stem = self.record_incident("timeout")
                    log(f"{self.tool}: TIMEOUT on {stem or hb['doc']}, killing worker")
                    self.kill()
                    return self._restart()
            return "running"

        # process exited
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        if not self.pending():
            log(f"{self.tool}: all docs processed")
            return "finished"
        if exit_code == 0:
            # clean exit but pending remain (shouldn't happen) -> restart
            log(f"{self.tool}: worker exited cleanly with pending docs, restarting")
            return self._restart()
        stem = self.record_incident("crashed")
        log(f"{self.tool}: worker died (exit {exit_code})"
            + (f" on {stem}" if stem else " before any doc"))
        return self._restart()

    def _restart(self):
        self.restarts += 1
        if self.restarts > MAX_RESTARTS_PER_TOOL:
            log(f"{self.tool}: exceeded {MAX_RESTARTS_PER_TOOL} restarts, aborting tool "
                f"({len(self.pending())} docs left unprocessed)")
            return "aborted"
        if not self.start(self.gpu):
            return "finished"
        return "running"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    on_kaggle = Path("/kaggle").exists()
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/kaggle/tmp/ohr_data" if on_kaggle else "data"))
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/kaggle/working/results" if on_kaggle else "results"))
    ap.add_argument("--envs-dir", type=Path,
                    default=Path("/kaggle/tmp/envs" if on_kaggle else "envs"))
    ap.add_argument("--tools", default=",".join(TOOLS_DEFAULT))
    ap.add_argument("--limit", type=int, default=None,
                    help="only the N smallest PDFs (smoke test)")
    ap.add_argument("--docs", default=None, help="comma-separated doc stems to run")
    ap.add_argument("--timeout-per-doc", type=int, default=1200, metavar="SECONDS")
    ap.add_argument("--retry-failed", action="store_true",
                    help="delete failed/timeout/crashed/empty metas first, so those docs rerun")
    ap.add_argument("--restore-from", type=Path, default=None,
                    help="results dir of a previous session's output to resume from")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="parallel workers (default: number of GPUs, min 1)")
    args = ap.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.restore_from:
        restore_from(args.restore_from, args.results_dir)

    pdf_dir = args.data_dir / "pdfs" / "law"
    pdfs = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.stat().st_size)
    if args.docs:
        wanted = {d.strip() for d in args.docs.split(",")}
        pdfs = [p for p in pdfs if p.stem in wanted]
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir} (run setup.sh first)")
    pdfs = [str(p) for p in pdfs]
    log(f"{len(pdfs)} docs x {len(tools)} tools")

    if args.retry_failed:
        n = 0
        for tool in tools:
            for meta_path in (args.results_dir / tool / "meta").glob("*.json"):
                try:
                    if json.loads(meta_path.read_text()).get("status") != "success":
                        meta_path.unlink()
                        n += 1
                except (json.JSONDecodeError, OSError):
                    meta_path.unlink()
                    n += 1
        log(f"--retry-failed: cleared {n} non-success metas")

    def venv_python(tool):
        if tool == "fake":
            return sys.executable
        p = args.envs_dir / tool / "bin" / "python"
        if not p.exists():
            raise SystemExit(f"venv for {tool} not found at {p} (run setup.sh first)")
        return str(p)

    runs = {t: ToolRun(t, venv_python(t), pdfs, args.results_dir, args.timeout_per_doc)
            for t in tools}

    gpus = detect_gpus()
    n_workers = args.max_workers or max(1, len(gpus))
    n_workers = min(n_workers, len(tools))
    gpu_pool = (gpus or [None] * n_workers)[:n_workers]
    log(f"GPUs detected: {gpus or 'none'} -> {n_workers} parallel worker(s)")

    queue = [t for t in tools if runs[t].pending()]
    for t in tools:
        if t not in queue:
            log(f"{t}: nothing pending, skipping")
    active: dict[str, ToolRun] = {}
    t_start = time.time()

    try:
        while queue or active:
            while queue and len(active) < len(gpu_pool):
                used = {r.gpu for r in active.values()}
                gpu = next((g for g in gpu_pool if g not in used), None)
                tool = queue.pop(0)
                if runs[tool].start(gpu):
                    active[tool] = runs[tool]
            time.sleep(POLL_S)
            for tool in list(active):
                state = active[tool].check()
                if state in ("finished", "aborted"):
                    del active[tool]
    except KeyboardInterrupt:
        log("interrupted -- killing workers (progress is checkpointed, rerun to resume)")
        for r in active.values():
            r.kill()
        raise SystemExit(130)

    log(f"done in {(time.time() - t_start) / 60:.1f} min")
    for tool in tools:
        meta_dir = args.results_dir / tool / "meta"
        statuses = {}
        for mp in meta_dir.glob("*.json"):
            try:
                s = json.loads(mp.read_text()).get("status", "?")
            except json.JSONDecodeError:
                s = "?"
            statuses[s] = statuses.get(s, 0) + 1
        log(f"{tool}: {statuses}")


if __name__ == "__main__":
    main()
