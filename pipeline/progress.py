"""Build progress, derived from artifacts and logs (the pipeline writes no
progress file of its own, so this reads what each stage leaves behind):

- stage and QC attempt tallies from the newest render_*.log in the book dir
- stage 4 windows narrated vs total (04_narration.progress.jsonl vs the
  window plan over 03_chapters.json)
- stage 5 takes rendered for THIS narration (segment files newer than the
  narration script) vs segments, with pace and ETA from file mtimes
- stage 6 output and the iCloud copy

One collector, two emitters: `main.py progress [--json] [--watch]` on the
terminal and `main.py dashboard` as a small local web page.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from pipeline.config import ARTIFACTS_DIR, OUTPUT_DIR

ICLOUD = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Audiobooks"
_STAGE_NAMES = {1: "extract", 2: "structure", 3: "chapterize", 4: "narrate", 5: "render", 6: "master"}
_ATTEMPT = re.compile(r"^stage (\d) attempt (\d+): (.*)$")
_TALLY = re.compile(r"([a-z_]+) \((\d+)\)")


def _newest_log(bdir: Path) -> Path | None:
    logs = sorted(bdir.glob("render*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _parse_log(path: Path | None) -> dict:
    out = {"stage": None, "attempts": [], "halted": False, "done": False, "crashed": False,
           "last_line": "", "last_event": "", "log": None, "log_mtime": None}
    if not path or not path.exists():
        return out
    out["log"] = str(path)
    out["log_mtime"] = path.stat().st_mtime
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Only the most recent launch matters.
    starts = [i for i, l in enumerate(lines) if l.startswith("=== supervisor attempt") or l.startswith("=== restarted")
              or l.startswith("=== relaunch") or l.startswith("=== finish") or l.startswith("=== remaster")]
    tail = lines[starts[-1]:] if starts else lines
    pending = ""
    for l in tail:
        s = l.strip()
        if s.startswith("QC stage "):
            out["stage"] = int(s[9]); out["attempts"] = [a for a in out["attempts"] if a["stage"] != out["stage"]]
            out["halted"] = False
            out["last_event"] = f"stage {out['stage']} {_STAGE_NAMES.get(out['stage'], '')} started"
        m = _ATTEMPT.match(s)
        if m or (pending and s and not s.startswith(("QC", "stage", "===", "feeding"))):
            text = (pending + " " + s) if pending else s
            mm = _ATTEMPT.match(text)
            if mm:
                tally = {k: int(v) for k, v in _TALLY.findall(mm.group(3))}
                if text.rstrip().endswith(",") or mm.group(3).strip().endswith(","):
                    pending = text; continue
                out["attempts"].append({"stage": int(mm.group(1)), "attempt": int(mm.group(2)), "flags": tally,
                                        "total": sum(tally.values())})
                out["last_event"] = f"stage {mm.group(1)} attempt {mm.group(2)}: {sum(tally.values())} flags"
                pending = ""
                continue
        if s.startswith("feeding "):
            out["last_event"] = s.replace("fixable violations back to", "flags back to")
        if "checks green" in s:
            out["last_event"] = s
        if s.startswith("halted at stage"):
            out["halted"] = True; out["last_event"] = s
        if ("BUILD DONE" in s or "REMASTER DONE" in s or s.startswith("Audiobook ready")
                or s.startswith("Reassembled:") or re.match(r"=== (build|finish) exited rc=0", s)):
            out["done"] = True; out["last_event"] = "audiobook ready"
        if s.startswith("Traceback") or "crashed:" in s:
            out["crashed"] = True; out["last_event"] = s[:120]
        if s and not s.startswith("Setting `pad_token_id`") and not s.startswith("Fetching"):
            out["last_line"] = s[:160]
    return out


def _window_plan(bdir: Path) -> int:
    chs = bdir / "03_chapters.json"
    if not chs.exists():
        return 0
    try:
        from pipeline.s4_narration import _windows
        data = json.loads(chs.read_text(encoding="utf-8"))
        return sum(len(_windows(c.get("text", ""))) for c in data["chapters"]
                   if c.get("matter", "body") == "body" and not c.get("front_matter"))
    except Exception:
        return 0


def _alive(bdir: Path) -> bool:
    ex = bdir / "01_extract.json"
    needle = "main.py build"
    if ex.exists():
        try:
            pdf = json.loads(ex.read_text(encoding="utf-8")).get("pdf", "")
            if pdf:
                needle = re.escape(Path(pdf).name[:40])  # pgrep -f takes a regex; titles carry parentheses
        except Exception:
            pass
    return subprocess.run(["pgrep", "-f", needle], capture_output=True).returncode == 0


def book_progress(bdir: Path) -> dict:
    now = time.time()
    log = _parse_log(_newest_log(bdir))
    stages_on_disk = sorted({p.name[:2] for p in bdir.glob("0*") if p.name[:2].isdigit()})
    narr = bdir / "04_narration.json"
    prog = bdir / "04_narration.progress.jsonl"
    segdir = bdir / "05_render" / "segments"
    title = bdir.name.replace("_", " ").title()
    out_manifest = OUTPUT_DIR / bdir.name / "manifest.json"
    if out_manifest.exists():
        try:
            title = json.loads(out_manifest.read_text()).get("title") or title
        except Exception:
            pass

    # Stage 4 windows
    win_total = _window_plan(bdir)
    win_done = 0
    if narr.exists():
        win_done = win_total
    elif prog.exists():
        keys = set()
        for line in prog.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
        win_done = min(len(keys), win_total) if win_total else len(keys)

    # Stage 5 takes for this narration
    seg_total = takes = 0
    pace = 0
    eta_h = None
    last_take = None
    if narr.exists():
        try:
            seg_total = len(json.loads(narr.read_text(encoding="utf-8"))["segments"])
        except Exception:
            seg_total = 0
        if segdir.exists():
            # A manifest that matches this narration is authoritative (it is
            # rewritten at the end of every render pass); mid-pass, before one
            # exists, count files newer than the narration script.
            manifest = bdir / "05_render" / "manifest.json"
            counted = False
            if manifest.exists():
                try:
                    rows = json.loads(manifest.read_text(encoding="utf-8"))["segments"]
                    present = [r for r in rows if (bdir / r["wav"]).exists()]
                    if seg_total and len(rows) == seg_total and len(present) >= 0.5 * seg_total:
                        takes = len(present)
                        counted = True
                except Exception:
                    pass
            mt = [p.stat().st_mtime for p in segdir.glob("*.wav")]
            if not counted:
                t0 = narr.stat().st_mtime
                this = [m for m in mt if m >= t0 - 1]
                takes = min(len(this), seg_total) if seg_total else len(this)
            recent = sum(1 for m in mt if now - m < 1800)
            pace = recent * 2
            if mt:
                last_take = max(mt)
            if pace and seg_total > takes:
                eta_h = (seg_total - takes) / pace

    m4b = OUTPUT_DIR / bdir.name / "book.m4b"
    icloud = None
    if ICLOUD.exists() and m4b.exists():
        for p in ICLOUD.glob("*.m4b"):
            if abs(p.stat().st_size - m4b.stat().st_size) < 1024 * 1024:
                icloud = p.name
    alive = _alive(bdir)

    # Phase and overall fraction (weights: script 10%, render 85%, master 5%)
    if log["done"] or (m4b.exists() and not alive and takes >= seg_total > 0):
        phase, frac = "done", 1.0
    elif log["stage"] == 6 or (narr.exists() and seg_total and takes >= seg_total and alive):
        phase, frac = "mastering", 0.95
    elif narr.exists() and (log["stage"] == 5 or takes):
        phase, frac = "rendering", 0.10 + 0.85 * (takes / seg_total if seg_total else 0)
    elif log["stage"] == 4 or prog.exists() or (stages_on_disk and "04" not in stages_on_disk and "03" in stages_on_disk):
        phase, frac = "writing script", 0.10 * (win_done / win_total if win_total else 0)
    elif stages_on_disk:
        phase, frac = "extracting", 0.02
    else:
        phase, frac = "not started", 0.0
    if log["halted"] and not log["done"]:
        phase = "halted for review"
    elif log["crashed"] and not alive and not log["done"]:
        phase = "crashed"
    elif not alive and phase not in ("done",) and (narr.exists() or prog.exists()):
        phase = f"stopped ({phase})"

    return {
        "id": bdir.name, "title": title, "phase": phase, "fraction": round(frac, 4),
        "alive": alive, "stage": log["stage"], "stage_name": _STAGE_NAMES.get(log["stage"] or 0, ""),
        "stages_on_disk": stages_on_disk,
        "script": {"windows_done": win_done, "windows_total": win_total},
        "render": {"takes": takes, "segments": seg_total, "pace_per_hour": pace,
                   "eta_hours": round(eta_h, 2) if eta_h is not None else None,
                   "last_take": last_take},
        "qc": {"attempts": log["attempts"], "halted": log["halted"]},
        "output": {"m4b": str(m4b) if m4b.exists() else None,
                   "m4b_mtime": m4b.stat().st_mtime if m4b.exists() else None,
                   "icloud": icloud},
        "last_event": log["last_event"], "last_line": log["last_line"],
        "log": log["log"], "log_age_s": round(now - log["log_mtime"]) if log["log_mtime"] else None,
        "updated": now,
    }


def all_progress(filter_text: str | None = None) -> list[dict]:
    books = [d for d in sorted(ARTIFACTS_DIR.iterdir()) if d.is_dir() and (d / "01_extract.json").exists()]
    if filter_text:
        books = [d for d in books if filter_text.lower().replace(" ", "_") in d.name]
    rows = [book_progress(d) for d in books]
    # Active builds first, then most recently touched.
    rows.sort(key=lambda r: (not r["alive"], -(r["render"]["last_take"] or r["output"]["m4b_mtime"] or 0)))
    return rows


DASHBOARD_HTML = """<!doctype html><meta charset="utf-8"><title>pdf2audiobook builds</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light dark}body{font:14px -apple-system,system-ui,sans-serif;margin:0;padding:16px;background:#f6f6f4;color:#1a1a1a}
@media(prefers-color-scheme:dark){body{background:#141414;color:#e8e8e8}.card{background:#1e1e1e!important;border-color:#333!important}.bar{background:#333!important}.muted{color:#999!important}}
h1{font-size:16px;margin:0 0 12px}.card{background:#fff;border:1px solid #e2e2e0;border-radius:10px;padding:14px 16px;margin-bottom:12px}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:12px}.title{font-weight:600;font-size:15px}
.phase{font-size:12px;padding:2px 8px;border-radius:999px;background:#e8f0fe;color:#1a4fb4}.phase.done{background:#e6f4ea;color:#137333}
.phase.halted,.phase.crashed{background:#fce8e6;color:#c5221f}.phase.stopped{background:#fef7e0;color:#9a6700}
.bar{height:10px;border-radius:5px;background:#e6e6e3;overflow:hidden;margin:8px 0 4px}.bar>i{display:block;height:100%;background:#1a73e8}
.bar.done>i{background:#137333}.muted{color:#666;font-size:12px}table{border-collapse:collapse;font-size:12px;margin-top:8px}td,th{padding:2px 8px 2px 0;text-align:left}
</style><h1>pdf2audiobook builds <span class="muted" id="ts"></span></h1><div id="books"></div>
<script>
function fmtEta(h){if(h==null)return"";if(h<1)return Math.round(h*60)+" min";const d=new Date(Date.now()+h*3600e3);return h.toFixed(1)+" h (≈"+d.toLocaleTimeString([],{hour:"numeric",minute:"2-digit"})+(h>12?" "+d.toLocaleDateString([],{weekday:"short"}):"")+")"}
function cls(p){return p.startsWith("done")?"done":p.startsWith("halted")?"halted":p.startsWith("crashed")?"crashed":p.startsWith("stopped")?"stopped":""}
async function load(){const r=await fetch("/api");const rows=await r.json();document.getElementById("ts").textContent="updated "+new Date().toLocaleTimeString();
document.getElementById("books").innerHTML=rows.map(b=>{const pct=Math.round(b.fraction*100);const rr=b.render,sc=b.script;
let detail="";if(b.phase.includes("script"))detail=`script: ${sc.windows_done}/${sc.windows_total} windows`;
else if(rr.segments)detail=`takes ${rr.takes.toLocaleString()}/${rr.segments.toLocaleString()}`+(rr.pace_per_hour?` · ${rr.pace_per_hour}/h · ETA ${fmtEta(rr.eta_hours)}`:"");
const qc=b.qc.attempts.length?`<table><tr><th>stage</th><th>attempt</th><th>flags</th><th>by check</th></tr>`+b.qc.attempts.map(a=>`<tr><td>${a.stage}</td><td>${a.attempt}</td><td>${a.total}</td><td class="muted">${Object.entries(a.flags).map(([k,v])=>k+" "+v).join(", ")}</td></tr>`).join("")+`</table>`:"";
const out=b.output.m4b?`<div class="muted">M4B ${new Date(b.output.m4b_mtime*1000).toLocaleString()}${b.output.icloud?" · in iCloud":""}</div>`:"";
return`<div class="card"><div class="row"><span class="title">${b.title}</span><span class="phase ${cls(b.phase)}">${b.phase}${b.stage?" · stage "+b.stage+" "+b.stage_name:""}</span></div>
<div class="bar ${cls(b.phase)}"><i style="width:${pct}%"></i></div><div class="row"><span>${pct}% · ${detail}</span><span class="muted">${b.alive?"process alive":"no process"}${b.log_age_s!=null?" · log "+Math.round(b.log_age_s/60)+" min ago":""}</span></div>
<div class="muted">${b.last_event||""}</div>${qc}${out}</div>`}).join("")}
load();setInterval(load,15000);
</script>"""


def serve(port: int = 8787, host: str = "0.0.0.0") -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/api"):
                body = json.dumps(all_progress()).encode()
                ctype = "application/json"
            else:
                body = DASHBOARD_HTML.encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    HTTPServer((host, port), H).serve_forever()
