#!/usr/bin/env python3
"""
TakeMeter as a web service: the hosted run path (capstone, Path A on Render).

    uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /           a small page with a text box, for trying it in a browser
    POST /classify   {"text": "..."}  or  {"posts": ["...", "..."]}  (max 50)
    GET  /health     200 if the model is loaded, 503 (with the reason) if not
    GET  /logs?n=50  the last n structured log records, as JSON lines

It uses the same checks and model code as classify.py (check_post, run_post,
load_model), so a post that's rejected on the command line is rejected here
for the same reason. On Render it loads the ONNX model (takemeter_onnx/),
because the PyTorch version doesn't fit in the free tier's 512 MB.

Every request leaves a record, including the ones that fail:
    event "startup"   model load time, backend, memory after loading
    event "fatal"     the model couldn't load (the server stays up and says so)
    event "post"      one per post, same fields as classify.py's log
    event "request"   one per HTTP request: path, status code, latency,
                      error if any, memory now and at peak
Records go to logs/server_log.jsonl AND to stdout, so they also show up in
Render's log viewer. Render's disk is wiped when the free instance sleeps, so
copy what you need from /logs (or the dashboard) the same session.
"""

import json
import os
import platform
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

import classify

MODEL_DIR = os.environ.get("TAKEMETER_MODEL", "takemeter_onnx")
LOG_PATH = os.environ.get("TAKEMETER_LOG", os.path.join("logs", "server_log.jsonl"))
# If set, /logs needs ?token=<this>. Logged inputs are whatever people typed.
LOGS_TOKEN = os.environ.get("LOGS_TOKEN")

MAX_POSTS_PER_REQUEST = 50
MAX_BODY_BYTES = 1_000_000


# ── Memory ───────────────────────────────────────────────────────────────────

def memory_mib():
    """
    (current, peak) resident memory in MiB, the unit of Render's "512Mi" limit.
    Linux only; (None, None) elsewhere.
    """
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
        with open("/proc/self/statm") as f:
            current = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        return round(current, 1), round(peak, 1)
    except (ImportError, OSError, ValueError, AttributeError):
        return None, None


# ── Log ──────────────────────────────────────────────────────────────────────

class ServerLog:
    """JSON lines to a file and to stdout. Thread-safe. Never raises."""

    def __init__(self, path):
        self.path, self.lock, self.file = path, threading.Lock(), None
        try:
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            self.file = open(path, "a", encoding="utf-8")
        except OSError as e:
            print(f"warning: can't write log {path} ({e}); logging to stdout only",
                  file=sys.stderr, flush=True)

    def write(self, event, run_id=None, **fields):
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "run_id": run_id, "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            print(line, flush=True)
            if self.file is not None:
                try:
                    self.file.write(line + "\n")
                    self.file.flush()
                except OSError:
                    self.file = None

    def tail(self, n):
        try:
            with open(self.path, encoding="utf-8") as f:
                return f.readlines()[-n:]
        except OSError:
            return []


log = ServerLog(LOG_PATH)


# ── Model, loaded once at startup ────────────────────────────────────────────

MODEL, MODEL_ERROR = None, None
STARTED = time.time()

_t = time.perf_counter()
try:
    MODEL = classify.load_model(MODEL_DIR)
    rss, peak = memory_mib()
    log.write("startup", model=MODEL_DIR, backend=MODEL["backend"],
              runtime=MODEL["runtime"], labels=MODEL["labels"],
              model_load_ms=round((time.perf_counter() - _t) * 1000, 1),
              rss_mib=rss, peak_rss_mib=peak, python=platform.python_version())
except classify.FatalError as e:
    MODEL_ERROR = str(e)
    rss, peak = memory_mib()
    log.write("fatal", model=MODEL_DIR, error=MODEL_ERROR, rss_mib=rss, peak_rss_mib=peak)


app = FastAPI(title="TakeMeter", docs_url="/docs", redoc_url=None)


def _finish(request_id, path, status, t0, payload, n_posts=0, error=None):
    """Write the per-request record and return the response."""
    rss, peak = memory_mib()
    log.write("request", run_id=request_id, path=path, status_code=status,
              latency_ms=round((time.perf_counter() - t0) * 1000, 1),
              n_posts=n_posts, error=error, rss_mib=rss, peak_rss_mib=peak)
    if isinstance(payload, dict):
        payload["request_id"] = request_id
    return JSONResponse(payload, status_code=status)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/classify")
async def classify_endpoint(request: Request):
    request_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()

    def fail(status, error):
        return _finish(request_id, "/classify", status, t0, {"error": error}, error=error)

    if MODEL is None:
        return fail(503, f"model not loaded: {MODEL_ERROR}")

    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return fail(413, f"request body over {MAX_BODY_BYTES} bytes")
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return fail(400, 'body is not JSON. Send {"text": "..."} or {"posts": [...]}')
    if not isinstance(data, dict) or ("text" not in data and "posts" not in data):
        return fail(400, 'expected {"text": "..."} or {"posts": ["...", ...]}')

    if "posts" in data:
        posts = data["posts"]
        if not isinstance(posts, list):
            return fail(400, '"posts" must be a list')
        if not posts:
            return fail(400, '"posts" is empty')
        if len(posts) > MAX_POSTS_PER_REQUEST:
            return fail(413, f"{len(posts)} posts; the limit is {MAX_POSTS_PER_REQUEST} per request")
    else:
        posts = [data["text"]]

    results = []
    for i, text in enumerate(posts, 1):
        p0 = time.perf_counter()
        record = classify.run_post(MODEL, text)  # never raises
        latency_ms = round((time.perf_counter() - p0) * 1000, 1)
        logged = dict(record)
        logged["input_chars"] = len(logged["input"])
        logged["input"] = logged["input"][:classify.LOG_INPUT_CHARS]
        log.write("post", run_id=request_id, source="http", row=i,
                  latency_ms=latency_ms, **logged)
        results.append({k: record[k] for k in
                        ("status", "label", "confidence", "tokens", "truncated", "reason")}
                       | {"latency_ms": latency_ms})

    return _finish(request_id, "/classify", 200, t0,
                   {"results": results, "backend": MODEL["backend"]}, n_posts=len(posts))


@app.get("/health")
def health():
    t0 = time.perf_counter()
    rss, peak = memory_mib()
    body = {"status": "ok" if MODEL else "model_unavailable",
            "uptime_s": round(time.time() - STARTED, 1),
            "backend": MODEL["backend"] if MODEL else None,
            "rss_mib": rss, "peak_rss_mib": peak}
    if MODEL is None:
        body["error"] = MODEL_ERROR
        return _finish(uuid.uuid4().hex[:8], "/health", 503, t0, body, error=MODEL_ERROR)
    # Render polls /health constantly; logging every poll would bury real traffic.
    return JSONResponse(body)


@app.get("/logs")
def logs(n: int = 50, token: str | None = None):
    if LOGS_TOKEN and token != LOGS_TOKEN:
        return PlainTextResponse("forbidden: add ?token=...\n", status_code=403)
    n = max(1, min(n, 500))
    return PlainTextResponse("".join(log.tail(n)), media_type="application/x-ndjson")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TakeMeter</title>
<style>
 :root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --accent:#c8102e; }
 @media (prefers-color-scheme: dark) { :root { --bg:#141414; --fg:#eee; --muted:#999; --line:#333; } }
 body { background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif;
        max-width:680px; margin:0 auto; padding:24px 16px; }
 textarea { width:100%; box-sizing:border-box; min-height:120px; font:inherit; padding:8px;
            background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px; }
 button { margin-top:8px; padding:8px 18px; font:inherit; border:0; border-radius:6px;
          background:var(--accent); color:#fff; cursor:pointer; }
 #out { margin-top:20px; white-space:pre-wrap; font-family:ui-monospace, monospace; font-size:14px; }
 .muted { color:var(--muted); font-size:14px; }
</style></head><body>
<h1>TakeMeter</h1>
<p class="muted">Labels an r/nba comment as <b>analysis</b> (a claim backed by a checkable fact),
<b>hot_take</b> (a confident claim with nothing to check) or <b>reaction</b> (about a game in
progress or just finished). Fine-tuned DistilBERT, test accuracy 0.60, so expect it to be wrong
fairly often.</p>
<textarea id="t" placeholder="Paste a comment…"></textarea>
<button id="go">Classify</button>
<div id="out"></div>
<p class="muted">API: <code>POST /classify</code> with <code>{"text": "..."}</code> ·
<a href="/health">/health</a> · <a href="/docs">/docs</a></p>
<script>
document.getElementById("go").onclick = async () => {
  const out = document.getElementById("out");
  out.textContent = "classifying… (the first request after a quiet spell can take a minute)";
  try {
    const r = await fetch("/classify", {method: "POST", headers: {"Content-Type": "application/json"},
                                         body: JSON.stringify({text: document.getElementById("t").value})});
    const j = await r.json();
    if (!r.ok) { out.textContent = "error " + r.status + ": " + j.error; return; }
    const p = j.results[0];
    out.textContent = p.status === "ok"
      ? p.label + "  (confidence " + p.confidence.toFixed(2) + ")" + (p.truncated ? "\\nnote: only the first 128 tokens were read" : "")
      : "rejected: " + p.reason;
  } catch (e) { out.textContent = "request failed: " + e; }
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
