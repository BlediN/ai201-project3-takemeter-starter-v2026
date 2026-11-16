#!/usr/bin/env python3
"""
Run one trial of the five capstone criteria (criteria.md) against the hosted
TakeMeter and save the real output.

    python run_criteria.py https://takemeter.onrender.com --label before --trial 1

Restart the Render service before each trial (Manual Deploy -> Restart
service): criterion 5 needs a separate boot per trial.

Order within a trial:  boot check (5) -> 1 -> 2 -> 3 -> 4 -> 5
  C1  check_bad_inputs()     15 bad inputs: no label, no 5xx/timeout, all logged
  C2  check_same_answers()   229 labels.csv posts: hosted vs local PyTorch
  C3  check_warm_latency()   10 warm single posts: median server ms, max round trip
  C4  check_big_batch()      50 x 222-token posts in one request: server ms
  C5  check_memory()         peak_rss_mib at startup and after the C4 batch

Crash handling (the harness, not the system under test):
  - /logs is snapshotted after every step and merged, so records survive a
    restart that wipes the instance's disk mid-trial.
  - Before C3 and C4 the script waits for /health and records any outage
    (seconds down, new startup records). A crash is charged to the step that
    caused it, and the later criteria still measure what they're about.

Writes:
  results_capstone_<label>_trial<N>.json        every number and response
  logs/capstone_<label>_trial<N>_server.jsonl   the server's /logs at the end
                                                (Render wipes its disk on sleep,
                                                so this copy is the run record)
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TARGETS = {
    "c1_cases": 15,
    "c2_posts": 229,
    "c3_median_server_ms": 250.0,
    "c3_max_round_trip_ms": 1000.0,
    "c4_server_ms": 30_000.0,
    "c5_peak_mib": 480.0,
}

WARM_POST = ("Jokic averaged 29.6 points, 12.7 rebounds and 10.2 assists last season "
             "on 58 percent shooting. That is not a hot take, that is the box score.")
LONG_POST = ("Jokic averaged 29.6 points on 58 percent shooting last season and led "
             "the league in assists per game. ") * 10          # 222 tokens


# ── HTTP ─────────────────────────────────────────────────────────────────────

def http(base, path, body=None, timeout=60):
    """
    (status, parsed-or-raw body, round_trip_ms). status None = no response
    (timeout / connection error); the error text goes in the body.
    `body` may be bytes (sent as-is) or any JSON value.
    """
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except Exception as e:  # timeout, reset, DNS
        return None, f"{type(e).__name__}: {e}", (time.perf_counter() - t0) * 1000
    ms = (time.perf_counter() - t0) * 1000
    try:
        return status, json.loads(raw), ms
    except ValueError:
        return status, raw.decode("utf-8", "replace"), ms


def fetch_logs(base):
    status, body, _ = http(base, "/logs?n=500", timeout=60)
    if status != 200:
        return []
    text = body if isinstance(body, str) else json.dumps(body)
    out = []
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def wait_healthy(base, limit_s=300):
    """
    Block until /health is ok. Returns (health body, seconds down): 0 if the
    first check was already healthy, else the time until it recovered.
    """
    t0 = time.time()
    first = True
    while time.time() - t0 < limit_s:
        status, body, ms = http(base, "/health", timeout=120)
        if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
            return body, (0.0 if first else round(time.time() - t0, 1))
        first = False
        time.sleep(5)
    sys.exit("service never became healthy")


class LogKeeper:
    """Merges /logs snapshots so a mid-trial restart doesn't lose records."""

    def __init__(self, base):
        self.base, self.records, self.seen = base, [], set()

    def snap(self):
        for r in fetch_logs(self.base):
            key = json.dumps(r, sort_keys=True)
            if key not in self.seen:
                self.seen.add(key)
                self.records.append(r)
        return self.records

    def startups(self):
        return [r for r in self.records if r.get("event") == "startup"]


# ── C1 ───────────────────────────────────────────────────────────────────────

def bad_cases():
    words = ("analysis hot take reaction box score rebounds assists ") * 200
    over_10k = (words * 2)[:10_001]
    assert len(over_10k.strip()) == 10_001
    big_body = json.dumps({"text": "a " * 550_000}).encode("utf-8")   # ~1.1 MB
    assert len(big_body) > 1_000_000
    return [
        # (id, description, body, level)
        ("1", "body not JSON", b"this is not json", "request"),
        ("2", "JSON with neither text nor posts", {"post": "wrong key"}, "request"),
        ("3", "posts not a list", {"posts": "Knicks in 6"}, "request"),
        ("4", "posts an empty list", {"posts": []}, "request"),
        ("5", "51 posts", {"posts": ["Knicks in 6"] * 51}, "request"),
        ("6", "body over 1 MB", big_body, "request"),
        ("7", "empty string", {"text": ""}, "post"),
        ("8", "whitespace only", {"text": "   \n\t"}, "post"),
        ("9", "emoji only", {"text": "\U0001F525\U0001F525\U0001F525"}, "post"),
        ("10", "link only", {"text": "https://streamable.com/abc123"}, "post"),
        ("11", "bare score", {"text": "23-9"}, "post"),
        ("12", "10,001 chars of words", {"text": over_10k}, "post"),
        ("13", "a number", {"text": 42}, "post"),
        ("14", "null", {"text": None}, "post"),
        ("15", "zero-width spaces only", {"text": "​​"}, "post"),
    ]


def check_bad_inputs(base, keeper):
    cases = []
    for cid, desc, body, level in bad_cases():
        status, resp, ms = http(base, "/classify", body, timeout=60)
        rid = resp.get("request_id") if isinstance(resp, dict) else None
        results = resp.get("results", []) if isinstance(resp, dict) else []
        labelled = any(r.get("label") is not None for r in results)
        if level == "request":
            handled = status in (400, 413) and not labelled
        else:
            handled = (status == 200 and len(results) == 1
                       and results[0].get("status") == "rejected" and not labelled)
        cases.append({"id": cid, "case": desc, "level": level, "http_status": status,
                      "request_id": rid, "round_trip_ms": round(ms, 1),
                      "response": resp if not isinstance(resp, dict) else
                      {k: v for k, v in resp.items()}, "handled": handled})

    logs = keeper.snap()
    for c in cases:
        recs = [r for r in logs if c["request_id"] and r.get("run_id") == c["request_id"]]
        has_request = any(r["event"] == "request" for r in recs)
        has_post = any(r["event"] == "post" for r in recs)
        c["logged"] = has_request and (has_post or c["level"] == "request")
        c["log_records"] = recs

    handled = sum(c["handled"] for c in cases)
    logged = sum(c["logged"] for c in cases)
    return {"handled": handled, "logged": logged, "n": len(cases),
            "met": handled == logged == TARGETS["c1_cases"], "cases": cases}


# ── C2 ───────────────────────────────────────────────────────────────────────

_LOCAL = None


def local_reference():
    """Labels from the notebook's PyTorch model + independent token counts."""
    global _LOCAL
    if _LOCAL is None:
        import pandas as pd
        import classify
        from transformers import AutoTokenizer
        posts = pd.read_csv("labels.csv")["text"].tolist()
        m = classify.load_model("takemeter_model")
        tok = AutoTokenizer.from_pretrained("takemeter_model")
        _LOCAL = [{"text": t, "label": classify.run_post(m, t)["label"],
                   "tokens": len(tok(t)["input_ids"])} for t in posts]
    return _LOCAL


def check_same_answers(base, keeper):
    ref = local_reference()
    hosted, failures = [], []
    for i in range(0, len(ref), 50):
        chunk = [r["text"] for r in ref[i:i + 50]]
        status, resp, ms = http(base, "/classify", {"posts": chunk}, timeout=300)
        keeper.snap()
        if status != 200:
            failures.append({"batch_start": i, "status": status, "response": resp})
            hosted += [None] * len(chunk)
        else:
            hosted += resp["results"]

    same_label = same_trunc = 0
    mismatches = []
    for i, (r, h) in enumerate(zip(ref, hosted)):
        want_trunc = r["tokens"] > 128
        ok_label = h is not None and h.get("label") == r["label"]
        ok_trunc = h is not None and h.get("truncated") == want_trunc
        same_label += ok_label
        same_trunc += ok_trunc
        if not (ok_label and ok_trunc):
            mismatches.append({"row": i + 1, "text": r["text"][:120],
                               "local_label": r["label"], "local_tokens": r["tokens"],
                               "hosted": h})
    n = len(ref)
    return {"same_label": same_label, "truncated_correct": same_trunc, "n": n,
            "n_over_128": sum(r["tokens"] > 128 for r in ref),
            "met": same_label == same_trunc == n == TARGETS["c2_posts"],
            "batch_failures": failures, "mismatches": mismatches}


# ── C3 ───────────────────────────────────────────────────────────────────────

def check_warm_latency(base):
    http(base, "/classify", {"text": WARM_POST})          # warm-up, untimed
    runs = []
    for _ in range(10):
        status, resp, ms = http(base, "/classify", {"text": WARM_POST})
        server = resp["results"][0]["latency_ms"] if status == 200 else None
        runs.append({"http_status": status, "request_id": resp.get("request_id")
                     if isinstance(resp, dict) else None,
                     "server_ms": server, "round_trip_ms": round(ms, 1)})
    server = [r["server_ms"] for r in runs if r["server_ms"] is not None]
    rtt = [r["round_trip_ms"] for r in runs]
    median_server = statistics.median(server) if len(server) == 10 else None
    return {"median_server_ms": median_server, "max_round_trip_ms": max(rtt),
            "mean_server_ms": round(statistics.mean(server), 1) if server else None,
            "met": (median_server is not None
                    and median_server <= TARGETS["c3_median_server_ms"]
                    and max(rtt) <= TARGETS["c3_max_round_trip_ms"]),
            "runs": runs}


# ── C4 ───────────────────────────────────────────────────────────────────────

def check_big_batch(base, keeper):
    status, resp, ms = http(base, "/classify", {"posts": [LONG_POST] * 50}, timeout=300)
    rid = resp.get("request_id") if isinstance(resp, dict) else None
    results = resp.get("results", []) if isinstance(resp, dict) else []
    labelled = sum(r.get("status") == "ok" for r in results)
    rec = next((r for r in keeper.snap()
                if r.get("run_id") == rid and r["event"] == "request"), None)
    server_ms = rec["latency_ms"] if rec else None
    return {"http_status": status, "labelled": labelled, "request_id": rid,
            "server_ms": server_ms, "round_trip_ms": round(ms, 1),
            "all_truncated": all(r.get("truncated") for r in results) if results else None,
            "request_record": rec,
            "met": (status == 200 and labelled == 50 and server_ms is not None
                    and server_ms <= TARGETS["c4_server_ms"])}


# ── C5 ───────────────────────────────────────────────────────────────────────

def check_memory(startup, batch_record):
    at_start = startup.get("peak_rss_mib") if startup else None
    after = batch_record.get("peak_rss_mib") if batch_record else None
    return {"startup_peak_mib": at_start, "after_batch_peak_mib": after,
            "startup_record": startup,
            "met": (at_start is not None and after is not None
                    and max(at_start, after) <= TARGETS["c5_peak_mib"])}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--label", required=True, choices=["before", "after"])
    p.add_argument("--trial", required=True, type=int, choices=[1, 2, 3])
    a = p.parse_args()
    base = a.url.rstrip("/")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    health, _ = wait_healthy(base)
    keeper = LogKeeper(base)
    keeper.snap()
    startups = keeper.startups()
    startup = startups[-1] if startups else None
    print(f"trial {a.trial} ({a.label})  boot: uptime {health['uptime_s']}s, "
          f"startup record {'found' if startup else 'MISSING'}")

    outages = []

    def recover(before_step):
        n_start = len(keeper.startups())
        h, waited = wait_healthy(base)
        keeper.snap()
        new = keeper.startups()[n_start:]
        if waited > 0 or new:
            outages.append({"before": before_step, "waited_s": waited,
                            "uptime_s_after": h["uptime_s"],
                            "new_startup_records": new})
            print(f"   outage before {before_step}: waited {waited}s, "
                  f"uptime now {h['uptime_s']}s, {len(new)} new startup record(s)")

    c1 = check_bad_inputs(base, keeper)
    print(f"C1 bad inputs   handled {c1['handled']}/15  logged {c1['logged']}/15")
    c2 = check_same_answers(base, keeper)
    print(f"C2 same answers labels {c2['same_label']}/{c2['n']}  "
          f"truncated {c2['truncated_correct']}/{c2['n']}  ({c2['n_over_128']} over 128)")
    recover("C3")
    c3 = check_warm_latency(base)
    print(f"C3 warm latency median server {c3['median_server_ms']} ms  "
          f"max round trip {c3['max_round_trip_ms']} ms")
    recover("C4")
    c4 = check_big_batch(base, keeper)
    print(f"C4 50-post batch status {c4['http_status']}  labelled {c4['labelled']}/50  "
          f"server {c4['server_ms']} ms")
    c5 = check_memory(startup, c4["request_record"])
    print(f"C5 memory       startup peak {c5['startup_peak_mib']} MiB  "
          f"after batch {c5['after_batch_peak_mib']} MiB")

    recover("end")
    out = {"label": a.label, "trial": a.trial, "url": base, "started_utc": started,
           "health_at_start": health, "targets": TARGETS, "outages": outages,
           "restarts_during_trial": len(keeper.startups()) - len(startups),
           "harness": "v2 (log snapshots + recovery waits)",
           "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
           "verdicts": {k: v["met"] for k, v in
                        {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}.items()}}
    path = Path(f"results_capstone_{a.label}_trial{a.trial}.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    Path("logs").mkdir(exist_ok=True)
    logpath = Path(f"logs/capstone_{a.label}_trial{a.trial}_server.jsonl")
    logpath.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                               for r in keeper.snap()), encoding="utf-8")
    print(f"\nverdicts {out['verdicts']}\nwrote {path} and {logpath}")


if __name__ == "__main__":
    main()
