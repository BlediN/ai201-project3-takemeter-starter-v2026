#!/usr/bin/env python3
"""
TakeMeter from the command line: label r/nba posts as analysis / hot_take /
reaction with the model the notebook saved to takemeter_model/.

    python classify.py "LeBron is washed, book it"
    python classify.py "post one" "post two"
    python classify.py --csv posts.csv            # needs a `text` column

This is the capstone (shape B, harden against failure). The happy path is a
few lines; most of this file is what happens when something goes wrong:

  Bad post (empty, no words, link only, over 10,000 characters, not text)
      -> status "rejected" with a reason. No label is forced onto it.
  Post longer than the model reads (max_length tokens)
      -> still labelled, but marked truncated=True with its token count.
  One post crashes the model
      -> status "error" for that post. The rest of the batch carries on.
  Missing or broken model folder, unreadable CSV, no `text` column, no input
      -> one-line message on stderr and exit code 2. No stack trace.

Exit codes: 0 the run finished (rejected posts are normal output, not a
failure), 2 the run could not start.

Every run appends to a structured log (default logs/classify_log.jsonl), one
JSON object per line:

  event "post"      one per post: ts, input, status, label, confidence,
                    tokens, truncated, reason, latency_ms
  event "run_end"   one per run: counts, model_load_ms, total_ms
  event "fatal"     a run that could not start, and why

If the log can't be written, the run still classifies and says so on stderr.
"""

import argparse
import csv
import json
import os
import platform
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL_DIR = "takemeter_model"
DEFAULT_LOG = os.path.join("logs", "classify_log.jsonl")

# The log keeps what came in, but not unboundedly: a 10,000-character post is
# cut to this many characters in the log (input_chars still says how long it was).
LOG_INPUT_CHARS = 2000

# Reddit caps a comment at 10,000 characters. Anything longer didn't come from
# r/nba, and tokenizing it is wasted work, so it is rejected rather than cut.
MAX_CHARS = 10_000

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
# A "word" is any run of two or more letters, in any alphabet. Digits alone
# ("23-9") don't count: no label in the taxonomy can be read off a bare score.
WORD_RE = re.compile(r"[^\W\d_]{2,}")

EXIT_OK = 0
EXIT_FATAL = 2


class FatalError(Exception):
    """Something that stops the whole run. Printed as one line, exit code 2."""


# ── Input checks ─────────────────────────────────────────────────────────────

def check_post(text):
    """
    Decide whether a post can be classified at all.

    Returns None if it can, or a short reason string if it can't.
    """
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return "missing"
    if isinstance(text, MalformedRow):
        return (f"malformed_row ({len(text.fields)} fields, "
                f"expected {text.expected})")
    if not isinstance(text, str):
        return f"not_text ({type(text).__name__})"
    stripped = text.strip()
    if not stripped:
        return "empty"
    if len(stripped) > MAX_CHARS:
        return f"too_long ({len(stripped)} chars > {MAX_CHARS})"
    without_links = URL_RE.sub(" ", stripped)
    if not WORD_RE.search(without_links):
        return "link_only" if URL_RE.search(stripped) else "no_words"
    return None


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(model_dir):
    """Load tokenizer + model + label names, or raise FatalError saying why."""
    path = Path(model_dir)
    rebuild = ("Build it by running notebook sections 1-5 and the "
               "'Save the model' cell (see RUNNING.md).")

    if not path.exists():
        raise FatalError(f"model folder not found: {path}. {rebuild}")
    if not path.is_dir():
        raise FatalError(f"model path is not a folder: {path}")

    needed = ["config.json", "model.safetensors"]
    missing = [f for f in needed if not (path / f).exists()]
    if not ((path / "tokenizer.json").exists() or (path / "vocab.txt").exists()):
        missing.append("tokenizer.json or vocab.txt")
    if missing:
        raise FatalError(f"model folder {path} is incomplete, missing: "
                         f"{', '.join(missing)}. {rebuild}")

    # Imported here so `--help` and input errors don't wait on torch.
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers.utils import logging as hf_logging
    except ImportError as e:
        raise FatalError(f"a package is missing ({e.name}). "
                         "Run: pip install -r requirements.txt")

    hf_logging.set_verbosity_error()  # no "sequence longer than 512" noise

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
    except Exception as e:
        first_line = str(e).strip().splitlines()[0] if str(e).strip() else ""
        if len(first_line) > 120:
            first_line = first_line[:120] + "…"
        raise FatalError(f"model folder {path} could not be loaded "
                         f"({type(e).__name__}: {first_line}). {rebuild}")
    model.eval()

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    max_length = 128
    meta_path = path / "takemeter_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            max_length = int(meta.get("max_length", max_length))
            if meta.get("labels") and meta["labels"] != labels:
                raise FatalError(f"label mismatch: model has {labels}, "
                                 f"takemeter_meta.json says {meta['labels']}")
        except (ValueError, TypeError) as e:
            raise FatalError(f"{meta_path} is not valid JSON ({e})")

    return {"torch": torch, "tokenizer": tokenizer, "model": model,
            "labels": labels, "max_length": max_length}


def classify_one(m, text):
    """Label one post that already passed check_post. Returns a partial record."""
    torch, tokenizer, model = m["torch"], m["tokenizer"], m["model"]

    n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
    enc = tokenizer(text, truncation=True, max_length=m["max_length"],
                    return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    best = int(probs.argmax())

    return {
        "label": m["labels"][best],
        "confidence": round(float(probs[best]), 4),
        "tokens": n_tokens,
        "truncated": n_tokens > m["max_length"],
    }


def run_post(m, text):
    """One post in, one record out. Never raises."""
    if isinstance(text, str):
        shown = text
    elif text is None or (isinstance(text, float) and text != text):
        shown = ""
    else:
        shown = repr(text)
    record = {"input": shown,
              "status": None, "label": None, "confidence": None,
              "tokens": None, "truncated": None, "reason": None}

    reason = check_post(text)
    if reason:
        record.update(status="rejected", reason=reason)
        return record

    try:
        record.update(classify_one(m, text.strip()))
        record["status"] = "ok"
    except Exception as e:  # one bad post must not kill the batch
        record.update(status="error", reason=f"{type(e).__name__}: {e}")
    return record


# ── Input sources ────────────────────────────────────────────────────────────

class MalformedRow:
    """A CSV line pandas couldn't split into the header's columns."""

    def __init__(self, fields, expected):
        self.fields, self.expected = fields, expected

    def __repr__(self):
        return ",".join(self.fields)


def read_csv_posts(csv_path):
    """
    Return the `text` column as a list, or raise FatalError saying why.

    A line with the wrong number of fields doesn't stop the run: it comes back
    as a MalformedRow at the end of the list and gets rejected like any other
    bad post, so it shows up in the output and the log instead of vanishing.
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        raise FatalError(f"CSV not found: {path}")
    bad_rows = []
    try:
        with open(path, encoding="utf-8") as f:
            n_cols = len(next(csv.reader(f), []))
        df = pd.read_csv(path, dtype={"text": object}, engine="python",
                         on_bad_lines=lambda fields: bad_rows.append(
                             MalformedRow(fields, n_cols)))
    except pd.errors.EmptyDataError:
        raise FatalError(f"CSV is empty: {path}")
    except UnicodeDecodeError:
        raise FatalError(f"CSV is not UTF-8 text: {path}. Re-save it as UTF-8.")
    except Exception as e:
        raise FatalError(f"CSV could not be read: {path} ({type(e).__name__}: {e})")

    if "text" not in df.columns:
        cols = ", ".join(map(str, df.columns)) or "(none)"
        raise FatalError(f"CSV has no `text` column: {path}. Columns found: {cols}")
    if df.empty and not bad_rows:
        raise FatalError(f"CSV has a header but no rows: {path}")
    return df["text"].tolist() + bad_rows


# ── Structured log ───────────────────────────────────────────────────────────

class RunLog:
    """
    Appends one JSON object per line to a .jsonl file.

    Each record is written and flushed as it happens, so a run that crashes
    halfway still leaves everything up to the crash on disk. A log that can't
    be written (read-only folder, path is a directory, disk full) is reported
    once on stderr and then ignored: losing the log must not lose the labels.
    """

    def __init__(self, path, run_id):
        self.path, self.run_id = path, run_id
        self.file, self.broken = None, False
        if path is None:
            return
        try:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            self.file = open(path, "a", encoding="utf-8")
        except OSError as e:
            self._fail(e)

    def _fail(self, e):
        if not self.broken:
            print(f"warning: can't write log {self.path} ({e}). "
                  "Classifying anyway, without a log.", file=sys.stderr)
        self.broken, self.file = True, None

    @property
    def active(self):
        return self.file is not None

    def write(self, event, **fields):
        if self.file is None:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "run_id": self.run_id, "event": event, **fields}
        try:
            self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.file.flush()
        except OSError as e:
            self._fail(e)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


# ── Output ───────────────────────────────────────────────────────────────────

def show(record, i, n):
    preview = record["input"].replace("\n", " ")
    preview = preview[:60] + ("…" if len(preview) > 60 else "")
    if record["status"] == "ok":
        flag = f"  [truncated: {record['tokens']} tokens]" if record["truncated"] else ""
        print(f"[{i}/{n}] {record['label']:<9} {record['confidence']:.2f}  {preview}{flag}")
    else:
        print(f"[{i}/{n}] {record['status'].upper():<9} {record['reason']}  {preview}")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Label r/nba posts as analysis / hot_take / reaction.")
    p.add_argument("posts", nargs="*", help="one or more posts, each in quotes")
    p.add_argument("--csv", help="a CSV file with a `text` column")
    p.add_argument("--model", default=DEFAULT_MODEL_DIR,
                   help=f"model folder (default: {DEFAULT_MODEL_DIR})")
    p.add_argument("--log", default=DEFAULT_LOG,
                   help=f"structured log, appended to (default: {DEFAULT_LOG})")
    p.add_argument("--no-log", action="store_true", help="don't write a log")
    return p.parse_args(argv)


def main(argv=None):
    # Windows consoles default to cp1252, which can't print emoji. A post full
    # of 🔥 should not crash the run on the way *out*.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args(argv)
    t_start = time.perf_counter()
    log = RunLog(None if args.no_log else args.log, uuid.uuid4().hex[:8])
    source = f"csv:{args.csv}" if args.csv else "cli"

    try:
        if args.csv and args.posts:
            raise FatalError("give posts on the command line OR --csv, not both")
        if args.csv:
            posts = read_csv_posts(args.csv)
        elif args.posts:
            posts = args.posts
        else:
            raise FatalError('nothing to classify. Try: python classify.py "your post"')

        t_load = time.perf_counter()
        m = load_model(args.model)
        model_load_ms = round((time.perf_counter() - t_load) * 1000, 1)
    except FatalError as e:
        print(f"error: {e}", file=sys.stderr)
        log.write("fatal", source=source, model=args.model, error=str(e),
                  total_ms=round((time.perf_counter() - t_start) * 1000, 1))
        log.close()
        return EXIT_FATAL

    counts = {"ok": 0, "rejected": 0, "error": 0}
    ok_latencies = []
    for i, text in enumerate(posts, 1):
        t0 = time.perf_counter()
        record = run_post(m, text)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        counts[record["status"]] += 1
        if record["status"] == "ok":
            ok_latencies.append(latency_ms)
        show(record, i, len(posts))

        logged = dict(record)
        logged["input_chars"] = len(logged["input"])
        logged["input"] = logged["input"][:LOG_INPUT_CHARS]
        log.write("post", source=source, row=i, latency_ms=latency_ms, **logged)

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    mean_ms = (round(sum(ok_latencies) / len(ok_latencies), 1)
               if ok_latencies else None)
    log.write("run_end", source=source, model=args.model, n_posts=len(posts),
              **counts, model_load_ms=model_load_ms, total_ms=total_ms,
              mean_latency_ms_ok=mean_ms, python=platform.python_version(),
              torch=m["torch"].__version__)
    logged_to = args.log if log.active else None
    log.close()

    print(f"\n{len(posts)} posts: {counts['ok']} labelled, "
          f"{counts['rejected']} rejected, {counts['error']} errors")
    print(f"model load {model_load_ms / 1000:.1f}s · "
          f"mean {mean_ms if mean_ms is not None else '-'} ms per labelled post · "
          f"total {total_ms / 1000:.1f}s")
    if logged_to:
        print(f"log: {logged_to}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
