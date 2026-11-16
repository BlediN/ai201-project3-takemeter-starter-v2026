#!/usr/bin/env python3
"""
Download the ONNX model during the Render build.

    MODEL_URL=https://github.com/<you>/<repo>/releases/download/<tag>/model.onnx \
        python fetch_model.py

model.onnx is 268 MB, over GitHub's 100 MB limit for a file in git, so it
lives as a GitHub Release asset and the build fetches it. Everything else the
server needs (tokenizer, config, label order) is committed in takemeter_onnx/.

The download is checked against takemeter_onnx/model.onnx.sha256 (committed),
so a truncated or wrong file fails the build here instead of failing at the
first request. Network errors are retried 3 times with a growing wait.

Exit codes: 0 model in place and verified, 1 it isn't (the build should stop).
"""

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEST = Path("takemeter_onnx") / "model.onnx"
SHA_FILE = Path("takemeter_onnx") / "model.onnx.sha256"
ATTEMPTS = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if not SHA_FILE.exists():
        print(f"error: {SHA_FILE} is missing. Run export_onnx.py and commit it.")
        return 1
    expected = SHA_FILE.read_text().split()[0].strip().lower()

    if DEST.exists() and sha256(DEST) == expected:
        print(f"{DEST} already present and verified, nothing to download")
        return 0

    url = os.environ.get("MODEL_URL")
    if not url:
        print("error: MODEL_URL is not set. Set it to the model.onnx release-asset URL "
              "(see RUNNING.md).")
        return 1

    tmp = DEST.with_suffix(".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            t0 = time.perf_counter()
            print(f"downloading {url} (attempt {attempt}/{ATTEMPTS})", flush=True)
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            got = sha256(tmp)
            if got != expected:
                raise ValueError(f"checksum mismatch: got {got[:12]}..., "
                                 f"expected {expected[:12]}...")
            tmp.replace(DEST)
            size_mb = DEST.stat().st_size / 1e6
            print(f"ok: {DEST} ({size_mb:.0f} MB) in {time.perf_counter() - t0:.1f}s, "
                  "checksum verified")
            return 0
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            if 400 <= e.code < 500:  # wrong URL or private repo: retrying won't help
                print(f"  failed: HTTP {e.code}. Check MODEL_URL, and that the "
                      "repo/release is public.")
                return 1
            print(f"  failed: HTTP {e.code}", flush=True)
            if attempt < ATTEMPTS:
                time.sleep(5 * attempt)
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"  failed: {e}", flush=True)
            tmp.unlink(missing_ok=True)
            if attempt < ATTEMPTS:
                time.sleep(5 * attempt)

    print(f"error: could not download a verified model after {ATTEMPTS} attempts")
    return 1


if __name__ == "__main__":
    sys.exit(main())
