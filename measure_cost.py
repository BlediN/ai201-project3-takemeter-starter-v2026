#!/usr/bin/env python3
"""
Measure what one TakeMeter run costs: send the same post N times and time it.

    python measure_cost.py https://<your-service>.onrender.com
    python measure_cost.py http://127.0.0.1:8000 --runs 10

The cost of this system is compute time, not money: there's no API, and
Render's free tier is $0. So "one run" = one POST /classify with one post, and
its cost is measured two ways:

  server_ms   the post's latency_ms from the server's own log record
              (input checks + tokenizing + the ONNX forward pass)
  round_ms    wall-clock time for the whole HTTP request, seen from here
              (server_ms + network + Render's proxy)

First it sends one untimed warm-up request, because a free instance that has
been asleep takes about a minute to wake and that would swamp the average. The
warm-up's time is printed separately as the cold-start cost.

Uses only the standard library, so it runs from any Python 3.9+.
"""

import argparse
import json
import statistics
import time
import urllib.request

POST = ("Jokic averaged 29.6 points, 12.7 rebounds and 10.2 assists last season "
        "on 58 percent shooting. That is not a hot take, that is the box score.")


def call(base, text, timeout=180):
    req = urllib.request.Request(
        base.rstrip("/") + "/classify",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return (time.perf_counter() - t0) * 1000, body


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("url", help="base URL of the running server")
    p.add_argument("--runs", type=int, default=10)
    args = p.parse_args()

    cold_ms, body = call(args.url, POST)
    print(f"warm-up (cold start if the instance was asleep): {cold_ms:,.0f} ms, "
          f"request {body.get('request_id')}\n")

    rows = []
    for i in range(1, args.runs + 1):
        round_ms, body = call(args.url, POST)
        r = body["results"][0]
        rows.append((round_ms, r["latency_ms"]))
        print(f"run {i:>2}  request {body['request_id']}  {r['label']:<9} "
              f"server {r['latency_ms']:>7.1f} ms   round trip {round_ms:>7.1f} ms")

    server = [s for _, s in rows]
    rtt = [r for r, _ in rows]
    print(f"\n{args.runs} runs, one post each")
    print(f"  server_ms  mean {statistics.mean(server):7.1f}   median {statistics.median(server):7.1f}"
          f"   min {min(server):7.1f}   max {max(server):7.1f}")
    print(f"  round_ms   mean {statistics.mean(rtt):7.1f}   median {statistics.median(rtt):7.1f}"
          f"   min {min(rtt):7.1f}   max {max(rtt):7.1f}")


if __name__ == "__main__":
    main()
