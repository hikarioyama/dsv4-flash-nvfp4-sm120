#!/usr/bin/env python3
"""DSv4-Flash MTP on/off throughput benchmark.

Measures:
  - single-stream decode throughput (tok/s)
  - multi-stream concurrent throughput (tok/s)
Prints a JSON line per run for easy comparison.
"""
import argparse, json, time, urllib.request, urllib.error, concurrent.futures as cf

URL = "http://localhost:{port}/v1/completions"
MODEL = "DeepSeek-V4-Flash"

PROMPTS = [
    "Explain how multi-token prediction (MTP) works in large language models and why it can speed up inference. "
    "Cover the draft model concept, acceptance rate, and the tradeoff between compute overhead and latency reduction. ",
    "Write a short story about a robot learning to paint watercolors in a quiet seaside town. ",
    "Describe the architecture of a modern GPU tensor core and how it accelerates mixed-precision matrix multiply. ",
]

def post(port, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(URL.format(port=port), data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    return resp, dt

def single_bench(port, prompt, max_tokens, n_runs, label):
    results = []
    for i in range(n_runs):
        body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": False}
        resp, dt = post(port, body)
        ct = resp["usage"]["completion_tokens"]
        tps = ct / dt
        results.append({"run": i, "completion_tokens": ct, "wall_s": round(dt,3),
                        "tok_s": round(tps,1)})
        print(json.dumps({"label": label, "mode": "single", "run": i,
                          "completion_tokens": ct, "wall_s": round(dt,3),
                          "tok_s": round(tps,1)}), flush=True)
    return results

def concurrent_bench(port, n_concurrent, max_tokens, n_rounds, label):
    """Each round fires n_concurrent requests simultaneously."""
    all_tps = []
    for r in range(n_rounds):
        bodies = [{"model": MODEL, "prompt": PROMPTS[j % len(PROMPTS)] * (1 + j // 3),
                   "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
                  for j in range(n_concurrent)]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(n_concurrent) as ex:
            futs = [ex.submit(post, port, b, 300) for b in bodies]
            resps = [f.result() for f in futs]
        dt = time.perf_counter() - t0
        total_ct = sum(r[0]["usage"]["completion_tokens"] for r in resps)
        agg_tps = total_ct / dt
        all_tps.append(agg_tps)
        print(json.dumps({"label": label, "mode": "concurrent", "round": r,
                          "n_concurrent": n_concurrent, "total_tokens": total_ct,
                          "wall_s": round(dt,3), "aggregate_tok_s": round(agg_tps,1)}),
              flush=True)
    return all_tps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--single-runs", type=int, default=5)
    ap.add_argument("--concurrent", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    # warmup (trigger JIT / cudagraph capture)
    for i in range(args.warmup):
        try:
            post(args.port, {"model": MODEL, "prompt": PROMPTS[0],
                 "max_tokens": 32, "temperature": 0.0, "stream": False}, 300)
        except Exception as e:
            print(json.dumps({"label": args.label, "warmup_error": str(e)}), flush=True)
    print(json.dumps({"label": args.label, "event": "warmup_done"}), flush=True)

    single_bench(args.port, PROMPTS[0], args.max_tokens, args.single_runs, args.label)
    concurrent_bench(args.port, args.concurrent, args.max_tokens, args.rounds, args.label)
    print(json.dumps({"label": args.label, "event": "done"}), flush=True)

if __name__ == "__main__":
    main()
