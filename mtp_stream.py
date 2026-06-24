#!/usr/bin/env python3
"""Stream a single ~1000-token completion from the DSv4-Flash server and print
TTFT + tok/s on the final line. Used by run_mtp_on.sh / run_mtp_off.sh to
eyeball MTP on-vs-off in the terminal.

decode tok/s = (completion_tokens - 1) / (last_token_time - first_token_time)
   -> pure inter-token decode rate; this is the number MTP changes.
e2e tok/s    = completion_tokens / total_wall  (includes prefill/TTFT)
TTFT         = time from request send to first streamed token.
"""
import argparse, json, sys, time, urllib.request

DEFAULT_PROMPT = (
    "Write a detailed, multi-section technical article (aim for ~900 words) on how "
    "a modern GPU runs large language model inference. Use these sections with headers:\n"
    "1. Memory hierarchy (HBM, L2, shared memory, registers) and why bandwidth matters.\n"
    "2. Tensor cores and mixed-precision matrix multiply.\n"
    "3. Attention and the KV cache during autoregressive decode.\n"
    "4. Why decode is memory-bound while prefill is compute-bound.\n"
    "5. How quantization (FP8/NVFP4) and speculative decoding (MTP) reduce latency.\n"
    "Be concrete and thorough in each section.\n\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--tag", default="")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    a = ap.parse_args()

    body = {
        "model": a.model, "prompt": a.prompt, "max_tokens": a.max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://localhost:{a.port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    print(f"\n=== {a.tag} | streaming up to {a.max_tokens} tokens ===\n", flush=True)
    t0 = time.perf_counter()
    t_first = None
    t_last = t0
    completion_tokens = 0
    chunk_text_count = 0

    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            ch = chunk.get("choices") or []
            if ch:
                txt = ch[0].get("text", "")
                if txt:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunk_text_count += 1
                    sys.stdout.write(txt)
                    sys.stdout.flush()
            u = chunk.get("usage")
            if u and u.get("completion_tokens"):
                completion_tokens = u["completion_tokens"]

    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    if completion_tokens == 0:          # fallback if server omits usage
        completion_tokens = chunk_text_count

    ttft = t_first - t0
    decode_s = max(t_last - t_first, 1e-9)
    e2e_s = max(t_end - t0, 1e-9)
    decode_tps = (completion_tokens - 1) / decode_s if completion_tokens > 1 else 0.0
    e2e_tps = completion_tokens / e2e_s

    print("\n")
    print("=" * 78)
    print(f" {a.tag} | TTFT {ttft*1000:7.1f} ms | decode {decode_tps:6.1f} tok/s "
          f"| e2e {e2e_tps:6.1f} tok/s | {completion_tokens} tok / {e2e_s:5.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
