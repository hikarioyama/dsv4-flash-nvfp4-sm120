# Long context on DeepSeek-V4-Flash: 3K → 1M tokens with two config flags

**Single-request context window: `3,072` → `1,048,576` (the model's full YaRN
ceiling). KV pool: `17,300` → `1,925,540` tokens. Zero changes to weights,
quantization, output quality, or decode/MTP speed — two serve flags.**

Measured on 2× RTX PRO 6000 (SM120), TP=2, NVFP4 weights, fp8 KV, MTP on.
The numbers below are read straight from the live vLLM `GPU KV cache size`
log line and `/metrics`.

> This is **not** SM120-specific and **not** an NVFP4 thing. The root cause is a
> generic vLLM KV-cache admission heuristic interacting with DeepSeek-V4's
> sparse-attention state cache. Any DeepSeek-V4-Flash deployment on vLLM is
> affected, on any GPU. The serve scripts here happen to be the SM120/B12X ones.
>
> Only `serve_b12x_tp2.sh` reads the `MNBT` / `MAXLEN` env vars. On any other
> launcher (including this repo's `serve.sh` MARLIN path) pass
> `--max-num-batched-tokens 512 --max-model-len <N>` to vLLM directly — the
> *insight* transfers; the exact byte numbers above are for this 2× RTX PRO 6000
> / B12X / fp8-KV setup.

---

## TL;DR

```bash
# before: single 3,072 ctx, KV pool 17,300 tokens
UTIL=0.93 MAXLEN=3072  MNBT=4096 SPEC=1 ./serve_b12x_tp2.sh

# after: single 1,048,576 ctx, KV pool 1,925,540 tokens
#        same KV budget; decode/MTP unchanged, long-prompt prefill slower
UTIL=0.93 MAXLEN=1048576 MNBT=512 SPEC=1 ./serve_b12x_tp2.sh
```

The win is **lowering `--max-num-batched-tokens`** (4096 → 512) and **raising
`--max-model-len`** (3072 → up to 1048576). The first collapses a ~256×
over-allocation; the second removes an artificial cap.

---

## The symptom

Default serving gave a **17,300-token** KV pool and a **3,072-token
single-request cap** — the pool is shared across ~5.6 concurrent streams, and
`--max-model-len=3072` hard-caps any single request at 3,072 tokens. On 2×96 GB
GPUs holding a 13B-active model, that felt wrong: the KV cache was clearly being
wasted, but `--kv-cache-dtype fp8` was already on and the whole KV is already fp8.

## What it is **not** (refuted, so you don't chase them)

- **Not bf16 KV.** The entire KV — SWA, both compressor caches, indexer — is
  already `fp8_ds_mla` (584 B/token = 448 NoPE fp8 + 128 RoPE bf16 + 8 scale).
  There is no bf16 KV anywhere to convert. *(A prior hypothesis blamed "3 dense
  layers holding bf16 full-MHA KV = 88%". Wrong on every count: those layers are
  SWA-only and already fp8.)*
- **Not NVFP4 KV.** Quantizing the KV further (fp8 → nvfp4) touches only the
  ~9% fp8-KV slice of per-request memory. Byte-exact projection: ×1.03 pool at
  3K, ×1.12 even at 128K. Weeks of kernel work for ~5% — it cannot touch the
  real cost. **Dead lever.**
- **Not VRAM.** The KV pool is ~7–8 GiB before and after. The tokens it holds
  changed, not the byte budget.
- **Not the model.** DeepSeek-V4-Flash declares its caches correctly.

## Root cause: the fp32 `CompressorStateCache`, over-billed by `max_num_batched_tokens`

DeepSeek-V4's sparse attention keeps a per-layer **`CompressorStateCache`**
(`vllm/models/deepseek_v4/compressor.py`, `assert dtype == torch.float32`).
There is one per compressor — the C4 and C128 main-attention compressors plus the
C4-layer indexers (~62 for this checkpoint's `compress_ratios`). They are **fp32**
and `state_dim = 2 * coff * 512`, so they are *heavy* per token.

Their **true** sliding window is tiny — `8` tokens (C4) / `128` (C128). But
vLLM's generic sliding-window admission formula sizes every paged sliding-window
cache by:

```
max_admission_blocks_per_request =
    cdiv( min(sliding_window - 1 + max_num_batched_tokens, max_model_len),
          block_size ) + 1
```

With `max_num_batched_tokens = 4096 ≫ window = 8`, each C4 state layer is billed
for **769 fp32 pages** (`cdiv(min(8-1+4096, 3072), 4)+1`, capped here by
`max_model_len`) when it semantically needs ~3 (`cdiv(8,4)+1`). That is a **~256×**
over-allocation, and it dominates: the fp32 state caches are **~89–91% of
per-request KV memory**. The compressor *does* transiently need
`max_num_batched_tokens` slots during a chunked-prefill step (its `save_partial_states`
kernel writes the whole in-flight chunk before compressing — capping admission to
the true window deadlocks, vLLM issue #39734), so the formula is **conservative,
not buggy**. It just over-provisions badly when `mnbt` is large and the window is
tiny.

`max_model_len = 3072` was a *separate* artificial cap from a prior session that
believed KV was the bottleneck.

## The wall formula (byte-exact; reproduces the live numbers)

```
pool_tokens            = num_blocks * max_model_len / num_block_per_request
num_blocks             = AVAIL // pool_bytes_per_block          # mnbt-independent
pool_bytes_per_block   = Σ(page_size × slots) = 1,039,680 B     # after cross-group slot sharing
num_block_per_request  = cdiv( per_request_bytes(mml, mnbt) / pool_bytes_per_block )
  per_request_bytes  ≈  A(const: SWA + fp32 state, mnbt-bound)  +  B × mml (compressed KV, linear)
```

At the default (`AVAIL≈6.97 GiB`, `mml=3072`, `mnbt=4096`):
`num_blocks=7203`, `num_block_per_request=1279`, `max_concurrency=7203/1279=5.63×`,
`pool = 5.63 × 3072 = 17,300` — exact match to live `/metrics`.

The big constant term `A` (the fp32 state cache) is why short `mml` is doubly
bad: you pay the full over-allocation *and* you amortize it over only 3072
tokens. Raising `mml` amortizes `A` away; lowering `mnbt` shrinks `A` itself.

## The fix

| flag | from → to | effect |
|---|---|---|
| `--max-num-batched-tokens` | 4096 → **512** | collapses the fp32 state over-allocation (the 89–91%). Floor: must be ≥ `max_num_seqs` (these scripts hardcode `--max-num-seqs 64` in `b12x_inner.sh`, so `MNBT ≥ 64`; to go lower, lower `--max-num-seqs` too). |
| `--max-model-len` | 3072 → **up to 1048576** | removes the artificial cap; amortizes the constant overhead. |

**Tradeoff (honest):** lowering `mnbt` makes prompt **prefill** run in smaller
chunks (more steps, slower prefill of very long prompts). **Decode and MTP are
unaffected** — they are `num_seqs`-bound, not `mnbt`-bound — so the +38% MTP
speed and ~150 t/s single-stream are untouched. A prompt longer than `mnbt` is
fine; chunked prefill splits it.

## Measured results (live, 2× RTX PRO 6000, fp8 KV, MTP on)

| `max_model_len` | `mnbt` | KV pool (tokens) | concurrency | single ctx | bytes/token |
|---|---|---|---|---|---|
| 3,072 *(default)* | 4096 | 17,300 | 5.63× | 3,072 | ~434 KiB |
| 32,768 | 512 | 721,699 | 22.0× | 32,768 | ~11.6 KiB |
| 131,072 | 512 | 1,403,251 | 10.7× | 131,072 | ~5.9 KiB |
| **1,048,576** | 512 | **1,925,540** | **1.84×** | **1,048,576** | **~4.25 KiB** |

Roughly 7–8 GiB of physical KV in every row (the `Available KV cache` log line
varies 7.16 → 7.97 GiB as the over-allocation overhead amortizes; the
bytes/token column is computed from each row's own figure). The tokens it holds
change, not the ~constant byte budget. "bytes/token" is what the eliminated
over-allocation was costing: ~434 KiB/token of mostly-empty fp32 reservation at
the default (7.16 GiB / 17,300), vs ~4.25 KiB/token — the real amortized fp8
cost — at 1M (7.8 GiB / 1,925,540).

## Single 1M context — the pool fits it (capacity-verified, not load-tested)

At `mml=1048576` the pool is **1,925,540 tokens** with concurrency **1.84×** — the
pool is sized for ~1.84 simultaneous 1M-token contexts, so a single 1M request
would use ~54% of pool capacity (not "barely fits"). **Verified:** the server
boots to `health 200` and generates coherently at `max_model_len=1048576` on
*short* prompts. **Not yet load-tested:** an actual ~1M-token prefill end-to-end —
prompt-length latency, the b12x prefill workspace under a real long prompt, and
numerical quality at full YaRN (factor 16) are unmeasured.

VRAM headroom at 1M is tight (~0.75 GiB free on the display-co-resident GPU0),
because the b12x prefill workspace scales with `mml`. That workspace is reserved
at startup and per-step activation is `mnbt`-bounded, so single / low-concurrency
1M serving is *expected* to hold; the tightness blocks raising
`--gpu-memory-utilization` further and high-concurrency at extreme context.

## Recommended configs

```bash
# Max single context (the headline): single 1M, pool ~1.93M tokens.
UTIL=0.93 MAXLEN=1048576 MNBT=512 SPEC=1 ./serve_b12x_tp2.sh

# Balanced default: single 256K (covers any realistic prompt), ~2 GiB VRAM
# headroom left for raising util / high concurrency.
UTIL=0.94 MAXLEN=262144 MNBT=512 SPEC=1 ./serve_b12x_tp2.sh
```

Pool capacity rises with `mml` (overhead amortizes) but VRAM headroom falls (the
prefill workspace grows). The knee is ~256K–512K; beyond it you trade a lot of
headroom for little extra pool. Pick `mml` a bit above your largest real prompt.

## Is this vLLM's fault or the model's?

Split it cleanly:

- **The sizing formula is vLLM's** (`kv_cache_interface.py`) — and it is the
  *generic* sliding-window formula, not DeepSeek-specific. It is correct but
  conservative: it bills every request at the prefill worst case.
- **Why it's dramatic here is the model's** fp32 `CompressorStateCache` (tiny
  window, heavy fp32 state).
- **The transient need is real and framework-agnostic** — the compressor writes
  `mnbt` tokens during prefill (issue #39734), so you cannot just shrink the
  cache. Switching frameworks won't auto-fix it; it depends on their KV manager.

So: model = innocent; vLLM's generic sizing = conservative; the **default
`mnbt=4096` + `mml=3072`** = mis-tuned for long context. The fix is tuning, with
a real (small) prefill-throughput tradeoff.

## Reproduce

```bash
# read the pool the binary actually allocates, for any setting:
UTIL=0.93 MAXLEN=1048576 MNBT=512 SPEC=1 ./serve_b12x_tp2.sh
docker logs dsv4b12x 2>&1 | grep -E "GPU KV cache size|Maximum concurrency"
# -> GPU KV cache size: 1,925,540 tokens
# -> Maximum concurrency for 1,048,576 tokens per request: 1.84x
```
