# DeepSeek-V4-Flash (NVFP4) on SM120 — RTX PRO 6000 serving kit

A minimal, working kit to serve **DeepSeek-V4-Flash (NVFP4)** on
**2× RTX PRO 6000 (SM120 / consumer Blackwell)** with vLLM — **including working
Multi-Token Prediction (MTP) speculative decoding**, which gives a measured
**+38% single-stream decode** on this hardware.

The model and its custom vLLM build target datacenter Blackwell (SM100 / B200).
Getting it to run on consumer Blackwell (SM120) means stepping on a series of
landmines. This repo clears them — and documents the hardest one in full: MTP
speculative decoding, which produced a numerically broken draft until the root
cause (a quantization-format mismatch in the MTP draft weights) was found and
fixed. See [`docs/MTP_INVESTIGATION.md`](docs/MTP_INVESTIGATION.md) for the
complete debugging log (Acts 1–7).

> Published as a development log, warts and all — including the four dead ends
> before the fix. Two serving paths are provided: a MARLIN path (`serve.sh`,
> no MTP) and a B12X native path (`serve_b12x_tp2.sh`, **MTP working**).

## Status

| Path | State |
|---|---|
| `serve_b12x_tp2.sh SPEC=1` — B12X native, **MTP on** | ✅ **works** — single-stream **108.8 → 150.6 t/s (+38%)**, draft acceptance 58–94% |
| `serve_b12x_tp2.sh SPEC=0` — B12X native, MTP off | ✅ works — baseline 108.8 t/s single-stream |
| `serve.sh` — MARLIN path, no MTP | ✅ works (decode ~64 t/s single, KV pool 307k tokens) |
| `serve-mtp.sh` — MARLIN path + MTP | ⚠️ draft broken on this path (~0% acceptance); the fix was landed on the **B12X** path, not MARLIN — use `serve_b12x_tp2.sh` |

## MTP on SM120 — solved (the headline)

**Symptom:** with MTP enabled the server boots and generates, but the draft is
rejected ~100% of the time (0% acceptance even on "repeat this sentence 30
times", where a working draft scores ~100%). The +7–13% you can measure is just
the broken draft occasionally landing trivial high-frequency tokens — not real.

**Root cause (proven):** in `nvidia/DeepSeek-V4-Flash-NVFP4`, the **main** experts
are **NVFP4** (block scale `float8_e4m3fn`, group_size 16, per-tensor
`weight_scale_2` + `input_scale`), but the **MTP draft** experts are **MXFP4**
(block scale `float8_e8m0fnu`, group_size 32, **no** global scales). At init the
draft expert prefix is `model.layers.{>=num_hidden_layers}.ffn.experts`, which
does **not** match the `"mtp.*"` entry in `quantization_config.ignore`, so the
loader routes the MXFP4 draft weights through the **NVFP4** method. That path
`copy_`s the e8m0 scale into an e4m3 param (a *numeric* conversion that destroys
the exponent bytes) and leaves the absent `weight_scale_2`/`input_scale` as
uninitialized garbage → the draft MoE dequantizes to noise → 0% acceptance.

**Fix (two non-invasive mount-overlay patches):**

- [`patches/quant_config.py`](patches/quant_config.py) — route experts whose
  layer index `>= num_hidden_layers` (the MTP draft) to `Mxfp4MoEMethod`
  (e8m0 scale, group_size 32, no global-sf — matching the checkpoint) instead of
  the NVFP4 method.
- [`patches/nvfp4.py`](patches/nvfp4.py) — widen the oracle clamp gate so the
  B12X backend applies `swiglu_limit` for `SILU` (this checkpoint is
  `hidden_act=silu`, `swiglu_limit=10`).

**Requires** the B12X native SM120 vLLM build
(`voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132`); the
MARLIN path was not fixed.

**Measured A/B** (same B12X build, `SPEC=1` vs `SPEC=0`, 2× RTX PRO 6000):

| metric | MTP off | MTP on | gain |
|---|---:|---:|---:|
| single-stream decode | 108.8 t/s | **150.6 t/s** | **+38% (1.385×)** |
| concurrent ×4 aggregate | ~252 t/s | ~328 t/s | ~1.30× |
| draft acceptance | — | 58% (diverse) / 94% (repeat) | mean accept len 2.16–2.88 |

(Observed live runs land ~98–110 t/s off and ~148–152 t/s on, depending on
prompt and warm state.)

See it in your own terminal — streams a ~1000-token completion live and prints
TTFT + tok/s on the last line:

```bash
./run_mtp_on.sh     # MTP on  (instant if server already in SPEC=1)
./run_mtp_off.sh    # MTP off (same B12X build; auto-restarts to SPEC=0, ~10 min)
```

Each script detects the server's current mode and only restarts when it must, so
the off-baseline is genuinely MTP-off and not a silent apples-to-oranges.

## Prerequisites

- 2× RTX PRO 6000 (SM120, compute 12.0), 96 GB VRAM each
- For the MTP path: the B12X native vLLM image above. For the MARLIN path: a
  DSv4-Flash image whose `ENTRYPOINT` is `vllm serve`.
- Model weights: `nvidia/DeepSeek-V4-Flash-NVFP4` (**no REAP, 256 experts**,
  ~157 GB, MTP module bundled)
  ```bash
  hf download nvidia/DeepSeek-V4-Flash-NVFP4 --local-dir ./models/DeepSeek-V4-Flash-NVFP4
  ```

All scripts take `MODEL_DIR`, `IMAGE`/`IMG`, `PORT`, etc. as env vars (see the top
of each script). Defaults assume a local layout; override to match yours.

## Quickstart

MTP path (recommended — the fast one):

```bash
MODEL_DIR=./models/DeepSeek-V4-Flash-NVFP4 SPEC=1 ./serve_b12x_tp2.sh  # MTP on
# ~10 min: weight load (157 GB) + torch.compile + warmup + speculator capture
./run_mtp_on.sh    # stream a 1000-token completion, print TTFT + tok/s
./stop.sh
```

MARLIN path (no MTP, simpler image):

```bash
MODEL_DIR=./models/DeepSeek-V4-Flash-NVFP4 ./serve.sh
./ready.sh ; ./smoke.sh ; ./stop.sh
```

## The 8 traps (the MARLIN path)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `command not found` for the start script | background shell has no PATH | use a full path / the scripts here |
| 2 | `image ... not built` | old run script wanted an old image name | image is `dsv4-flash-sm120:local` |
| 3 | `unrecognized arguments: serve /model` | image `ENTRYPOINT` is already `vllm serve` | `--entrypoint vllm` + `serve /model ...` |
| 4 | `Cannot find any model weights` | the pointed-at dir held only `config.json` | use the dir with the real safetensors |
| 5 | `Unsupported expert number: 160` | REAP variants (180B=160 / 162B=144) aren't supported | use the **non-REAP nvidia build (256 experts)** |
| 6 | nvidia build shows only `refs` in hf-cache | weights not actually downloaded | `hf download` the full 157 GB |
| 7 | `No NvFp4 MoE backend supports the deployment configuration` | SM120 has no native NVFP4 MoE backend (TRTLLM/CUTEDSL gate on family-100) | **`VLLM_TEST_FORCE_FP8_MARLIN=1`** to force MARLIN |
| 8 | MARLIN warning | SM120 has no native FP4 compute | weight-only FP4 dequant runs (slower) |

## Known limitations

- **`--max-model-len 3072` on the B12X/MTP path.** B12X weights take ~80 GiB/GPU,
  leaving ~1.75 GiB for KV (~4335 tokens). This is a memory-budget limit, not an
  MTP correctness limit; acceptance and decode speed are unaffected. Long-context
  serving on the B12X path needs a separate KV-budget plan.
- **MARLIN is weight-only FP4 dequant** — slower decode than the B12X native path
  (compare ~64 t/s MARLIN vs 108.8 t/s B12X off, single-stream).
- **NVFP4 KV cache unsupported** on the DSv4 sparse-MLA path (fp8 KV only).
- **MTP is fixed on the B12X path only**, not MARLIN (`serve-mtp.sh`).

## Benchmarks

**B12X native, single-stream A/B (2× RTX PRO 6000 SM120):**

| | MTP off | MTP on |
|---|---:|---:|
| decode | 108.8 t/s | 150.6 t/s (**+38%**) |

Raw streamed runs: [`docs/bench_mtp_on.jsonl`](docs/bench_mtp_on.jsonl) /
[`docs/bench_mtp_off.jsonl`](docs/bench_mtp_off.jsonl).

**MARLIN path (no MTP), concurrency sweep:**

- fp8 KV pool: **307,855 tokens** (max concurrency 2.35× @131k context).

  | concurrency | aggregate | per-stream | efficiency |
  |---:|---:|---:|---:|
  | 1 | 63.8 t/s | 63.8 | — |
  | 4 | 210.1 t/s | 55.0 | 0.82 |
  | 8 | 364.0 t/s | 48.0 | 0.71 |

## Scripts

| Script | Purpose |
|---|---|
| `serve_b12x_tp2.sh` | **B12X native path**, `SPEC=1`=MTP on / `SPEC=0`=baseline — the fast, working MTP path |
| `b12x_inner.sh` | in-container launch command for `serve_b12x_tp2.sh` |
| `patches/quant_config.py` | route MTP draft experts to `Mxfp4MoEMethod` (the MTP fix) |
| `patches/nvfp4.py` | widen the B12X oracle clamp gate for SILU |
| `run_mtp_on.sh` / `run_mtp_off.sh` | stream a ~1000-token completion, print TTFT + tok/s; auto-ensure server mode |
| `mtp_stream.py` | the streaming client used by the two `run_*` scripts |
| `bench_mtp.py` | scripted MTP on/off throughput benchmark (single + concurrent) |
| `serve.sh` | MARLIN path, no MTP |
| `serve-mtp.sh` | MARLIN path + MTP (draft broken on this path — kept for the record) |
| `ready.sh` / `smoke.sh` / `stop.sh` | poll health+KV / smoke test / stop |
| `bench_throughput.sh` | MARLIN-path concurrency sweep |
| `mtp_accept.sh` | MTP acceptance-rate measurement from server logs |

## License

MIT — see [`LICENSE`](LICENSE).

This repo contains only scripts and documentation. It does **not** redistribute
the model weights or the vLLM/NVIDIA source; those remain under their own
licenses. The patches under `patches/` are small mount-overlay shims; the
debugging doc describes how to instrument a local install but does not copy
upstream code.
