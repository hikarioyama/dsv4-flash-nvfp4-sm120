# DeepSeek-V4-Flash (NVFP4) on SM120 — RTX PRO 6000 serving kit

A minimal, working kit to serve **DeepSeek-V4-Flash (NVFP4)** on
**2× RTX PRO 6000 (SM120 / consumer Blackwell)** with vLLM.

The model and its custom vLLM build target datacenter Blackwell (SM100 / B200).
Getting it to run on consumer Blackwell (SM120) means stepping on a series of
landmines. This repo is the result of clearing them — plus an honest, fully
documented account of the one we **haven't** cleared yet: **Multi-Token
Prediction (MTP) speculative decoding produces a numerically broken draft on
SM120.** See [`docs/MTP_INVESTIGATION.md`](docs/MTP_INVESTIGATION.md).

> This is published as a development log, warts and all. The MTP path is known
> broken on SM120; `serve.sh` (no MTP) works and is what you should run.

## Status

| Path | State |
|---|---|
| `serve.sh` — DSv4-Flash, no MTP | ✅ works (decode ~64 t/s single, KV pool 307k tokens) |
| `serve-mtp.sh` — MTP speculative decoding | ⚠️ boots & generates, but **draft is broken** (~0% acceptance). Diagnosis ongoing — see investigation doc |

## Prerequisites

- 2× RTX PRO 6000 (SM120, compute 12.0), 96 GB VRAM each
- A DSv4-Flash vLLM docker image whose `ENTRYPOINT` is `vllm serve`
  (referred to here as `dsv4-flash-sm120:local`)
- Model weights: `nvidia/DeepSeek-V4-Flash-NVFP4` (**no REAP, 256 experts**,
  ~157 GB, MTP module bundled)
  ```bash
  hf download nvidia/DeepSeek-V4-Flash-NVFP4 --local-dir ./models/DeepSeek-V4-Flash-NVFP4
  ```

All scripts take `MODEL_DIR`, `IMAGE`, `PORT`, etc. as env vars (see the top of
each script). Defaults assume a local layout; override to match yours.

## Quickstart

```bash
MODEL_DIR=./models/DeepSeek-V4-Flash-NVFP4 ./serve.sh   # start (fp8 KV, MoE=MARLIN)
./ready.sh      # wait for ready + print KV pool (~5 min: weight load + warmup)
./smoke.sh      # smoke test (one coding prompt)
./stop.sh       # stop
```

## The 8 traps (the point of this repo)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `command not found` for the start script | background shell has no PATH | use a full path / the scripts here |
| 2 | `image ... not built` | old run script wanted an old image name | image is `dsv4-flash-sm120:local` |
| 3 | `unrecognized arguments: serve /model` | image `ENTRYPOINT` is already `vllm serve`, so `vllm serve /model` doubles it | `--entrypoint vllm` + `serve /model ...` |
| 4 | `Cannot find any model weights` | the pointed-at dir held only `config.json` | use the dir with the real safetensors |
| 5 | `Unsupported expert number: 160` | REAP variants (180B=160 / 162B=144 experts) aren't supported by the image's MoE kernel | use the **non-REAP nvidia build (256 experts)** |
| 6 | nvidia build shows only `refs` in hf-cache | weights not actually downloaded | `hf download` the full 157 GB |
| 7 | `No NvFp4 MoE backend supports the deployment configuration` | SM120 has no native NVFP4 MoE backend (TRTLLM/CUTEDSL gate on family-100) | **`VLLM_TEST_FORCE_FP8_MARLIN=1`** to force MARLIN |
| 8 | MARLIN warning | SM120 has no native FP4 compute | weight-only FP4 dequant runs (slower; see below) |

## Why these settings

- **MoE backend = MARLIN (forced)**: vLLM's auto backend selection rejects every
  NVFP4 MoE backend for the SM120 + DSv4 combination. `VLLM_TEST_FORCE_FP8_MARLIN=1`
  forces MARLIN. Success looks like `Using 'MARLIN' NvFp4 MoE backend` in the log.
- **KV = fp8**: this image's `CacheConfig` only supports up to fp8; NVFP4 KV is not
  wired for DSv4's sparse-MLA backend.
- **sparse-MLA env** (`VLLM_TRITON_MLA_SPARSE*`): DSv4 uses MLA + DSA (a lightning
  indexer / `compress_ratios`). Attention runs on the Triton sparse-MLA kernel,
  not FlashInfer FA2.
- **TP=2 / no EP / custom all-reduce off**: a stable config for 2× SM120 + sparse MLA.

## Known limitations

- **MARLIN is weight-only FP4 dequant** — slower decode than native NVFP4
  (B200/SM100 FP4 tensor cores). Fine for capability, a handicap for speed
  benchmarks. Native would need an SM120 NVFP4 MoE kernel.
- **NVFP4 KV cache unsupported** on the DSv4 sparse-MLA path.
- **MTP speculative decoding is broken on SM120.** It boots and the VRAM concerns
  turned out to be a config bug (now fixed — KV actually *increases* to 546k
  tokens with MTP on), but the **draft model produces numerically wrong logits**
  (~0% acceptance on trivially predictable text). Full debugging log, with every
  hypothesis ruled out and the current leading suspect (an int8-vs-uint8 dtype
  mismatch in the MTP expert weights), is in
  [`docs/MTP_INVESTIGATION.md`](docs/MTP_INVESTIGATION.md).

## Benchmarks (nvidia NVFP4 / MARLIN / fp8 KV, 2× RTX PRO 6000 SM120)

- **fp8 KV pool: 307,855 tokens** (max concurrency 2.35× @131k context).
- Throughput (512 tok/req), no MTP:

  | concurrency | aggregate | per-stream | efficiency |
  |---:|---:|---:|---:|
  | 1 | 63.8 t/s | 63.8 | — |
  | 4 | 210.1 t/s | 55.0 | 0.82 |
  | 8 | 364.0 t/s | 48.0 | 0.71 |

  Not saturated at N=8 (eff 0.71); MARLIN weight-only FP4 leaves headroom for a
  native NVFP4 path.

## Scripts

| Script | Purpose |
|---|---|
| `serve.sh` | start DSv4-Flash (no MTP) — **the working path** |
| `serve-mtp.sh` | start with MTP enabled (boots, draft broken — for investigation) |
| `ready.sh` | poll `/health`, print the KV pool size once up |
| `smoke.sh` | one coding prompt to confirm coherent output |
| `bench_throughput.sh` | concurrency sweep throughput |
| `mtp_accept.sh` | MTP acceptance-rate measurement |
| `stop.sh` | stop the container |

## License

MIT — see [`LICENSE`](LICENSE).

This repo contains only scripts and documentation. It does **not** redistribute
the model weights or the vLLM/NVIDIA source; those remain under their own
licenses. The debugging doc describes how to instrument a local install but does
not copy upstream code.
