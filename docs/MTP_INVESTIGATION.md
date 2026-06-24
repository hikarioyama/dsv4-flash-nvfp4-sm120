# MTP on SM120: a debugging log (still open)

This is the honest, in-progress account of trying to make **Multi-Token
Prediction (MTP / self-speculative decoding)** work for DeepSeek-V4-Flash
(NVFP4) on **2× RTX PRO 6000 (SM120, consumer Blackwell)**.

Short version: MTP **boots and runs**, but the **draft model is numerically
broken** on SM120 — it accepts ~0% of its proposals on trivially predictable
text. The "MTP eats all my VRAM" belief we started with turned out to be a
config bug (fixed, and KV cache actually *grew*). The real problem is deeper and
**not yet fully root-caused**. The current leading suspect is a concrete,
verified anomaly: the MTP block's quantized expert weights are stored as `int8`
while every main-model layer stores them as `uint8`.

Nothing here is inflated. Where a thing is proven, it says so. Where it's a
hypothesis, it says that too.

---

## Act 1 — "MTP eats VRAM" was a misdiagnosis (and the fix gave us *more* KV)

The handed-down belief was: *enabling MTP makes ~62 GB of KV cache vanish.*

What actually happens, from the logs:

1. **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** had been set to reclaim KV.
   It means "reserve **zero** memory for CUDA graphs." So the KV pool was sized
   to fill almost all of VRAM — and then **CUDA graph capture OOM'd**, *after*
   KV was already allocated. The OOM happens at capture time, so it reads like
   "the memory disappeared into KV." It didn't; it was never reserved for the
   graphs MTP needs.
2. The real memory pressure was **`max-num-batched-tokens=8192`** driving a large
   activation peak (MARLIN MoE dequant × 8192 tokens × 256 experts), on a model
   whose weights already occupy ~82% of each GPU.

**Fix** (in `serve-mtp.sh`):
- remove `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (let vLLM reserve graph memory),
- `--max-num-batched-tokens 2048` (shrinks the activation peak),
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (defragments).

**Result — with MTP enabled:**

```
Available KV cache memory: 5.14 GiB
GPU KV cache size: 546,872 tokens     (max concurrency 4.17x @131k)
```

That's **more** than the no-MTP baseline (307,855 tokens). So the VRAM story was
backwards: MTP's real overhead was a fixable activation peak, and fixing it
roughly doubled the KV pool. (The `max-num-batched-tokens` win applies to
`serve.sh` too.)

---

## Act 2 — But MTP doesn't speed anything up: the draft is broken

With it finally booting cleanly, measure acceptance:

| prompt | mean accept length | draft acceptance |
|---|---|---|
| code generation | 1.18–1.24× | ~20% |
| prose | 1.08× | ~8% |
| **"repeat this exact sentence 30 times"** | **1.00×** | **0.0%** |

A working draft head must score **near 100%** on pure repetition — the
continuation is literally the same sentence again. **0% is mathematically
impossible for a functioning draft.** So MTP isn't "weak" here; it's broken.

Decoding the draft's actual predictions confirms it:

```
draft argmax → 'anim' 'ultad' ' αρσενικό' '这三种' 'ipi' ' til' ' живело' '}package' ...
ground truth → 'The'  ' quick' ' brown'  ' fox'  ' jumps' ' over' ' the'   ' lazy' ...
```

Multilingual garbage, uncorrelated with the obvious answer. Not shifted, not
off-by-one — **uncorrelated**.

(The ~+7–13% single-stream speedup you can measure on code is just the broken
draft occasionally landing trivial high-frequency tokens. It is not real.)

---

## Act 3 — Ruling things out (a custom instrumented build, 12 reloads)

We built a throwaway instrumented vLLM image (dump draft inputs / intermediates /
predictions; see [`DEBUG_INSTRUMENTATION.md`](DEBUG_INSTRUMENTATION.md)) and
killed hypotheses one at a time. **All seven were falsified by measurement:**

| Hypothesis | How it was ruled out |
|---|---|
| Plumbing / off-by-one feeding the draft | In eager mode the draft's input residual is healthy (norm ~582, no NaN). The output is uncorrelated garbage, not a shifted version of the truth. |
| CUDA-graph capture/binding | `--enforce-eager` → still 0%. |
| Weight load failure | The loader's per-layer presence check passes; `mega_moe` is off (`moe_backend=auto`). |
| **"The kernels are SM100-only"** | The **main model uses the same MHC tilelang kernels** (`mhc_*_tilelang`, `hc_head_fused_kernel_tilelang`) and is correct. The kernel file gates explicitly on `is_device_capability_family(120)` and runs an SM120 TF32 path. SM120 is supported. |
| NaN / numeric explosion | Every stage of the draft layer is finite and sane: `h_proj+e_proj` max ~4 → `mtp_block` max ~150 → `mhc_post` max ~90, no NaN. |
| Weight post-processing skipped for the draft | The draft loads via the normal `get_model()`, so MARLIN's `process_weights_after_loading` runs. |
| DSA sparse attention | Disabling `VLLM_TRITON_MLA_SPARSE` → draft still 0% (and the main model still generates coherently). |

The shape of the failure after all this: the draft layer produces output of the
**right magnitude but the wrong direction**. Acceptance is ~20% on code but ~2%
on prose — *partial signal*, not uniform randomness (a random 128k-vocab
predictor would be ~0.001%). That signature points at **numerical degradation**,
not a structural break — i.e. **subtly wrong weights**, not wrong wiring.

---

## Act 4 — The current lead: an int8-vs-uint8 dtype mismatch

Reading the checkpoint directly (no vLLM, no GPU), comparing the MTP block's MoE
expert weights to the main layers':

```
MTP  block expert weights (w1/w3.weight)  →  torch.int8    (95 tensors)
MAIN layer expert weights (w1/w3.weight)  →  torch.uint8   (all, 256 tensors)
```

**Verified and consistent across the checkpoint: the MTP experts are `int8`;
every main-layer expert is `uint8`.** Both hold the same packed NVFP4 bytes
(values 0–255).

Why this is a strong suspect: NVFP4 packs two 4-bit (E2M1) codes per byte.
Unpacking the high nibble is `byte >> 4`. On a **signed** `int8`, `>>` is an
**arithmetic** shift — for bytes with the top bit set it sign-extends and
corrupts the high nibble; on `uint8` it's a logical shift and is correct. Any
step that treats the packed byte as signed (a shift, a widening `to(int32)`, a
compare) will dequantize the MTP weights wrong while leaving the main weights
fine.

And the loader does **not** normalize it: the custom MTP weight loader
reinterprets the **scale** tensors (`float8_e8m0fnu → uint8`) but passes the
`int8` **weight** through untouched, straight into the MARLIN dequant path that
the main weights reach as `uint8`.

So the mechanism candidate is:
> The MTP expert weights are `int8` in the checkpoint; nothing reinterprets them
> as `uint8`; the SM120-forced MARLIN dequant treats the sign bit wrong; the MTP
> MoE computes a finite-but-wrong transform; the draft predicts garbage.

**Status: this is the leading hypothesis, not yet proven causal.** The decisive
test is a one-line loader patch — reinterpret the int8 expert weight as uint8
before loading, mirroring how the scales are already handled — and re-measure
acceptance on the repetition prompt. If acceptance jumps, root cause confirmed
and MTP fixed. That test is pending.

---

## What you should do today

- Run `serve.sh` (no MTP). It works.
- Don't trust MTP speedups on SM120 until the above is resolved.
- Fastest sanity check for "is the draft alive?": ask it to **repeat one
  sentence 30 times** and read the acceptance rate. 0% = broken draft; ~100% =
  healthy.

## Open threads / next steps

1. Patch the MTP expert-weight load to `.view(torch.uint8)` and re-measure (the
   decisive test).
2. If that's not it, split `mtp_block` to dump attention-output vs MoE-output
   separately and localize within the layer.
3. Either way, this likely wants an upstream fix in the NVIDIA DSv4 build's MTP
   weight handling; the evidence here (7 ruled-out hypotheses + the dtype
   mismatch) is meant to be handed up.

## Reproduction notes / gotchas

- The active runner in this build is `vllm/v1/worker/gpu_model_runner.py` (V1),
  **not** `vllm/v1/worker/gpu/model_runner.py` (V2). Instrument the right one.
- The draft `forward` is captured inside a CUDA graph; Python instrumentation
  there won't fire on replay. Use `--enforce-eager` to see it run.
- After editing an installed `.py`, delete its `__pycache__/*.pyc` (a stale
  `.pyc` will silently win) or run with `PYTHONDONTWRITEBYTECODE=1`.

---

## Act 5 — B12X native backend test (2026-06-24): **boots, draft still 0% accepted**

A different vLLM build with a native SM120 W4A16/W4A8 MoE backend
(`voipmonitor/vllm:chthonic-...-b12x0ff2847-pr20-cu132`, `--moe-backend b12x`)
was supposed to be the fix. The upstream `local-inference-lab/rtx6kpro` doc
reports acceptance ~0.68 on this backend, and it was the strongest candidate to
confirm "MARLIN fallback is the root cause."

**What happened (proven):**

- The B12X image pulled and the server **boots to ready** on this exact hardware
  (2× RTX PRO 6000, TP2). Weights load (79.96 GiB/GPU), MTP draft loads (39
  params), all CUDA graphs (PIECEWISE + FULL) capture, `/health` returns 200.
- **But the nvidia/NVFP4 checkpoint is rejected by the B12X oracle gate** at
  init with:
  > `ValueError: Model sets swiglu_limit=10.0, but the explicitly requested
  > moe_backend='b12x' does not apply the SwiGLU clamp. Use
  > 'flashinfer_trtllm' or 'flashinfer_cutlass' instead.`
- Root cause of that gate: `nvfp4.py::_backend_supports_clamp` only allows B12X
  to clamp when `activation == SWIGLUOAI_UNINTERLEAVE`. Our checkpoint is
  `hidden_act=silu` (→ `MoEActivation.SILU`) with `swiglu_limit=10.0`, so it is
  **over-blocked**. The B12X kernel itself (`b12x_moe.py`) *does* support SILU
  (`_supports_activation` lists it) and *does* apply `swiglu_limit` for
  `quant_mode == w4a8_nvfp4` (this model) regardless of activation. So the gate
  is more conservative than the kernel.
- **A 1-line mount-overlay patch** (`patches/nvfp4.py`) widens the gate to allow
  `SILU`/`SWIGLUOAI` too. With the patch, init passes and the server starts.

**The decisive measurement (proven):**

With B12X + the clamp patch + MTP enabled, the repeat-prompt and counting tests
still report:

> `SpecDecoding metrics: ... Accepted: 0 tokens, Drafted: N,
> Per-position acceptance rate: 0.000, 0.000, Avg Draft acceptance rate: 0.0%`

The main model generates correctly (repeat prompt loops the right sentence;
counting is correct), so the **target** model is fine. The **draft** is still
proposing tokens that the target rejects 100%. This is the same 0% signature as
the MARLIN build.

**What this means:**

- B12X native backend, by itself, does **not** fix the draft on the
  `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint. The upstream 0.68 result was on a
  different weight repo (`deepseek-ai/DeepSeek-V4-Flash`), not this NVFP4 one.
- The leading "MARLIN fallback is the root cause" hypothesis is **weakened, not
  confirmed**. A genuinely different backend (B12X) still produces a 0%-accepted
  draft. The shared factor across both MARLIN and B12X is the **checkpoint's MTP
  weights**, not the MoE backend.
- This redirects the root cause toward the **MTP draft weights themselves**
  (consistent with the earlier `int8` vs `uint8` anomaly in the MTP block) or a
  draft-model **load/forward path** that is independent of `moe_backend`. Recall
  (Act 4 / config): the MTP block is in `quantization_config.ignore` → it is
  **BF16, not NVFP4**. So the draft's MoE experts are unquantized; an
  `int8`/`uint8` reinterpretation would only matter if something still unpacks
  those BF16 bytes as packed FP4. That mismatch is the new prime suspect.

**Status:** MTP draft **still broken (0% accepted)** on B12X. Root cause is
**not** the MoE backend. Next decisive test is either (a) the upstream
`deepseek-ai/DeepSeek-V4-Flash` weights on B12X, or (b) the MARLIN-side
`int8→uint8` reinterpreted-view patch in the MTP load path.

### Reproduction

- Image: `voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132`
- Script: `~/scratch_vllm/serve_b12x_tp2.sh` (+ `b12x_inner.sh` + `patches/nvfp4.py`)
- Key flags: `--moe-backend b12x --linear-backend b12x --attention-backend
  B12X_MLA_SPARSE`, speculative config `method=mtp,num_speculative_tokens=2`,
  `--max-model-len 3072` (KV budget ~1.55 GiB → 3840 tokens), `--gpu-memory-utilization 0.875`
- Caveat: `--max-model-len` had to drop to 3072 because B12X weights consume
  ~80 GiB/GPU, leaving only ~1.55 GiB for KV. This does not affect the
  acceptance measurement (the repeat/counting prompts are tiny).

---

## Act 6 — ROOT CAUSE FOUND + FIXED (2026-06-24): MTP draft experts are MXFP4, loader forced NVFP4

**Root cause (proven by checkpoint inspection + dtype copy_ tests + code trace):**

The `nvidia/DeepSeek-V4-Flash-NVFP4` checkpoint stores the **main model** experts as
**NVFP4** (block scales `float8_e4m3fn`, group_size=16, with `weight_scale_2` +
`input_scale` per-expert global scales), but stores the **MTP draft** experts as
**MXFP4** (block scales `float8_e8m0fnu`, group_size=32, NO `weight_scale_2` /
`input_scale`). Evidence from `model-00046-of-00046.safetensors`:

| key | dtype | shape | format |
|---|---|---|---|
| `layers.0.ffn.experts.0.w1.weight` | `uint8` | (2048,2048) | NVFP4 packed |
| `layers.0.ffn.experts.0.w1.weight_scale` | `float8_e4m3fn` | (2048,256) | group_size=16 |
| `layers.0.ffn.experts.0.w1.weight_scale_2` | `float32` | () | per-tensor global sf |
| `layers.0.ffn.experts.0.w1.input_scale` | `float32` | () | per-tensor input sf |
| `mtp.0.ffn.experts.0.w1.weight` | `int8` | (2048,2048) | MXFP4 packed |
| `mtp.0.ffn.experts.0.w1.scale` | `float8_e8m0fnu` | (2048,128) | group_size=32 |
| (no `weight_scale_2` / `input_scale` for mtp) | — | — | MXFP4 has none |

The `config.json` `quantization_config.ignore` list includes `"mtp.*"`, signaling
"MTP is not quantized the same way." But the **expert prefix** at init time is
`model.layers.{num_hidden_layers+i}.ffn.experts` (the `mtp.` checkpoint name is
only used at weight-load time, not at `get_quant_method` time). `is_layer_skipped`
checks `prefix in ignored_layer` for experts; `model.layers.43.ffn.experts` does
not match `"mtp.*"`, so the draft experts are **not** skipped.

Result: `DeepseekV4FP8Config.get_quant_method` routes the draft experts to
`ModelOptNvFp4FusedMoE` (NVFP4), which:
- declares `w13_weight_scale` as `float8_e4m3fn` (group_size=16),
- requires `weight_scale_2` + `input_scale` (per-tensor global scales).

The MTP loader (`mtp.py`) reinterprets the e8m0 scale via `view(torch.uint8)`
(to preserve raw exponent bytes), then the weight_loader does
`param.data.copy_(loaded_weight)` into the **e4m3** param. `copy_` from
`uint8`→`float8_e4m3fn` performs a **numeric conversion** (proven: byte `0x80`
e8m0=2.0 → e4m3 byte `0x70`=2.0, NOT bit-preserving), destroying the scale
exponent bytes. Additionally, `weight_scale_2`/`input_scale` are absent from the
checkpoint, leaving those params as **uninitialized `torch.empty` garbage**.

Both corruptions make the draft MoE dequant produce garbage → 0% acceptance.

**The fix (proven by measurement):**

Patch `quant_config.py::get_quant_method` to route experts whose layer index
`>= num_hidden_layers` (i.e. MTP draft layers) to `Mxfp4MoEMethod` instead of
`ModelOptNvFp4FusedMoE`. `Mxfp4MoEMethod`:
- declares `w13_weight_scale` as `uint8` (e8m0 bytes preserved by `view(uint8)`
  → `copy_` uint8→uint8 = identity),
- uses `mxfp4_block=32` matching the checkpoint group_size,
- has **no** `weight_scale_2`/`input_scale` requirement,
- supports `B12X` backend with `swiglu_limit` (gemm1_clamp_limit).

Shape check confirms exact compatibility:
- Mxfp4 `w13_weight_scale = (E, 2*inter, hidden//32) = (E, 4096, 128)` == MTP
  fused w1+w3 scale `(4096, 128)`. ✓
- Mxfp4 `w2_weight_scale = (E, hidden, inter//32) = (E, 4096, 64)` == MTP
  w2 scale `(4096, 64)`. ✓

**Measurement (proven, 2026-06-24 16:28):**

With the Mxfp4 routing patch + the nvfp4 clamp-gate patch, B12X TP2, MTP enabled:

- Repeat-prompt (20× "The quick brown fox..."): **Avg Draft acceptance rate
  69.4%**, Per-position 0.907 / 0.481, Mean acceptance length 2.39.
- Diverse prompts (counting, capitals): **74.8%** then **89.3%**, Per-position
  up to 0.929 / 0.857, Mean acceptance length up to 2.79.

This matches the upstream `local-inference-lab/rtx6kpro` report of ~0.68
acceptance for this backend. **MTP is now working.**

**Patches (mount-overlay, non-invasive):**
- `patches/nvfp4.py` — widen `_backend_supports_clamp` to allow B12X + SILU clamp.
- `patches/quant_config.py` — route MTP draft experts (layer >= num_hidden_layers)
  to `Mxfp4MoEMethod` (matches the MXFP4 checkpoint format).

**Caveats:**
- `--max-model-len 3072` is still required (B12X weights ~80 GiB/GPU → only
  ~1.75 GiB KV → 4335 tokens). This is a memory budget limit, not an MTP
  correctness limit; acceptance is measured on tiny prompts.
- The `int8` vs `uint8` weight-dtype anomaly from Act 4 is now explained: MTP
  experts are MXFP4 (stored `int8`), main experts are NVFP4 (stored `uint8`).
  The `int8` is NOT a bug — it is the MXFP4 checkpoint convention. The bug was
  forcing NVFP4 (e4m3-scale, group_size=16, global-sf) interpretation onto
  MXFP4 (e8m0-scale, group_size=32, no-global-sf) data.

**Status: MTP RESOLVED.** Draft acceptance ~69-89% on this hardware
(2× RTX PRO 6000, TP2, B12X MXFP4 draft backend).
