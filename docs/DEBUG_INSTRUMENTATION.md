# How the MTP draft was instrumented

This documents the throwaway instrumentation used to diagnose the broken MTP
draft (see [`MTP_INVESTIGATION.md`](MTP_INVESTIGATION.md)). It is **not** a patch
file and does not copy any upstream code — only the small dump snippets we
inserted and where, so the diagnosis is reproducible against your own install.

All snippets are gated on an env var so they're inert unless you opt in:

```bash
-e DSV4_MTP_DEBUG=1 -e PYTHONDONTWRITEBYTECODE=1
```

## Workflow

Because the model runs in a worker subprocess, you can't introspect it live;
you patch the installed files, rebuild, and read the container logs.

```bash
# 1. copy the file out, edit, copy back into a running container
docker cp <ctr>:/usr/local/lib/python3.12/dist-packages/vllm/<path> ./x.py
#    ... add the dump snippet ...
docker cp ./x.py <ctr>:/usr/local/lib/python3.12/dist-packages/vllm/<path>

# 2. kill stale bytecode (a stale .pyc silently wins over your edit)
docker exec <ctr> bash -lc 'rm -f .../__pycache__/<file>*.pyc; touch .../<file>.py'

# 3. freeze into an image and run with the debug env
docker commit <ctr> dsv4-flash-sm120:mtpdbg
docker run ... -e DSV4_MTP_DEBUG=1 -e PYTHONDONTWRITEBYTECODE=1 \
  --enforce-eager ...   # eager is required, see below
```

### Two gotchas that cost real time

- **Wrong runner.** This build ships two runners. The active one is
  `vllm/v1/worker/gpu_model_runner.py` (call it V1); there's also
  `vllm/v1/worker/gpu/model_runner.py` (V2). Logs from `gpu_model_runner.py:*`
  tell you which is live. Instrument the live one.
- **CUDA graphs hide the draft.** The draft `forward` is captured in a FULL CUDA
  graph; on replay the Python body does not execute, so a dump there never
  fires. Run with `--enforce-eager` to make the draft run in Python. (This is
  also how we ruled out cudagraph as the cause: eager still gives 0% acceptance.)

## The dumps (DeepSeek-V4 nvidia MTP model)

Three insertion points in the MTP model module
(`vllm/models/deepseek_v4/nvidia/mtp.py`):

**(a) Draft layer input** — at the top of the predictor layer's `forward`, after
the `inputs_embeds` assert. Skip CUDA-graph capture and dummy (zero-norm) runs:

```python
import os, torch
if os.environ.get("DSV4_MTP_DEBUG") and not torch.cuda.is_current_stream_capturing():
    ph = previous_hidden_states
    if float(ph[-1].float().norm()) > 0:        # real, non-dummy step
        print(f"[fwd] ids={input_ids.flatten()[-4:].tolist()} "
              f"pos={positions.flatten()[-4:].tolist()} "
              f"ph{tuple(ph.shape)} nan={bool(ph.isnan().any())} "
              f"norm={float(ph[-1].float().norm()):.1f}", flush=True)
```

**(b) Per-stage intermediates** — capture the hidden state after `h_proj+e_proj`,
after `mtp_block`, and after `mhc_post`, to find where the signal turns wrong:

```python
# after each stage, stash a reference; then once:
def _st(t): tf = t[-1].float(); return f"max={float(tf.abs().max()):.1f} nan={bool(t.isnan().any())}"
print(f"[stage] A={_st(h_after_proj)} B={_st(h_after_block)} C={_st(h_after_mhc)}", flush=True)
```

**(c) Draft prediction** — in `compute_logits`, after logits are produced:

```python
if os.environ.get("DSV4_MTP_DEBUG") and logits is not None \
        and not torch.cuda.is_current_stream_capturing():
    am = logits[-8:].argmax(dim=-1).flatten().tolist()
    if max(am) > 0:
        print(f"[logit] draft_argmax={am} nan={bool(logits.isnan().any())}", flush=True)
```

Decode `draft_argmax` with the model tokenizer and compare to the ground-truth
continuation. For "repeat the same sentence", the gold tokens are obvious; if the
draft's tokens are unrelated multilingual fragments, the draft is broken.

**(d) Target-side residual stash** (optional, in
`vllm/models/deepseek_v4/nvidia/model.py`, right after the target copies its
pre-hc_head residual into the MTP buffer) — to confirm the draft receives what
the target produced:

```python
if os.environ.get("DSV4_MTP_DEBUG") and not torch.cuda.is_current_stream_capturing():
    f = hidden_states.flatten(1)
    if float(f[-1].float().norm()) > 0:
        print(f"[stash] {tuple(f.shape)} norm={float(f[-1].float().norm()):.1f}", flush=True)
```

## The offline weight check (no GPU, no vLLM)

The decisive lead came from reading the checkpoint directly. Compare the dtype of
the MTP block's MoE expert weights against the main layers':

```python
import json
from safetensors import safe_open
D = "<model dir>"
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
def dtype(k):
    with safe_open(f"{D}/{idx[k]}", framework="pt") as f:
        return f.get_tensor(k).dtype
print("MTP ", dtype("mtp.0.ffn.experts.0.w1.weight"))      # -> torch.int8
print("MAIN", dtype("layers.0.ffn.experts.0.w1.weight"))   # -> torch.uint8
```

The mismatch (`int8` vs `uint8` for the same packed NVFP4 bytes) is the current
leading root-cause candidate.
