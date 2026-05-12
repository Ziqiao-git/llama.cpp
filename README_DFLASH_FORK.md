# llama.cpp — DFlash + Gemma 4 on Apple Silicon

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that
adds **working Gemma 4 26B-A4B DFlash speculative decoding on Mac / Metal**.
Base: PR [#22105](https://github.com/ggml-org/llama.cpp/pull/22105) HEAD
(`67cb0d507`, 2026-04-27). All extra commits live on branch
[`dflash-swa-full-detect`](../../tree/dflash-swa-full-detect).

For per-line review notes see [dflash_gemma4_review.md](../../blob/dflash-swa-full-detect/README_DFLASH_FORK.md)
(this file) and `dflash_gemma4_review.md` outside the repo at the workspace root.

---

## What this fork adds

Three problems block Gemma 4 DFlash on stock #22105; this fork fixes all three.

### 1. Variable-length `target_layer_ids` (was hard-coded to 5)

Stock #22105 stores `target_layer_ids` as `std::array<int, 5>`. The Gemma 4 26B-A4B
DFlash draft and Qwen3.5-122B-A10B both ship **6** target layer ids and immediately
crash with:

```
key dflash.target_layer_ids has wrong array length; expected 5, got 6
```

This fork bumps the slot to 16 and tracks the actual length via
`n_dflash_target_layer_ids`. See commit [`b113d75a8`](../../commit/b113d75a8).

### 2. Gemma 4 target plumbing

Gemma 4 multiplies `tok_embd` by `sqrt(n_embd)` and softcaps logits with
`tanh(x/30)*30`. The DFlash decoder reuses the target's `tok_embd` and
`lm_head`, so it must mirror both. We also fix an off-by-one in the
extraction hook for Gemma 4 specifically (the analogous Qwen 3.5 hook reads
`inpL` at the layer top; the original Gemma 4 hook read `cur` at the layer
bottom, one layer too far). See commits
[`b113d75a8`](../../commit/b113d75a8) and [`e60583afb`](../../commit/e60583afb).

### 3. Per-layer sliding-window attention in the DFlash decoder

Gemma 4 DFlash drafts use a mix of sliding-window and full-attention layers
(`[SW, SW, SW, SW, full]`, window=2048). Without sliding-window-aware causal
masking the decoder runs every layer as bidirectional full-attention, halving
accept rate. We pipe the per-layer pattern from the HF config through GGUF
into a host-filled F16 mask tensor and pass it to `build_attn_mha` only on
sliding layers. See commit [`cbfb7f7f2`](../../commit/cbfb7f7f2).

The same commit also wires DFlash's own `final_logit_softcapping` (independent
of the target's) through GGUF, and adds an MLX-quantized→GGUF tensor-name
translation so you can convert Gemma 4 GGUF target weights directly from MLX
4bit safetensors (no fp16 source download needed).

---

## Result

Benchmark: 24 gsm8k prompts (thinking on), temp=0, draft_max=16, M3 Max 40 GB.

| Backend | target quant | tok/s | accept rate | accept length / 16 |
|---|---|---:|---:|---:|
| MLX 4bit (reference) | MLX affine 4bit (~15 GB) | 118.6 | **42.79%** | 7.85 |
| MLX 8bit | MLX affine 8bit (~26 GB) | 98.0 | 44.30% | 8.09 |
| **llama.cpp Q4_K_M (this fork)** | GGUF K-quant (~16 GB) | **49.1** | **40.08%** | 7.41 |
| **llama.cpp Q8_0** (from MLX 4bit dequant) | GGUF Q8_0 (~25 GB) | **44.1** | **40.46%** | 7.47 |

**Accept rate within ~2.7pp of MLX**. The 49 vs 118 t/s gap is the ggml-metal
kernel vs MLX-Metal kernel — engine-level, not algorithmic. Same gap we
observed on Qwen3.5-4B DFlash, unrelated to this fork.

Acceptance got there from a 7.5% starting point through three sequential fixes:

| Stage | Accept rate (24 prompts) | tok/s |
|---|---:|---:|
| Before any fix (broken extraction hook) | 7.5% (single prompt) | 18.7 |
| + off-by-one hook fix (`e60583afb`) | 26.1% | 33.9 |
| + apples-to-apples chat template + 24-prompt avg | 37.5% | 50.4 |
| **+ per-layer sliding mask + softcap (`cbfb7f7f2`)** | **40.08%** | **49.08** |

---

## Quick start

```bash
git clone -b dflash-swa-full-detect https://github.com/Ziqiao-git/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target llama-speculative-simple
```

### Convert Gemma 4 DFlash draft

You need a target tokenizer dir (any HF-format Gemma 4 26B-A4B — MLX or
Google's original works since we only need tokenizer files):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U transformers torch sentencepiece protobuf

PYTHONPATH=$PWD/gguf-py python convert_hf_to_gguf.py \
  z-lab/gemma-4-26B-A4B-it-DFlash \
  --outtype bf16 \
  --target-model-dir mlx-community/gemma-4-26b-a4b-it-8bit \
  --outfile gemma-4-26b-a4b-dflash.gguf
```

(Pass local paths to anything that's already on disk.)

### Convert Gemma 4 target from MLX 4bit (optional, only if you don't already
have a target GGUF)

```bash
pip install mlx mlx-lm

# Step 1: MLX 4bit → dequantized bf16 safetensors (~47 GB)
python -m mlx_lm convert \
  --hf-path mlx-community/gemma-4-26b-a4b-it-4bit \
  --mlx-path gemma-4-26b-a4b-it-bf16-dequant \
  --dequantize

# Step 2: bf16 → GGUF Q8_0 (~25 GB)
PYTHONPATH=$PWD/gguf-py python convert_hf_to_gguf.py \
  gemma-4-26b-a4b-it-bf16-dequant \
  --outtype q8_0 \
  --fuse-gate-up-exps \
  --outfile gemma-4-26B-A4B-it-Q8_0.gguf
```

### Run speculative decoding

```bash
./build/bin/llama-speculative-simple --dflash \
  -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -md gemma-4-26b-a4b-dflash.gguf \
  -p "Solve step by step: A train travels 60 mph for 2.5 hours, then 80 mph for 1.5 hours. Total distance?" \
  -n 256 --temp 0 --top-k 1 --seed 42 --draft-max 16 -c 4096
```

**`-c 4096` is required on M3 Max for Q8_0** — the default `n_ctx=262144`
allocates an 80 GB KV cache and OOMs the GPU.

Expected log lines (Gemma 4 path):

```
load_hparams: DFlash layer_attn = [sw, sw, sw, sw, full], sliding_window = 2048
load_hparams: DFlash final_logit_softcap = 30.00
set_dflash: DFlash target = Gemma4: embed_scale=53.0660, final_logit_softcap=30.00
...
accept    = ~40%
```

---

## Supported models

| Target | DFlash draft | Status on this fork |
|---|---|---|
| Qwen3.5 4B / 9B / 35B-A3B / 122B-A10B | matching DFlash | works (regression-tested 4B at 40.7 t/s, 28.7% accept) |
| Qwen3.6 35B-A3B | matching DFlash | inherits upstream #22105 support |
| **Gemma 4 26B-A4B-it** | `z-lab/gemma-4-26B-A4B-it-DFlash` | **works (new in this fork)** |
| Gemma 4 31B (dense) | `z-lab/gemma-4-31B-it-DFlash` | not tested but should work — same code paths |

---

## What's NOT in this fork

- **No retest against latest llama.cpp master.** Branch is based on PR #22105's
  HEAD (2026-04-27). Master since then merged #22506 (low-prob draft trim) and
  #22679 (device-side spec checkpoint); rebasing would conflict in
  `llama-context.cpp` and `llama-graph` — plan ~half a day.
- **No Metal kernel optimization.** Closing the 49 vs 118 tok/s gap to MLX is
  ggml-metal kernel work, unrelated to DFlash. Tracked in upstream #22400
  (currently stalled on Apple).
- **Default `n_ctx` is unchanged.** Users with limited GPU memory must pass
  `-c 4096` explicitly.
- **No CI / tests added.** Manual benchmark in `/tmp/run_q[48]_bench.sh`
  reproduces the table above.

---

## Credits

- Base DFlash framework: [@ruixiang63](https://github.com/ruixiang63) and contributors in PR [#22105](https://github.com/ggml-org/llama.cpp/pull/22105)
- DFlash algorithm: [z-lab/dflash](https://github.com/z-lab/dflash), MLX reference impl in [`dflash.model_mlx`](https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py)
- llama.cpp: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

---

## License

MIT, same as upstream llama.cpp.
