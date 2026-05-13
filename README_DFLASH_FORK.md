# llama.cpp — DFlash + Gemma 4 on Apple Silicon

Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) adding
working **Gemma 4 26B-A4B DFlash speculative decoding** on Mac / Metal,
based on draft PR [#22105](https://github.com/ggml-org/llama.cpp/pull/22105)
(commit `67cb0d507`, 2026-04-27). All additions live on branch
[`dflash-swa-full-detect`](../../tree/dflash-swa-full-detect).

## Result

Gemma 4 26B-A4B-it, M3 Max 40 GB, temp=0, seed=42, draft_max=16.

**Short prompts** — 24 gsm8k prompts with thinking, ctx ≤ 102 tokens:

| backend | target quant | tok/s | accept rate |
|---|---|---:|---:|
| MLX 4bit (reference) | MLX affine 4bit (~15 GB) | 118.6 | 42.79% |
| MLX 8bit | MLX affine 8bit (~26 GB) | 98.0 | 44.30% |
| **llama.cpp Q4_K_M (this fork)** | GGUF K-quant (~16 GB) | **49.1** | **40.08%** |
| llama.cpp Q8_0 (from MLX 4bit dequant) | GGUF Q8_0 (~25 GB) | 44.1 | 40.46% |

**Long prompts** — single prompt, 256-token continuation, `-c <prompt+1024>`:

| prompt tokens | accept rate | tok/s |
|---:|---:|---:|
|  2,212 | 32.46% | 47.1 |
|  5,657 | 48.33% | 42.1 |
|  8,419 | 46.67% |  2.7 |

The 6k row's tok/s collapses because the Q4_K_M target plus an 8K ubatch
saturates unified memory; the accept rate is unaffected.

## Supported model matrix

| Target | DFlash draft | Status |
|---|---|---|
| Qwen3.5 4B / 9B / 35B-A3B / 122B-A10B | matching DFlash | works (4B regression-tested at 40.7 t/s) |
| Qwen3.6 35B-A3B | matching DFlash | inherits upstream #22105 support |
| **Gemma 4 26B-A4B-it** | `z-lab/gemma-4-26B-A4B-it-DFlash` | **works (new in this fork)** |
| Gemma 4 31B (dense) | `z-lab/gemma-4-31B-it-DFlash` | untested but same code paths |

## What this fork changes

Four classes of fix on top of #22105. Click commit hashes for details.

### 1. Variable-length `target_layer_ids` ([`b113d75a8`](../../commit/b113d75a8))

`#22105` hard-codes the array to 5. The Gemma 4 26B-A4B and Qwen3.5-122B
drafts both ship **6** target layers and crash with
`expected 5, got 6`. Bumped to a 16-slot array tracked by an explicit
length counter.

### 2. Gemma 4 target plumbing ([`b113d75a8`](../../commit/b113d75a8), [`e60583afb`](../../commit/e60583afb))

The DFlash decoder reuses the target's `tok_embd` and `lm_head`, so it
must mirror Gemma 4's `sqrt(n_embd)` embedding scale and
`tanh(x/30)*30` logit softcap. Adds the corresponding fields to
`llama_dflash`, populated when `model.arch == LLM_ARCH_GEMMA4`. Also
adds the feature-extraction hook in `gemma4-iswa.cpp` (analogous to
the existing Qwen 3.5 hook), reading `inpL` at the layer top.

### 3. Per-layer sliding-window attention + own softcap + MLX→GGUF expert names ([`cbfb7f7f2`](../../commit/cbfb7f7f2))

Gemma 4 DFlash drafts use `[SW, SW, SW, SW, full]` with `sliding_window=2048`.
Adds three GGUF keys (`dflash.layer_sliding`, `dflash.attention.sliding_window`,
`dflash.final_logit_softcapping`), an F16 mask tensor wired through
`build_attn_mha` only on sliding layers, and a softcap path that prefers
the DFlash GGUF value over the target's.

Also adds an MLX→GGUF expert-tensor rename
(`experts.switch_glu.{gate,up,down}_proj` → `experts.{gate,up,down}_proj`)
so Gemma 4 GGUF target weights can be produced directly from MLX 4bit
safetensors via `mlx_lm convert --dequantize` + the standard
`--fuse-gate-up-exps` flag, with no need for the original BF16 source.

### 4. SWA mask uses absolute query position ([`581cfec89`](../../commit/581cfec89))

The DFlash draft caller populates the noise ubatch with `pos = i`
(block-local 0..15). The SWA mask must reason about absolute sequence
position to evaluate the sliding bound; compute `q_pos = n_ctx + q`
directly instead of reading `ubatch.pos[q]`, matching what RoPE in
`dflash.cpp` already does (`inp_pos_full = [0..n_total-1]`).

Without this fix, the SWA bound never fires on long prompts: the mask
silently degrades to "ctx fully visible", the draft attends to
context ranges far outside its training distribution, and accept rate
collapses (-22pp on 6k prompts).

### CI plumbing ([`3a321a6af`](../../commit/3a321a6af), [`be202f0bb`](../../commit/be202f0bb))

Registers `gemma4` in `convert_hf_to_gguf_update.py` so the
pre-tokenizer-hashes CI check passes, and adds an `assert` in the
EAGLE3 lm_head fallback so Python type-check passes. No functional
change.

## Quick start

### Build

```bash
git clone -b dflash-swa-full-detect https://github.com/Ziqiao-git/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target llama-speculative-simple
```

### Convert Gemma 4 DFlash draft

Needs a target tokenizer directory (any HF-format Gemma 4 26B-A4B-it).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U transformers torch sentencepiece protobuf

PYTHONPATH=$PWD/gguf-py python convert_hf_to_gguf.py \
  z-lab/gemma-4-26B-A4B-it-DFlash \
  --outtype bf16 \
  --target-model-dir mlx-community/gemma-4-26b-a4b-it-8bit \
  --outfile gemma-4-26b-a4b-dflash.gguf
```

### Convert Gemma 4 target from MLX 4bit (optional)

If you don't already have a Gemma 4 target GGUF and want to avoid
downloading the BF16 source:

```bash
pip install mlx mlx-lm

# MLX 4bit → dequantized bf16 safetensors (~47 GB)
python -m mlx_lm convert \
  --hf-path mlx-community/gemma-4-26b-a4b-it-4bit \
  --mlx-path gemma-4-26b-a4b-it-bf16-dequant \
  --dequantize

# bf16 → GGUF Q8_0 (~25 GB)
PYTHONPATH=$PWD/gguf-py python convert_hf_to_gguf.py \
  gemma-4-26b-a4b-it-bf16-dequant \
  --outtype q8_0 \
  --fuse-gate-up-exps \
  --outfile gemma-4-26B-A4B-it-Q8_0.gguf
```

### Run

Short prompt:

```bash
./build/bin/llama-speculative-simple --dflash \
  -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -md gemma-4-26b-a4b-dflash.gguf \
  -p "Solve step by step: ..." \
  -n 256 --temp 0 --top-k 1 --seed 42 --draft-max 16 -c 4096
```

Long prompt (e.g. ~5 700 tokens):

```bash
./build/bin/llama-speculative-simple --dflash \
  -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  -md gemma-4-26b-a4b-dflash.gguf \
  -f long_prompt.txt \
  -n 256 --temp 0 --top-k 1 --seed 42 --draft-max 16 \
  -c 6700 -b 6144 -ub 6144
```

Required flags for long prompts on M3 Max:

- `-c <prompt_tokens + 1024>`: don't oversize, KV cache scales with this
- `-b <ubatch>` and `-ub <ubatch>`: ubatch must satisfy `ubatch >= prompt_tokens`
  (encoder requires single-shot ingest) but should be the minimum that
  works to keep compute buffer in budget

Expected log lines on Gemma 4:

```
load_hparams: DFlash layer_attn = [sw, sw, sw, sw, full], sliding_window = 2048
load_hparams: DFlash final_logit_softcap = 30.00
set_dflash: DFlash target = Gemma4: embed_scale=53.0660, final_logit_softcap=30.00
```

## Limitations

- Branch is based on PR #22105's HEAD (2026-04-27). Upstream master has
  since merged #22506 (low-prob draft trim) and #22679 (device-side spec
  checkpoint); rebasing is unblocked work, not done here.
- No Metal kernel optimization. The 49 vs 118 tok/s gap to MLX is engine-
  level (ggml-metal vs MLX Metal kernel), unrelated to DFlash.
- 24-prompt gsm8k regression bench is in the repo as bash scripts under
  `/tmp/run_q[48]_bench.sh` (run-and-aggregate); no automated CI test
  added.

## Credits

- DFlash framework + Qwen support: [@ruixiang63](https://github.com/ruixiang63) et al. in [#22105](https://github.com/ggml-org/llama.cpp/pull/22105)
- DFlash algorithm: [z-lab/dflash](https://github.com/z-lab/dflash); MLX reference in [`dflash/model_mlx.py`](https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py)
- SWA mask construction pattern: cross-checked against [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)'s SD-073 (their dflash draft caller emits absolute pos, ours doesn't, hence the q_pos hardcode in fix 4)
- llama.cpp: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

## License

MIT, same as upstream llama.cpp.
