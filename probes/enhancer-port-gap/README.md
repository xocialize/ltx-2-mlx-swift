> # ⛔ SUPERSEDED — 2026-08-16, same day. See **AB-R-0085**.
> **The conclusion below is wrong.** There is no systemic mlx-swift-lm gap: its model compute is at
> **parity** with Python `mlx_lm` (prefill 0.62 s vs 0.71 s — Swift faster; decode 61 vs 65 tok/s).
> The 2.6× was two unrelated costs:
> 1. **a DEBUG build** — `RunLTX2` is `swift build`, and mlx-swift's `Cmlx` target sets no
>    optimization flags, so libmlx's CPU path compiles at `-O0`. ABBA release↔debug: decode
>    **26.4 → 61 tok/s (2.31×)**. That is the whole decode half.
> 2. **a fixed 1.30 s per-pass chat-template + tokenize cost with no model forward** (1.14 s of it
>    tokenization; Python does the same string in 0.45 ms). The "1.90 s prefill" below is
>    `1.30 s prepare + 0.62 s prefill`.
>
> 🔑 **Why the method below produced a false finding:** it split the phases with two *wall-clock*
> runs and subtracted, which charges all fixed per-pass overhead to prefill and the remainder to
> decode — mismeasuring both in the same direction and manufacturing the "uniform factor" that made
> one systemic cause look compelling. Take the split from the generator's own
> `GenerateCompletionInfo.promptTime`/`generateTime` instead (`RunLTX2 --gen-gap-probe`).
>
> Kept unedited below as the record of how the wrong answer was reached.

# The ~2.6× Swift-vs-oracle generation gap — measured, same machine

Arm E (AB-R-0075) recorded that Swift's enhancer pass was ~2× slower per word than the oracle's and
left it unexplained. This settles it: **the gap is real, it is roughly UNIFORM across prefill and
decode, and it is in the Swift generative stack — not in LTX code.**

## Method

`oracle_enhance.py` mirrors `ltx_core_mlx` `base_encoder._enhance` exactly — same
`gemma_t2v_system_prompt.txt` (4175 chars / 928 tokens), same user template `"user prompt: <p>"`,
`mlx_lm.generate`, `make_sampler(temp=0.7)`, `mx.random.seed(10)`, `max_tokens=512`, same
`mlx-community/gemma-3-12b-it-4bit`. Run with the oracle's own venv:

    cd LTX_DEV/ltx-2-mlx && .venv/bin/python <this dir>/oracle_enhance.py

Swift side: `RunLTX2 --enhancer-bench <gemmaDir> <maxTokens> <seed> --no-seam`.
**`maxTokens=1` isolates prefill** on both sides — comparing totals alone is what left this
ambiguous in arm E.

## Result (2026-08-16, same machine, minutes apart)

| phase | oracle (Python mlx_lm) | Swift (mlx-swift-lm) | ratio |
|---|---|---|---|
| prefill, 948 tok | **0.66 / 0.65 s** | **1.90–1.95 s** | **≈2.9×** |
| decode | 174 tok in 2.67 s = **65 tok/s** | 164 tok in 6.67 s = **24.6 tok/s** | **≈2.6×** |
| total (E1) | **3.33 / 3.31 s** | 7.30–8.57 s | ≈2.4× |

Oracle reproduces to 0.02 s across two runs, and its 3.3 s independently confirms the 3.1 s in
`ltx-2-mlx/docs/claims/C5-prompt-enhancer.md` — so the original figure was NOT a measurement
artifact, as I had suspected it might be.

## Reading it

🔑 **Uniform across both phases ⇒ not one slow kernel and not the system prompt.** My first
hypothesis was that Swift re-prefills the 928-token system prompt every pass; it does, but that is
only 1.93 s of a 7.56 s pass, and the oracle prefills the same tokens in 0.66 s. Prefill and decode
are each ~2.6–2.9× slower, which points at something systemic in the Swift generation loop
(per-token overhead, eval granularity, sampler or KV-cache handling) rather than at any single op.

⚠️ **This is NOT LTX code.** Both stacks share the same `libmlx` C++ core; the difference is the
front-end (`mlx_lm` vs `mlx-swift-lm` driven through `ChatSession`). LTX is just where it surfaced.
**Anything in the fleet generating text through mlx-swift-lm is likely paying the same factor** —
which is why this was raised cross-area rather than filed as an LTX bug.

⚠️ **Not diagnosed.** This measures the gap and localises it to "both phases, Swift front-end". It
does not identify the cause. Do not quote a mechanism.
