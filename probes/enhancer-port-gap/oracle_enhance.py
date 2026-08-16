"""Same-machine oracle side of the C5 enhancer, to settle whether the ~2x Swift gap is real.

Mirrors ltx_core_mlx base_encoder._enhance EXACTLY: same system prompt file, same user template
("user prompt: <p>"), mlx_lm.generate, make_sampler(temp=0.7), mx.random.seed(10), max_tokens=512.
Reports prefill and decode SEPARATELY (mlx_lm exposes both) so it decomposes the same way the Swift
harness now does -- comparing totals alone is what made this ambiguous.
"""
import time, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = "/Volumes/Satechi/Models/mlx-community/gemma-3-12b-it-4bit"
SP = Path("packages/ltx-core-mlx/src/ltx_core_mlx/text_encoders/gemma/encoders/prompts/gemma_t2v_system_prompt.txt")
system = SP.read_text()
cases = [("E1", "a lighthouse keeper climbing the spiral stairs"),
         ("E2", "a red fox in snow")]

t0 = time.time(); model, tok = load(MODEL); print(f"load {time.time()-t0:.2f}s", flush=True)
print(f"system prompt: {len(system)} chars, {len(tok.encode(system))} tokens", flush=True)

for name, p in cases:
    chat = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": f"user prompt: {p}"}],
        tokenize=False, add_generation_prompt=True)
    ntok_in = len(tok.encode(chat))
    mx.random.seed(10)
    t = time.time()
    out = generate(model=model, tokenizer=tok, prompt=chat, max_tokens=512,
                   sampler=make_sampler(temp=0.7), verbose=False)
    dt = time.time() - t
    n_out = len(tok.encode(out))
    print(f"{name}: in={ntok_in} tok  out={n_out} tok  {dt:.2f}s  -> {n_out/dt:.1f} tok/s aggregate",
          flush=True)

# prefill isolation: same chat, 1 token out
name, p = cases[0]
chat = tok.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": f"user prompt: {p}"}],
    tokenize=False, add_generation_prompt=True)
mx.random.seed(10); t = time.time()
generate(model=model, tokenizer=tok, prompt=chat, max_tokens=1, sampler=make_sampler(temp=0.7), verbose=False)
print(f"PREFILL-only (max_tokens=1): {time.time()-t:.2f}s", flush=True)
