# LTX-2.3 — Consumer MVP Readiness

**Status: OPEN** — memory optimization is CLOSED-COMPLETE (see baseline below); this checklist is
what separates "optimized" from "shippable to a consumer." When M1–M5 are ✅, the stack is MVP.
Speed work is deliberately NOT here — it is the Release-2 theme ([SPEED-PLAN.md](SPEED-PLAN.md)),
opened only if M1's target-hardware numbers demand it.

Owners: **pkg** = package agent (this repo) · **app** = Xcode agent (LTXVideoTesting, via
AGENT_BRIDGE) · **user** = hands-on / business.

---

## Baseline — what is already DONE and measured (do not re-litigate)

| Area | State |
|---|---|
| Memory tiers | 4 governed profiles, all measured (LOW-TIER-PLAN T0–T4): compact24 **15.36 GB** / balanced32 16.07 / standard64 37.51 / max128 72.7 (481f i2v + LoRA). 24 GB M5 MBP budget met with headroom. 16 GB deliberately unsupported (int4 DiT alone ≈ the budget; no smaller checkpoint exists). |
| Engine governance | Split footprints + tier hints charged by the MemoryGovernor; LRU-evict-to-admit proven in-app; enhancement LLM governed (`mlx-gemma-llm-swift` v0.1.0, BRIDGE-LTX-006 closed, 15.02 GB acceptance). |
| Throughput fixes | Two-track encode deadlock fixed (audio-first); DiT kernel warmup at load; fused MLXFast norms; chunked VAE decode (halo 5, seam-clean); connector int8. |
| Correctness | Full parity-gate suite green on mlx-swift-lm 3.31.4; i2v first-frame orientation fixed; LoRA re-apply across DiT evict/reload; NFKC/neg-prompt class fixed. |
| GUI | Tier picker + clamp surfacing + quant-follow + governed Enhance, all acceptance-validated by the Xcode agent. |

---

## M1 — Target-hardware validation (the highest-value item)  ▶ user (hands-on)

**Gap:** every number above was measured on the 128 GB M5 Max desktop. A real 24 GB M5 MacBook
Pro differs in GPU-core count, laptop thermals, and true memory pressure (the OS and other apps
live inside that 24 GB too). Peak memory will hold — the envelope math guarantees it — but
wall-clock and sustained-load behavior are UNKNOWN on the machine the compact24 tier exists for.

**Do:** follow **[LTX_TESTING/M1-TARGET-HARDWARE-PLAN.md](../LTX_TESTING/M1-TARGET-HARDWARE-PLAN.md)**
(the runnable session plan — weights copy, run matrix R0–R6, the S1 profiled run, results tables).
Summary, on a base 24 GB M5 MBP (and ideally one 32 GB machine):
1. Cold start → tier auto-defaults to compact24 → t2v 704×512×240 request (proves clamp) →
   record load s / run s / peak GB / output correctness (512×288×121).
2. Repeat 3× back-to-back for thermal drift (run-time creep = throttling signal).
3. One i2v run with the adapter LoRA (the heavier path).
4. One governed Enhance → generate (Gemma swap-in under real memory pressure).

**Accept:** peak ≤ 16.8 GB (expected ~15.4); no watchdog/OOM; run time recorded honestly —
**if 121f lands ≥ ~3 min, open SPEED-PLAN S1 before shipping**; output clean at int4.

**Record results in this file** (table below) — this doc is the ledger.

| machine | tier | request | load s | run s | peak GB | 3×-drift | verdict |
|---|---|---|---|---|---|---|---|
| MBP M5 Pro 24 GB (Mac17,9, macOS 27.0, weights on TB5 ext.) | compact24/int4 | R1 t2v 704×512×240 → clamped 512×288×121 ✅ | 10.0 | 110.2 (first-run shape compile) | 14.87 | — | ✅ |
| 〃 | 〃 | R2–R4 same, back-to-back | 7.5–7.9 | 86.1 / 85.5 / 81.4 | 14.89–14.91 | **negative** (got faster; no throttle at this envelope) | ✅ |
| 〃 | 〃 | R5 i2v + adapter LoRA | 8.2 | ≈525 true (JSON 3628.7 contaminated by a 52-min silent adapter download — see notes) | **18.86** | — | ⚠️ exceeds 16.8 watch line; 0.25 GB under Metal workingSet 19.1; NO paging/watchdog |
| 〃 | 〃 | R6 governed Enhance → generate | 15.8 | 280.9 (incl. enhance 51.2 + gaps) | 15.07 | — | ✅ no memory stacking; governor confirmed |

**M1 VERDICT (2026-07-05): ✅ SHIP for the core t2v path** — peak 14.9 ≪ 16.8, no
watchdog/OOM/paging, 121f ≈ 85 s warm (≪ the 3-min speed trigger), clamp + tier-default +
governed-Enhance all confirmed on target. S1 closed alongside: **16.1 s per output-second** (≤ 20
⇒ ship as-is; full baseline in SPEED-PLAN §S1). **One flagged exception:** the i2v+adapter path
peaks 18.86 GB — works, but with ~0.25 GB Metal-ceiling headroom on a 24 GB machine; needs an
i2v-specific clamp or memory pass before THAT path is consumer-shippable (BRIDGE-LTX-012).
Session artifacts + operator notes: `/Volumes/Satechi/Testing/ltx-portable/results-MacBook-Pro-20260705-133428/`
(app bugs found there — silent adapter download shown as "Generating", adapter auto-select at
launch, backbone quant-follow miss — are app-side follow-ups in the bridge mailbox). Output
quality: MP4s verified 512×288×5.04 s; operator notes flag no int4 artifacts (explicit visual
sign-off still worth a minute when convenient).

## M2 — Quit-during-generation crash  ▶ app (+ pkg if a cancel seam is missing)

**Gap:** external AppleEvent quit mid-generation crashes in MLX teardown
(`ThreadPool::enqueue — Not allowed on stopped ThreadPool`). Known since the 120f/LoRA session;
parked as app-robustness. A consumer WILL Cmd-Q mid-run on day one.

**Do:** app terminate handler: intercept quit → cancel the generation Task → await drain (bounded,
e.g. 10 s) → then allow teardown. Package side commits to cancellation points that make the drain
fast (see M3).

**Accept:** Cmd-Q + AppleEvent quit during (a) denoise, (b) decode, (c) encode → clean exit, no
crash log, ≤ ~10 s.

## M3 — Mid-run cancellation actually stops the run  ▶ pkg then app

**Gap:** the wrapper honors `Task.checkCancellation` before/after generation, but a consumer
Cancel button must stop an in-flight denoise/decode in seconds, not after it completes.

**Do (pkg):** add `try Task.checkCancellation()` per denoise step (the loop already has per-step
profiler hooks — same seam) and per VAE decode chunk. Cheap, no perf cost.
**Do (app):** Cancel button wired to the run Task; verify UI returns to idle and a follow-up run
works (DiT state clean, LoRA re-apply intact).

**Accept:** cancel at step 2/8 → run stops < 5 s; immediate re-run succeeds; cancel during decode
→ same. (Also the mechanism M2's drain relies on.)

## M4 — True cold-start first-run experience  ▶ app + user

**Gap:** all validation ran with weights pre-materialized on DEV_ARCHIVE. A consumer's first run =
download (Gemma 7.5 GB + DiT 11.3 GB int4 + components) → license acknowledgment → first load
(kernel compile) → first generation. The engine's downloader/progress/store surfaces exist
(`mlxengine-implementation` seams) but the full path has never run end-to-end on an empty machine.

**Do:** on a machine (or fresh user account / emptied models folder) with NOTHING cached: pick the
models folder → observe download progress for every component → license acceptance UX for the
LTX-2 Community weights → first load (expect the one-time kernel-compile pause — verify the UI
says "Loading", not frozen) → first generation.

**Accept:** zero manual file placement; progress visible end-to-end; interrupted download resumes
or fails legibly; total time-to-first-video recorded (this number IS the consumer first
impression — put it in the M1 table).

## M5 — Weights-distribution license posture  ✅ CLOSED (operator go, 2026-07-05)

**Operator decision (2026-07-05): reviewed and found acceptable for our business needs — written
go.** The checklist below is retained as the record of what the review covered; if the product's
positioning, revenue posture, or weights host changes, M5 REOPENS (§2 revenue gate and §A.20
non-compete bind on an ongoing basis, not once).

**Gap:** the engineering license posture (inference code Apache, weights LTX-2 Community,
engine-allowlisted) was settled for EVALUATION. Consumer shipping is a different act: end users
downloading the weights pulls in §3 (license passthrough + AI-output labeling as enforceable
end-user terms), and §2 (≥$10M revenue gate) + §A.20 (non-compete with Lightricks' video
products) bind the shipper on an ongoing basis.

**Do (product-counsel pass, not engineering):**
- [ ] Confirm shipper entity remains under the §2 revenue gate (or obtain the paid license).
- [ ] End-user terms carry the LTX-2 Community license + Attachment-A AUP passthrough.
- [ ] AI-generated-output labeling in the product (§3 obligation).
- [ ] §A.20 non-compete reviewed against the product's positioning.
- [ ] Weights download source settled (current: `dgrauet/ltx-2.3-mlx` mirror; decide whether to
      graduate to a first-party or mlx-community host, with the license file served alongside).
- [ ] Gemma weights: Gemma Terms §3.1 notice + terms passthrough wherever the enhancer ships
      (already engine-allowlisted as `LicenseRef-Gemma-Terms`; the notice is a product-UI item).

**Accept:** written go/no-go. This is the item a profiler never shows and is most easily forgotten.

---

## Exit

- [x] M1 ✅ (2026-07-05: verdict SHIP, 16.1 s/os — table above; i2v+adapter exception → BRIDGE-LTX-012)
- [ ] M2 ✅  · [ ] M3 ✅  · [ ] M4 ✅  · [x] M5 ✅ (2026-07-05 operator go)

When all five are checked: **declare MVP**, tag the repos, and move speed work to
[SPEED-PLAN.md](SPEED-PLAN.md) as the Release-2 theme.
