# Run progress — what the UI can show while a video generates

Written for design/UX. No Swift required. Every sequence below was **measured**, not designed on
paper (`LTX_PROGRESS=1`, 2026-08-22).

---

## 1. The stepper has two shapes, and the app is told which up front

**The node list depends on the machine.** Smaller Macs run a shorter pipeline. Ask the package for
`plannedStages(...)` before the run and it returns the ordered list, so the connecting line can be
drawn on first paint.

**Smaller Macs (24 / 32 GB) — 4 nodes**

> Read prompt → Generate → Render frames → Finish

**Larger Macs (64 / 128 GB) — 6 nodes**

> Read prompt → Generate → Upscale → Refine → Render frames → Finish

The extra two nodes are a genuine second pass: the first generates at lower resolution, then it is
upscaled and **refined** at full resolution. That is why larger machines look different rather than
just faster.

### Nodes that only sometimes appear — and they go FIRST

| node | when | why it matters to the user |
|---|---|---|
| **Download model** | first run only | tens of GB. Minutes to tens of minutes. |
| **Prepare image adapter** | image-to-video, first time only | **4.93 GB**. ⚠️ Without this node the app appears frozen — the download happens mid-run with nothing else on screen. |

⚠️ **Prompt enhancement is NOT in this list.** It runs through a different component, so if the app
offers it, the app owns that node and places it before "Read prompt".

---

## 2. What each node can show on the card

| node | live counter | notes |
|---|---|---|
| **Read prompt** | ✅ `layer 34 of 48` | fast, steady ticks — good for early liveness |
| **Generate** | ✅ `step 5 of 8` | the slowest ticks; on 6-node runs also `pass 1 of 2` |
| **Upscale** | ❌ none | short |
| **Refine** | ✅ `step 2 of 3` | `pass 2 of 2` |
| **Render frames** | ⚠️ **usually none** — see §4 | can be very slow at high resolution |
| **Finish** | ❌ none | writing the file |

**Always available, independent of progress** (from the engine, not this list): memory in use,
memory headroom, and a system-pressure signal for the "don't launch other apps" overlay.

**Static facts worth putting on the card** — known before the run starts: output size and length,
quality tier, and whether weights are being streamed from disk.

🚨 **Show the ACTUAL output size, not what was requested.** The pipeline silently reduces geometry
to what the machine can handle — ask for 1280×704 on most tiers today and you get 704×512. The user
must not discover that in the exported file.

---

## 3. ⚠️ Do not build a percentage bar out of these steps

**The steps are not equal in time.** 48 prompt-reading steps can pass in the time of one generate
step. A bar driven by counting steps will race, then appear frozen.

✅ **The pulse-on-current-node design avoids this entirely** — animate on event *arrival*, use the
counters as text. If a determinate bar is ever required, weight the phases by measured share rather
than by step count.

---

## 4. "Render frames" now reports a counter — but you learn the total only when it starts

**This section previously warned that this node reports nothing at high resolution. That is fixed.**
Frame decoding used to count **chunks**, and chunking only engages on long clips — so a short clip at
high resolution, the case where decoding takes *longest*, reported nothing at all.

It now counts **decode windows**, which is the work it actually does: `chunks × spatial tiles`. HD
geometries are tiled 2×2, so the units were there all along. Concretely:

| Geometry | Windows reported | Before |
|---|---|---|
| 1920×1088 × 121f | **4** (1 chunk × 4 tiles) | nothing |
| 704×512, short | **1** (honest 1/1) | nothing |
| 704×512, long clip | chunks × 1 | same as before |

**One thing to design for:** the *total* is not in the plan up front. It depends on resolution,
frame count, and decode-tuning overrides, so `RunStage.expectedSubSteps` is `nil` for this node
while `emitsSubSteps` is `true`. Render it **indeterminate until its first report**, which carries
the real denominator; from then on it is an ordinary counted node. Promising a number in the plan
that the run then contradicts would be worse than promising none.

The denominator is **constant within a run** and steps are strictly increasing, so a stepper will
never jump backwards (gated: `--decode-progress-gate` case 4).

⚠️ It is still the slowest node at HD (AB-R-0118 measured the HD peak as decode-bound), and 4 steps
over many minutes is a *coarse* counter. Keep the "still working" affordance — an elapsed timer or
copy setting the expectation — and do not treat 4 steps as a smooth bar.

---

## 5. Rough shape of a run, for pacing intuition

10-second clip at 720p on a 128 GB Mac took **~15 minutes**; at 1080p, **~54 minutes**. Short
previews at small sizes run in **1–3 minutes**.

⚠️ These are single observations on one machine and vary by ±20% or more with heat and load.
**Use them for pacing decisions, never as a countdown shown to the user.**

---

## 6. Correlating events to nodes (for the engineer wiring it up)

Each event carries a phase name, an optional `step`/`totalSteps`, and an optional
`stage`/`totalStages`.

- Match `phase` to the node's `phase`.
- `denoise` appears **twice** on 6-node runs — disambiguate with the event's `stage` (1 or 2)
  against the node's `occurrence`.
- A node with `emitsProgress == false` (**Download model**, **Prepare image adapter**) will never
  receive an event. Drive it from the engine's download progress, or show indeterminate.
- `expectedSubSteps` is `nil` when genuinely unknown up front — **never treat `nil` as zero**.

The plan is gated (`--ltx25-package-gate` cases 53–56) against the measured sequences, so if the
pipeline gains or loses a phase the gate fails rather than every host's stepper quietly
desynchronising.
