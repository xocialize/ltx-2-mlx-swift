# AVFoundation volume probe — refutes "AVFoundation can't read straight off /Volumes/Satechi"

That claim lived in `LTX25-PORT-PLAN.md` §V C3 as an unsourced parenthetical (docs commit
`32dd6a8`) and shaped arm C's prerequisites. **It is false** — see AB-L-0041.

    xcrun swiftc -O main.swift -o avtest
    ./avtest <path-a.mp4> <path-b.mp4>

Loads asset metadata AND decodes every frame through `AVAssetReader` to BGRA, reporting frame
count, reader status, a first-frame checksum, and timings.

## Result (2026-08-15)

Satechi and a local copy both decode 121/121 frames, `status=completed`, **byte-identical
first-frame checksum**. A 1080p 14 s clip decodes cold off Satechi in 240 ms.

⚠️ **Pass the SAME path twice before believing any A/B here.** The first measurement showed
Satechi 137 ms vs local 50 ms — pure first-decode-in-process cost (VideoToolbox/codec init).
Reversing argument order reverses the result; `./avtest X X` reproduces the gap on one file
(123 ms → 50 ms). The volume contributes nothing measurable.
