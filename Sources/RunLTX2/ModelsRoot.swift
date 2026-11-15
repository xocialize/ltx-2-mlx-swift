// ModelsRoot.swift — where the harness looks for weights on this machine.
//
// Every gate, probe and smoke in RunLTX2 builds its weight paths from `ltxModelsRoot`. Before this
// file the root was a literal in 27 files, so when the archive moved volumes the board stopped
// finding its weights, one gate at a time, each with its own "no such file".
//
// Resolution, first hit wins:
//   1. `LTX_MODELS_ROOT` — explicit, never second-guessed (a secondary station's staging copy).
//   2. The first known root that holds an LTX tree (`xocialize/`, `dgrauet/` or `ltx-granules-25/`).
//   3. The first known root, so a missing archive fails on a path that names it.
//
// The layout under the root is the fleet archive's: `xocialize/ltx-2.5-mlx{,-ditq8,-q8}`,
// `dgrauet/ltx-2.3-mlx{,-q8,-q4}`, `mlx-community/gemma-3-12b-it-4bit`, `ltx-granules{,-25}/<quant>`,
// `ltx-lora-cache/`, `_smoke/`, `_ForgeSmokeCorpus/`.

import Foundation

/// Known archive roots, newest first.
private let ltxKnownModelsRoots = ["/Volumes/DEV_VOL1/Models/Models", "/Volumes/Satechi/Models"]

/// The models root every RunLTX2 harness resolves weight paths against. See the file header.
let ltxModelsRoot: String = {
    if let explicit = ProcessInfo.processInfo.environment["LTX_MODELS_ROOT"], !explicit.isEmpty {
        return explicit
    }
    let fm = FileManager.default
    let markers = ["xocialize", "dgrauet", "ltx-granules-25"]
    return ltxKnownModelsRoots.first { root in
        markers.contains { fm.fileExists(atPath: "\(root)/\($0)") }
    } ?? ltxKnownModelsRoots[0]
}()
