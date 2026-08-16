/// Re-export so app consumers of `MLXLTX2` reach the enhancer seam without importing `LTX2`
/// directly (BRIDGE-LTX-004 — the app's PromptEnhanceKit `generate:` closure wraps this).
/// `@_exported` (not a typealias) so member/initializer lookup works through `import MLXLTX2`
/// alone — a plain typealias exposes only the type NAME across the module boundary.
///
/// Nothing to widen here for the enhancement seed (AB-R-0075): `@_exported` re-exports the whole
/// type, so `generate(system:user:maxTokens:temperature:seed:)` reaches app consumers as-is. And
/// there is deliberately no `metaData` key for it — enhancement is NOT reachable from a
/// `T2VRequest`; `MLXLTX2Package.run` only ever generates video, and the app calls this seam
/// directly (or, engine-hosted, `GemmaLLMPackage` — BRIDGE-LTX-006). A caller that wants one number
/// to reproduce a whole run passes the same seed to `generate(...)` and to `t2v(...)`; see the
/// decision recorded on `GemmaTextGenerator.generate`.
@_exported import struct LTX2.GemmaTextGenerator
