/// Re-export so app consumers of `MLXLTX2` reach the enhancer seam without importing `LTX2`
/// directly (BRIDGE-LTX-004 — the app's PromptEnhanceKit `generate:` closure wraps this).
/// `@_exported` (not a typealias) so member/initializer lookup works through `import MLXLTX2`
/// alone — a plain typealias exposes only the type NAME across the module boundary.
@_exported import struct LTX2.GemmaTextGenerator
