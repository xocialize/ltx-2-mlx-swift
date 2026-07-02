import LTX2

/// Re-export so app consumers of `MLXLTX2` reach the enhancer seam without importing `LTX2`
/// directly (BRIDGE-LTX-004 — the app's PromptEnhanceKit `generate:` closure wraps this).
public typealias GemmaTextGenerator = LTX2.GemmaTextGenerator
