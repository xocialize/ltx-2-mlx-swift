import MLXToolKit

/// Typed reads of a `metaData` value. These lived in `LoRARegistry.swift` until the adapters moved
/// onto the engine's typed contract (AB-A-0197); keyframes, retake/extend and the harness still
/// read `metaData` through them.
extension MetaValue {
    /// String payload.
    public var asString: String? { if case .string(let s) = self { return s }; return nil }
    /// Numeric payload as Float (accepts int or double).
    public var asFloat: Float? {
        switch self {
        case .double(let d): return Float(d)
        case .int(let i): return Float(i)
        default: return nil
        }
    }
    /// Integer payload. A `.double` is accepted only when it is exactly integral — a frame index
    /// of 40.5 is a caller bug, and silently truncating it would pin the wrong frame.
    public var asInt: Int? {
        switch self {
        case .int(let i): return i
        case .double(let d): return d == d.rounded() ? Int(d) : nil
        default: return nil
        }
    }
    public var asArray: [MetaValue]? { if case .array(let a) = self { return a }; return nil }
    public var asObject: [String: MetaValue]? { if case .object(let o) = self { return o }; return nil }
}
