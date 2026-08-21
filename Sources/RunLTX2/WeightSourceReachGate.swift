// WeightSourceReachGate.swift — `--weight-sources-reach-gate`, the check that would have caught it.
//
// 🚨 **WHY THIS EXISTS.** `--ltx25-package-gate` asserts that repo NAMES resolve correctly — that
// int8 derives `-ditq8`, that the encoder sibling is `-q8`, that coercion preserves a custom repo.
// All true, all green, and **none of it can tell you whether the repo EXISTS.** LTX-2.5's shipping
// default was `xocialize/ltx-2.5-mlx` for months; that repo is a 404 (verified with a valid token,
// so "absent", not "private"). The real tree is `mlx-community/ltx-2.5-mlx`. Every consumer without
// local weights would have failed on first run, and no gate could see it.
//
// **Resolution and reachability are different kinds of test.** We had the first and assumed it
// covered the second.
//
// ⚠️ NETWORK-DEPENDENT — deliberately NOT on the CI board. Run it before a release, after touching
// any repo name, and whenever a `WeightSource` is added. An offline failure is not a defect here.
//
// usage: RunLTX2 --weight-sources-reach-gate

import Foundation
import MLXLTX2
import MLXToolKit

func weightSourceReachGate() async throws {
    // The shipping matrices: every family × quant a user can actually register.
    var configs: [(String, LTX2Configuration)] = []
    for (family, label) in [(LTXFamily.ltx23, "2.3"), (.ltx25, "2.5")] {
        for quant in [Quant.bf16, .int8, .int4] {
            var c = LTX2Configuration(family: family)
            c.quant = quant
            configs.append(("\(label)/\(quant.rawValue)", c))
            // …and the streamed variant, which swaps the transformer source for granules.
            var s = c
            s.streamedBlocks = true
            configs.append(("\(label)/\(quant.rawValue)+streamed", s))
            // …and the int8-encoder sibling, a DIFFERENT components repo.
            var e = c
            e.textEncoderQuant = .int8
            configs.append(("\(label)/\(quant.rawValue)+enc-int8", e))
        }
    }

    var seen: [String: Bool] = [:]
    var failures: [String] = []
    func reachable(_ repo: String) async -> Bool {
        if let c = seen[repo] { return c }
        var req = URLRequest(url: URL(string: "https://huggingface.co/api/models/\(repo)")!)
        req.timeoutInterval = 20
        // Public reachability is what a consumer gets, so this is deliberately UNauthenticated:
        // a repo that resolves only with our token is not published as far as users are concerned.
        let ok: Bool
        do {
            let (_, resp) = try await URLSession.shared.data(for: req)
            ok = (resp as? HTTPURLResponse)?.statusCode == 200
        } catch { ok = false }
        seen[repo] = ok
        return ok
    }

    print("[reach-gate] checking every repo the shipping configurations declare (unauthenticated)")
    var repoToArms: [String: Set<String>] = [:]
    for (label, cfg) in configs {
        for src in cfg.weightSources {
            repoToArms[src.repo, default: []].insert("\(label):\(src.role)")
        }
    }
    for repo in repoToArms.keys.sorted() {
        let ok = await reachable(repo)
        let arms = repoToArms[repo]!.sorted().prefix(3).joined(separator: ", ")
        print("  \(ok ? "✅" : "❌") \(repo)  ← \(arms)\(repoToArms[repo]!.count > 3 ? " …" : "")")
        if !ok { failures.append(repo) }
    }

    if failures.isEmpty {
        print("[reach-gate] PASS ✅ — every declared WeightSource is publicly reachable")
    } else {
        print("[reach-gate] FAIL ❌ \(failures.count) unreachable: \(failures.joined(separator: ", "))")
        print("[reach-gate] ⚠️ a consumer without local weights CANNOT materialize these arms")
        exit(1)
    }
}
