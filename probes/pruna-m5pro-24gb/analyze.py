#!/usr/bin/env python3
"""run.log -> per-run table, per-arm stats, MEASURABLE/NOISE verdict (bench-e2e doctrine), worst-case DECLARE, clip checks."""
import re, sys, statistics, subprocess, pathlib, itertools
log = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "run.log").read_text().splitlines()
runs, cur = [], None
for line in log:
    if line.startswith("=== DONE"): break
    if line.startswith("=== "):
        cur = {"name": line[4:].strip()}; runs.append(cur); continue
    if cur is None: continue
    m = re.search(r"\] run ([\d.]+)s", line)
    if m: cur["run_s"] = float(m.group(1))
    m = re.search(r"resident\(held\)=([\d.]+) GB act=([\d.]+) GB peak=([\d.]+) GB", line)
    if m: cur["res"], cur["act"], cur["peak"] = map(float, m.groups())
    m = re.search(r"^\s*([\d.]+) real", line)
    if m: cur["real_s"] = float(m.group(1))
    m = re.search(r"^\s*(\d+)\s+peak memory footprint", line)
    if m: cur["foot_gb"] = int(m.group(1)) / 1e9
    m = re.search(r"swap (before|after):.*used = ([\d.]+)M", line)
    if m: cur["swap_" + m.group(1)] = float(m.group(2)) / 1024
    m = re.search(r"ACCEPTANCE .*→ (✅ WITHIN|❌ OVER)", line)
    if m: cur["verdict"] = m.group(1)
    m = re.search(r"^exit=(\d+)", line)
    if m: cur["exit"] = int(m.group(1))
def kind(n):
    if n.startswith("burn-in"): return "burn-in"
    if "warmup" in n: return "warm-up"
    if n.startswith("diag"): return "diagnostic"
    return "measured"
def arm(n): return "pruna" if " pruna " in f" {n} " else "stock"
print(f"{'run':34} {'kind':10} {'gen s':>7} {'wall s':>7} {'res':>5} {'act':>6} {'peak':>6} {'foot':>6} {'swapΔ':>6} verdict")
for r in runs:
    if "run_s" not in r: print(f"{r['name']:34} {kind(r['name']):10} (incomplete)"); continue
    sd = r.get("swap_after", 0) - r.get("swap_before", 0)
    print(f"{r['name']:34} {kind(r['name']):10} {r['run_s']:7.1f} {r.get('real_s',0):7.1f} {r.get('res',0):5.2f} {r.get('act',0):6.2f} {r.get('peak',0):6.2f} {r.get('foot_gb',0):6.2f} {sd:+6.2f} {r.get('verdict','?')} exit={r.get('exit','?')}")
def stats(rows, label):
    t = [r["run_s"] for r in rows]
    if not t: return None
    med = statistics.median(t); lo, hi = min(t), max(t)
    print(f"  {label:8} n={len(t)} median {med:6.1f} s  min–max {lo:.1f}–{hi:.1f} (range {hi-lo:.1f})  "
          f"peak max {max(r['peak'] for r in rows):.2f} GB  resident max {max(r['res'] for r in rows):.2f}  act max {max(r['act'] for r in rows):.2f}  "
          f"footprint max {max(r.get('foot_gb',0) for r in rows):.2f} GB")
    return med, hi - lo
for k in ("measured", "diagnostic"):
    rows = [r for r in runs if kind(r["name"]) == k and "run_s" in r]
    if not rows: continue
    print(f"\n{k.upper()} ({rows[0]['name'].split()[1 if k=='diagnostic' else 1]})")
    s = stats([r for r in rows if arm(r["name"]) == "stock"], "stock")
    p = stats([r for r in rows if arm(r["name"]) == "pruna"], "pruna")
    if s and p:
        d = p[0] - s[0]; bar = s[1]
        print(f"  Δmedian pruna−stock = {d:+.1f} s ({d/s[0]*100:+.1f} %) vs stock's within-arm range {bar:.1f} s → "
              f"{'MEASURABLE' if abs(d) > bar else 'NOISE'} (bench-e2e doctrine: |Δmed| must exceed the baseline arm's spread)")
# clip checks: within-arm identical? across arms different?
logs = pathlib.Path("logs")
def same(a, b): return subprocess.run(["cmp", "-s", str(a), str(b)]).returncode == 0
for k in ("measured", "diagnostic"):
    rows = [r for r in runs if kind(r["name"]) == k and "run_s" in r]
    clips = {r["name"]: logs / (r["name"].replace(" ", "_") + ".mp4") for r in rows}
    clips = {n: p for n, p in clips.items() if p.exists()}
    if len(clips) < 2: continue
    by = {"stock": [p for n, p in clips.items() if arm(n) == "stock"], "pruna": [p for n, p in clips.items() if arm(n) == "pruna"]}
    for a, ps in by.items():
        pairs = list(itertools.combinations(ps, 2))
        if pairs: print(f"  clips {k} {a}: {sum(same(x,y) for x,y in pairs)}/{len(pairs)} within-arm pairs byte-identical")
    cross = [(x, y) for x in by["stock"] for y in by["pruna"]]
    if cross: print(f"  clips {k} stock×pruna: {sum(same(x,y) for x,y in cross)}/{len(cross)} pairs identical (0 expected — decoders differ)")
