import json, glob, os, collections
base = "out/bench3"
arm = {"c5": collections.defaultdict(dict), "c8": collections.defaultdict(dict)}
for f in glob.glob(os.path.join(base, "bench3_rows_c*.json")):
    tag = os.path.basename(f)[len("bench3_rows_"):-5]
    a = "c5" if tag.startswith("c5") else "c8" if tag.startswith("c8") else None
    if a is None: continue
    ck = "s20000" if "final" in tag else tag.split("_")[1]
    for r in json.load(open(f)):
        if r["gate"] != "ungated": continue
        m = r["metrics"]
        arm[a][r["bench"]][ck] = (m["visible"]["mean"], m["occluded"]["mean"],
                                  m["occluded"]["med"], m["reacquire"]["mean"])
names = ["visible", "occl mean", "occl med", "re-acq"]
order = ["lab02_occ", "occ_s11", "occ_s12", "occ_s13", "lab02_depth"]
print("=" * 82)
print("OVERLAP TEST -- 3 late checkpoints per arm, BOTH measured this session, ungated")
print("=" * 82)
tally = collections.Counter()
for b in order:
    if b not in arm["c5"] or b not in arm["c8"]: continue
    print("\n  %s" % b)
    for i, n in enumerate(names):
        c5v = [arm["c5"][b][k][i] for k in sorted(arm["c5"][b])]
        c8v = [arm["c8"][b][k][i] for k in sorted(arm["c8"][b])]
        if not c5v or not c8v: continue
        c5lo, c5hi, c8lo, c8hi = min(c5v), max(c5v), min(c8v), max(c8v)
        if c8hi < c5lo:   v, key = "c8 BETTER, separated", "better"
        elif c8lo > c5hi: v, key = "c8 WORSE, separated", "worse"
        else:             v, key = "overlap", "overlap"
        tally[key] += 1
        print("    %-10s c5 %6.3f-%6.3f (n=%d)   c8 %6.3f-%6.3f (n=%d)   %s"
              % (n, c5lo, c5hi, len(c5v), c8lo, c8hi, len(c8v), v))
print("\n" + "=" * 82)
print("  separated BETTER : %d" % tally["better"])
print("  separated WORSE  : %d" % tally["worse"])
print("  overlap          : %d" % tally["overlap"])
print("  total tested     : %d" % sum(tally.values()))
