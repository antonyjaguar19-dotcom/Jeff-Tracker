"""Score one checkpoint on BOTH benches and append a row to out/ckpt_scores.tsv.

Kept as one command on purpose. The step-2000 checkpoint improved the occlusion bench and
was flat-to-worse on DAVIS occlusion accuracy at the same time, which is exactly the pair
of facts that goes missing when the two benches are run separately and only the
better-looking one gets written down. Running them together makes it awkward to report one
without the other, which is the point.

    python score_ckpt.py ^
        --ckpt weights\\step3000.ckpt --tag s3000

Columns: the occlusion bench (visible / occluded / re-acquire, plus the coverage figure
without which the occluded number means nothing) and TAP-Vid DAVIS strided (AJ / d_avg /
OA). The baseline row for LocoTrack-B is in METHOD.md; --arch locotrack reproduces it.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
REPO = ROOT
PY = sys.executable
TSV = os.path.join(ROOT, "out", "ckpt_scores.tsv")


def run(cmd):
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit("[ERROR] {} failed".format(" ".join(cmd[1:3])))
    return p.stdout


def ungated_occluded(shot, npz_path):
    """Occluded-frame error over EVERY frame, ignoring the export's visibility gate.

    The exported occluded rows describe what an artist receives; this describes what the
    model actually knows. Reporting only the first led me to call LocoTrack's occlusion
    coverage 5.9% when the model emits a position on every frame -- see METHOD.md.
    """
    sys.path.insert(0, HERE)
    from score_occlusion import load_shot, warp, is_occluded
    _, Hs, occ = load_shot(shot)
    d = np.load(npz_path)
    tr = d["tracks"]
    T = min(len(Hs), tr.shape[0])
    src = warp(np.linalg.inv(Hs[0]), tr[0].astype(np.float64))
    err = np.zeros((T, tr.shape[1]))
    ocl = np.zeros((T, tr.shape[1]), bool)
    for t in range(T):
        g = warp(Hs[t], src)
        err[t] = np.hypot(tr[t, :, 0] - g[:, 0], tr[t, :, 1] - g[:, 1])
        ocl[t] = is_occluded(occ, t, g)
    e = err[ocl]
    return (float(np.median(e)), 100.0 * float((e < 5).mean())) if e.size else (
        float("nan"), float("nan"))


def grab(text, label):
    """Pull 'mean X med Y p95 Z' off one line of score_occlusion output."""
    m = re.search(label + r"\s+mean\s+([\d.]+)\s+med\s+([\d.]+)", text)
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"),) * 2


def main() -> int:
    ap = argparse.ArgumentParser(description="score a checkpoint on both benches")
    ap.add_argument("--ckpt", default=None, help="omit with --arch locotrack for baseline")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arch", default="dbtrack", choices=["locotrack", "dbtrack"])
    ap.add_argument("--shot", default=os.path.join(REPO, "bench", "synth", "lab02_occ"))
    ap.add_argument("--model-res", default="256x256")
    ap.add_argument("--skip-tapvid", action="store_true")
    a = ap.parse_args()

    name = "lab02_occ_" + a.tag
    track = [PY, os.path.join(HERE, "run_dbtrack.py"),
             "--plate", os.path.join(a.shot, "plate"), "--name", name,
             "--arch", a.arch, "--seed", "corners", "--points", "600",
             "--model-res", a.model_res, "--work-width", "1280", "--no-render"]
    if a.ckpt:
        track += ["--ckpt", a.ckpt]
    run(track)

    occ = run([PY, os.path.join(HERE, "score_occlusion.py"), "--shot", a.shot,
               "--bot", os.path.join(HERE, "out", name + "__dbtrack.txt")])
    vis_m, vis_md = grab(occ, "VISIBLE")
    occ_m, _ = grab(occ, "OCCLUDED")
    rea_m, _ = grab(occ, "RE-ACQUIRE")
    cov = re.search(r"coverage ([\d.]+)% of occluded", occ)
    cov = float(cov.group(1)) if cov else float("nan")
    ung_med, ung_pct = ungated_occluded(
        a.shot, os.path.join(HERE, "out", name + "__dbtrack.npz"))

    aj = da = oa = float("nan")
    if not a.skip_tapvid:
        tv = [PY, os.path.join(HERE, "eval_tapvid.py"), "--mode", "strided",
              "--arch", a.arch, "--model-res", a.model_res]
        if a.ckpt:
            tv += ["--ckpt", a.ckpt]
        out = run(tv)
        def m(k):
            mm = re.search(r"^" + k + r"\s+([\d.]+)", out, re.M)
            return float(mm.group(1)) if mm else float("nan")
        aj, da, oa = m("AJ"), m("delta_avg"), m("OA")

    hdr = ("tag\tvisible_mean\tvisible_med\tocc_ungated_med\tocc_ungated_pct5"
           "\toccluded_mean\tcoverage%\treacquire_mean\tAJ\tdelta_avg\tOA")
    row = ("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.1f}\t{:.3f}\t{:.1f}\t{:.3f}"
           "\t{:.1f}\t{:.1f}\t{:.1f}").format(
        a.tag, vis_m, vis_md, ung_med, ung_pct, occ_m, cov, rea_m, aj, da, oa)
    os.makedirs(os.path.dirname(TSV), exist_ok=True)
    new = not os.path.isfile(TSV)
    with open(TSV, "a") as fh:
        if new:
            fh.write(hdr + "\n")
        fh.write(row + "\n")

    print(hdr.replace("\t", "  "))
    print(row.replace("\t", "  "))
    print()
    print("appended to {}".format(TSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
