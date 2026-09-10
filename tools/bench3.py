"""Three engines, one bench, identical footage and identical seeds.

Jeff-Tracker, TAPNext++ (what the bot ships today) and CoTracker3 (the target), run over
the same synthetic occlusion bench, which has EXACT ground truth for every point on every
frame. That is the difference between this and `smoke_ct3_vs_dbt.py`: the smoke test runs
on real plates with no truth and can only report self-consistency, so it can say the two
disagree but never which one is right. Here it can.

What is held equal, deliberately:

  * frames are decoded ONCE and the same array is handed to all three engines
  * seeds are the same Shi-Tomasi corners from frame 1, the same array for all three
  * every engine's output is scaled back to plate pixels and scored by the same scorer,
    from the same ground truth, with the same occlusion mask

What is NOT held equal, and must be read with the table: each engine runs at its own
native resolution, because that is the configuration each one actually ships as.
CoTracker3 resizes internally to 384x512 regardless of what it is fed; Jeff-Tracker runs at
256x256 because 384x680 produced confident 1000 px errors (see FINDINGS.md); TAPNext runs
at whatever BTR_TAPNEXT_IMG resolves to. A resolution-matched pass is a separate
experiment, not this one -- the numbers below say what each product does, not which
architecture is better per pixel.

Scoring is done TWICE for every engine and both rows are printed:

  ungated     every frame the model produced. The honest input for RE-ACQUIRE
  gate 0.50   only frames the model was confident about -- what the 3DE export contains

The pair matters. The export gate hides a track that returns from an occlusion wrong AND
unsure, so reading re-acquisition from the export alone is survivorship: on the LocoTrack
baseline the gate moves the worst re-acquire from 429.55 px to 8.69 px while hiding only
0.6% of the frames. Quoting either number without the other overstates or understates the
thing an artist actually has to clean up.

CoTracker3 is CC-BY-NC-4.0 and is used here as a measurement target only -- see
LICENSES.md. No CoTracker output trains, tunes or selects anything.

    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\bench3.py ^
        --shot bench\\synth\\lab02_occ
    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\bench3.py --all
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
REPO = ROOT
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jefftrack.io import (  # noqa: E402
    list_frames, read_frame, seed_corners, write_3de)
import score_occlusion as SO  # noqa: E402

# CoTracker3 is CC-BY-NC-4.0 and is NOT vendored, downloaded or redistributed by this
# repository -- see docs/LICENSES.md. It is loaded from wherever the user already has it,
# named by environment variable, and is used to produce comparison NUMBERS only. Without
# these set, the cotracker3 engine simply reports that it is unavailable and the other
# engines still run.
CT_CODE = os.environ.get("JT_COTRACKER_DIR", "")
CT_CKPT = os.environ.get("JT_COTRACKER_CKPT", "")

# The locked shot set. Every phase re-runs exactly these, so a number from Phase 3 can be
# put directly beside the same number from Phase 0. Adding a bench is allowed; removing or
# changing one breaks every comparison already recorded, so it is not.
LOCKED_BENCHES = ["lab02_occ", "occ_s11", "occ_s12", "occ_s13"]

DEFAULT_JT_CKPT = os.path.join(ROOT, "weights", "jefftracker.ckpt")


def log(msg):
    print(msg, flush=True)


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- engines
# Every runner returns (tracks (T,N,2) in FED pixels, vis (T,N) bool, conf (T,N) float,
# seconds, peak_gb, res_label). Scaling to plate space happens once, outside.

def run_jefftrack(frames, pts, ckpt, arch, model_res, chunk=64, window=0,
                  anchor_query=True, window_overlap=None):
    from jefftrack.engine import JeffTrackEngine  # noqa: PLC0415
    # None means "whatever the engine's default is", so this file does not carry a second
    # copy of a tuned number that would silently go stale when the engine's moves.
    kw = {} if window_overlap is None else {"window_overlap": int(window_overlap)}
    eng = JeffTrackEngine(device="cuda", model_size="base", ckpt=ckpt,
                          model_res=model_res, arch=arch, query_chunk_size=chunk,
                          window=window, anchor_query=anchor_query, **kw)
    free_vram()
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)
    t0 = time.time()
    tracks, vis, conf = eng.track_queries_conf(frames, q)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if eng.device == "cuda" else 0.0
    del eng
    free_vram()
    return (tracks, vis.astype(bool), conf, dt, peak,
            "{}x{} chunk {}".format(model_res[0], model_res[1], chunk))


def run_tapnext(frames, pts):
    """The bot's own GPU backend, so the comparison includes the incumbent.

    TAPNext returns visibility but no continuous confidence, so conf is the boolean.
    Stated rather than papered over: any confidence-threshold row for this engine is a
    yes/no, and its 'ungated' and 'gate 0.50' rows are therefore identical by construction.
    """
    # TAPNext++ likewise is not vendored. It is Apache-2.0, so there is no licence reason
    # not to -- it is simply a different model and out of scope for this repository.
    tn_root = os.environ.get("JT_TAPNEXT_ROOT", "")
    if not tn_root:
        raise RuntimeError(
            "TAPNext++ is not set up here. Point JT_TAPNEXT_ROOT at a tree containing the "
            "google-deepmind/tapnet checkout and its tapnextpp checkpoint to include it.")
    sys.path.insert(0, tn_root)
    from tapnext_engine import TapNextEngine  # noqa: PLC0415
    eng = TapNextEngine(tool_root=tn_root, device="cuda")
    free_vram()
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)[None]
    t0 = time.time()
    tracks, vis = eng.track_queries(frames, q)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if eng.device == "cuda" else 0.0
    res = "{}x{}".format(eng.img, eng.img)
    del eng
    free_vram()
    vis = np.asarray(vis).astype(bool)
    return tracks, vis, vis.astype(np.float32), dt, peak, res


def run_cotracker3(frames, pts, max_side=1024):
    """CoTracker3 offline through its own predictor, loaded from JT_COTRACKER_DIR.

    Query frame 0, so backward_tracking has nothing to add and is left off. The predictor
    resizes internally to 384x512 whatever it is fed, so `max_side` only bounds what is
    uploaded to the GPU, not what the model sees.
    """
    if not CT_CODE or not os.path.isdir(CT_CODE):
        raise RuntimeError(
            "CoTracker3 is not set up here, by design -- it is not vendored. To include it "
            "in the comparison, point JT_COTRACKER_DIR at a checkout of facebookresearch/"
            "co-tracker and JT_COTRACKER_CKPT at its scaled_offline checkpoint. Note the "
            "CC-BY-NC-4.0 licence on both (docs/LICENSES.md).")
    if not os.path.isfile(CT_CKPT):
        raise RuntimeError("JT_COTRACKER_CKPT does not point at a file: {}".format(CT_CKPT))
    if CT_CODE not in sys.path:
        sys.path.insert(0, CT_CODE)
    from cotracker.predictor import CoTrackerPredictor  # noqa: PLC0415

    T, H, W = frames.shape[:3]
    scale = min(1.0, float(max_side) / float(max(W, H)))
    w, h = int(round(W * scale)), int(round(H * scale))

    model = CoTrackerPredictor(checkpoint=CT_CKPT, offline=True, v2=False, window_len=60)
    model = model.cuda().eval()
    free_vram()

    buf = np.empty((T, h, w, 3), np.uint8)
    for i, fr in enumerate(frames):
        small = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA) if scale != 1.0 else fr
        buf[i] = small[:, :, ::-1]
    video = torch.from_numpy(buf).permute(0, 3, 1, 2)[None].float().cuda()

    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts * scale],
                       1).astype(np.float32)
    queries = torch.from_numpy(q)[None].cuda()
    t0 = time.time()
    with torch.no_grad():
        tr, vs = model(video, queries=queries, backward_tracking=False)
    dt = time.time() - t0
    tracks = tr[0].detach().cpu().numpy() / scale          # back to fed pixels
    vis = vs[0].detach().cpu().numpy().astype(bool)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, video, queries, tr, vs
    free_vram()
    # CoTracker's visibilities are already thresholded at 0.9 inside the predictor, so
    # there is no continuous confidence to hand back. conf is the boolean, stated as such.
    return tracks, vis, vis.astype(np.float32), dt, peak, "384x512 (internal)"


def parse_engine(token, default_res, default_chunk):
    """`name` or `name/res=384x512,chunk=256`.

    A configuration is its own series on the sheet rather than a flag hidden in a caption:
    Phase 1 exists to find out whether the Phase 0 gap was the model or the way it was
    being RUN, and that question is unanswerable if two configs share a label.
    """
    name, _, rest = token.partition("/")
    opts = {}
    # `;` separates options, NOT `,` -- the outer --engines list already owns the comma,
    # and using it here silently swallowed "chunk=256" into the res string.
    for kv in rest.replace(",", ";").split(";"):
        if not kv.strip():
            continue
        k, _, v = kv.partition("=")
        opts[k.strip()] = v.strip()
    res = opts.get("res", default_res)
    chunk = int(opts.get("chunk", default_chunk))
    res_t = tuple(int(v) for v in res.lower().split("x"))
    # Windowing is part of a configuration's identity for the same reason the resolution
    # is: on a clip longer than the window budget it changes the result more than any
    # other single setting, and two windowings sharing a label is how a seam artifact gets
    # read as a model property.
    win = int(opts.get("win", 0))
    overlap = int(opts["ov"]) if "ov" in opts else None
    anchor = str(opts.get("anchor", "1")).lower() not in ("0", "false", "no")
    label = name
    if "res" in opts:
        label += "_r{}x{}".format(*res_t)
    if "chunk" in opts:
        label += "_c{}".format(chunk)
    if "win" in opts:
        label += "_w{}".format(win)
    if "ov" in opts:
        label += "_ov{}".format(overlap)
    if not anchor:
        label += "_noanchor"
    return {"name": name.strip(), "res": res_t, "chunk": chunk, "label": label,
            "win": win, "ov": overlap, "anchor": anchor}


# --------------------------------------------------------------------------- driver
def run_bench(shot_dir, engines, points, work_width, jt_ckpt, jt_arch, jt_res, out_dir):
    name = os.path.basename(shot_dir.rstrip("/\\"))
    plate = os.path.join(shot_dir, "plate")
    files, first_frame = list_frames(plate, 1, 0)

    frames, plate_wh = [], None
    for f in files:
        img, wh = read_frame(f, work_width)
        plate_wh = wh
        frames.append(img)
    frames = np.stack(frames)
    T, Hw, Ww = frames.shape[:3]
    plate_w, plate_h = plate_wh
    sx, sy = plate_w / float(Ww), plate_h / float(Hw)

    pts = seed_corners(frames[0], points)
    log("\n{}\n{}  {} frames  plate {}x{}  fed {}x{}  {} seeds\n{}".format(
        "=" * 78, name, T, plate_w, plate_h, Ww, Hw, len(pts), "=" * 78))

    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for spec in engines:
        eng_name, label = spec["name"], spec["label"]
        log("\n[{}] running...".format(label))
        try:
            if eng_name == "jefftrack":
                out = run_jefftrack(frames, pts, jt_ckpt, jt_arch,
                                    spec["res"], spec["chunk"],
                                    window=spec.get("win", 0),
                                    anchor_query=spec.get("anchor", True),
                                    window_overlap=spec.get("ov"))
            elif eng_name == "tapnext":
                out = run_tapnext(frames, pts)
            elif eng_name == "cotracker3":
                out = run_cotracker3(frames, pts)
            else:
                raise SystemExit("[ERROR] unknown engine {}".format(eng_name))
        except Exception as exc:  # noqa: BLE001
            # One engine failing must not discard the other two -- the same reason the
            # backend loaders in app.py capture their import errors instead of raising.
            log("[{}] FAILED: {!r}".format(label, exc))
            results[label] = {"error": repr(exc)}
            free_vram()
            continue

        tracks, vis, conf, dt, peak, res = out
        tracks_plate = tracks.copy().astype(np.float32)
        tracks_plate[..., 0] *= sx
        tracks_plate[..., 1] *= sy

        base = os.path.join(out_dir, "{}__{}".format(name, label))
        np.savez_compressed(base + ".npz", tracks=tracks_plate, visibility=vis,
                            confidence=conf, first_frame=first_frame,
                            plate_size=np.array([plate_w, plate_h]))
        write_3de(base + ".txt", tracks_plate, vis, first_frame, plate_h)
        log("[{}] {:.1f}s  {:.3f}s/frame  peak {:.2f} GB  model res {}  visible {:.1f}%"
            .format(label, dt, dt / T, peak, res, 100.0 * vis.mean()))
        results[label] = {"npz": base + ".npz", "seconds": dt, "s_per_frame": dt / T,
                             "peak_gb": peak, "model_res": res,
                             "visible_pct": 100.0 * float(vis.mean())}
    return name, results, first_frame


def score_all(shot_dir, name, results, reacq):
    """Score every engine ungated and at the export gate, and collect the rows."""
    rows = []
    for eng_name, info in results.items():
        if "npz" not in info:
            continue
        for gate, label in ((0.0, "ungated"), (0.5, "gate 0.50")):
            seq, ff = SO.seq_from_npz(info["npz"], gate)
            log("\n--- {} / {} / {} ---".format(name, eng_name, label))
            m = SO.score(shot_dir, seq, reacq, ff, control=False,
                         source="[{} {}]".format(eng_name, label))
            rows.append({"bench": name, "engine": eng_name, "gate": label,
                         "metrics": m if isinstance(m, dict) else None})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Jeff-Tracker vs TAPNext++ vs CoTracker3 on a truth bench")
    ap.add_argument("--shot", default=None, help="a bench dir under bench/synth")
    ap.add_argument("--all", action="store_true",
                    help="run the locked bench set: " + ", ".join(LOCKED_BENCHES))
    ap.add_argument("--engines", default="jefftrack,tapnext,cotracker3")
    ap.add_argument("--points", type=int, default=600)
    ap.add_argument("--work-width", type=int, default=1280)
    ap.add_argument("--ckpt", default=DEFAULT_JT_CKPT)
    ap.add_argument("--arch", default="jefftrack", choices=["locotrack", "jefftrack"])
    ap.add_argument("--model-res", default="256x256")
    ap.add_argument("--chunk", type=int, default=64,
                    help="query_chunk_size. The cross-track block can only "
                         "attend across tracks that SHARE a chunk, and "
                         "training ran at 256 while inference defaults to 64 "
                         "-- so the block is deployed seeing a quarter of the "
                         "neighbours it was trained on.")
    ap.add_argument("--reacq", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "bench3"))
    ap.add_argument("--tag", default="", help="suffix for the summary json, e.g. phase0")
    a = ap.parse_args()

    engines = [parse_engine(e.strip(), a.model_res, a.chunk)
               for e in a.engines.split(",") if e.strip()]
    jt_res = tuple(int(v) for v in a.model_res.lower().split("x"))

    if a.all:
        shots = [os.path.join(REPO, "bench", "synth", b) for b in LOCKED_BENCHES]
    elif a.shot:
        shots = [a.shot if os.path.isabs(a.shot) else os.path.join(REPO, a.shot)]
    else:
        ap.error("--shot or --all is required")

    missing = [s for s in shots if not os.path.isdir(os.path.join(s, "plate"))]
    if missing:
        raise SystemExit(
            "[ERROR] bench missing (bench/synth is gitignored -- rebuild it):\n  "
            + "\n  ".join(missing)
            + "\n  runtime\\python311\\python.exe experiments\\Jeff-Tracker\\"
              "make_occlusion_bench.py --src bench\\synth\\lab02 --out <dir> "
              "--occluders 8 --seed <n>")

    all_rows = []
    for shot_dir in shots:
        name, results, _ = run_bench(shot_dir, engines, a.points, a.work_width,
                                     a.ckpt, a.arch, jt_res, a.out)
        all_rows.extend(score_all(shot_dir, name, results, a.reacq))
        # MERGE, never replace. A run that names only some engines must not delete the
        # cost rows of the ones it did not run -- the first Phase 1 pass did exactly that
        # and erased every Phase 0 timing, which the score sheet reads.
        cost_path = os.path.join(a.out, "{}_cost.json".format(name))
        summary = {"bench": name, "engines": {}}
        if os.path.isfile(cost_path):
            try:
                with open(cost_path) as fh:
                    summary["engines"] = json.load(fh).get("engines", {})
            except (OSError, ValueError):
                summary["engines"] = {}
        summary["engines"].update({k: {kk: vv for kk, vv in v.items() if kk != "npz"}
                                   for k, v in results.items()})
        with open(cost_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    tag = ("_" + a.tag) if a.tag else ""
    path = os.path.join(a.out, "bench3_rows{}.json".format(tag))
    with open(path, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    log("\n[done] rows -> {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
