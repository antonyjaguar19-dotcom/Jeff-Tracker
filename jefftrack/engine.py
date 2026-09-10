"""Jeff-Tracker engine -- LocoTrack (Apache-2.0) behind a stable engine interface.

Why this exists: most of the strongest open point trackers ship under NonCommercial
terms on both code and weights, which rules them out of a commercial pipeline. LocoTrack
(cvlab-kaist, ECCV 2024) is Apache-2.0 end to end, so it is the base here. See LICENSES.md
for the full provenance ledger.

The public surface is deliberately a drop-in shape for an existing TAPNext-style engine:

    track_queries(frames_bgr, queries, fp16=False) -> (tracks (T,N,2) xy, vis (T,N) bool)
    track_grid(frames_bgr, grid_size, grid_query_frame=0, segm_mask=None) -> same

so Stage-1 work survives if Jeff-Tracker is later wired into a host pipeline. Nothing
outside this repository is touched by this file.

One thing is deliberately NOT the same: track_queries_conf() also returns a continuous
per-frame confidence. A TAPNext-style engine collapses its visibility logits to a bool
and the continuous value is lost before the caller sees it, which then has to be
reconstructed photometrically from pixels at lower quality. LocoTrack hands it over
directly, so this engine keeps it.

Self-check (no plate needed, falls back to CPU if there is no GPU):

    python -m jefftrack.engine --selftest
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
# engine.py lives inside the package, so the repo root -- which is what holds
# vendor/, weights/ and pydeps/ -- is one level up.
ROOT = os.path.dirname(HERE)
PYDEPS = os.environ.get("JEFFTRACK_PYDEPS", os.path.join(ROOT, "pydeps"))
VENDOR_PT = os.path.join(ROOT, "vendor", "locotrack", "locotrack_pytorch")
# pydeps first: the embeddable runtime is in isolated mode, so PYTHONPATH is ignored and
# the only way these land on the path is from inside the process. Same reason and same
# order as experiments/track_on/run_trackon.py:36-43.
for _p in (PYDEPS, VENDOR_PT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_CKPT = os.environ.get(
    "JEFFTRACK_CKPT", os.path.join(ROOT, "weights", "locotrack_base.ckpt"))

# LocoTrack was trained at 256x256 (LocoTrack.initial_resolution). Feeding it a larger
# video does not just upscale -- get_feature_grids() infers a LADDER of refinement
# resolutions from the input size (locotrack_model.py:549 -> utils.generate_default_
# resolutions), so a bigger model_res buys real sub-pixel accuracy at real VRAM cost.
# 256x256 is the safe default; 384x512 is the high-accuracy setting the LocoTrack paper
# reports, at real VRAM cost.
DEFAULT_MODEL_RES = (256, 256)


def _load_jefftrack(ckpt_path: str, model_size: str, device: str, **kw):
    """LocoTrack plus cross-track attention (see jefftrack/model/jefftrack_model.py).

    Zero-initialised, this is LocoTrack bit for bit -- check_identity.py proves it -- so
    running the whole bench through arch='jefftrack' on an untrained checkpoint must
    reproduce the arch='locotrack' numbers exactly. That is a pipeline-level identity
    check, not just a model-level one.
    """
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from jefftrack.model.jefftrack_model import load_jefftrack  # type: ignore
    return load_jefftrack(ckpt_path, model_size=model_size, device=device, **kw)


def _load_locotrack(ckpt_path: str, model_size: str, device: str):
    """Build the vendor model. Kept separate so an import failure is legible."""
    try:
        from models.locotrack_model import LocoTrack  # type: ignore
    except Exception as exc:  # surfaced with the path, not raised blind
        raise SystemExit(
            "[ERROR] cannot import LocoTrack from {}: {!r}\n"
            "        vendor/ is gitignored -- see LICENSES.md for how it is populated."
            .format(VENDOR_PT, exc)
        )
    model = LocoTrack(model_size=model_size)
    # The published checkpoints are Lightning checkpoints: real weights live under
    # 'state_dict' with every key prefixed 'model.'. Mirrors the vendor's load_model().
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["state_dict"] if "state_dict" in blob else blob
    state = {k.replace("model.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            "[ERROR] checkpoint does not fit LocoTrack(model_size={!r}): "
            "{} missing, {} unexpected. first missing={} first unexpected={}"
            .format(model_size, len(missing), len(unexpected),
                    missing[:3], unexpected[:3])
        )
    return model.to(device).eval()


class JeffTrackEngine:
    """LocoTrack, fed and read in raster coordinate conventions.

    Coordinates in:  queries are [frame, x, y] in the pixel space of the frames handed in.
    Coordinates out: tracks are [x, y] in that same pixel space, y DOWN (OpenCV raster).
                     The 3DE y-flip belongs to the exporter, not here.

    Windowing is a VRAM measure, not part of the algorithm -- but the way a window is
    re-queried decides whether a track keeps its IDENTITY across the seam, and getting that
    wrong costs more than any other setting in this file. See track_queries_conf. Measured
    on the occlusion bench at window 40, against ground truth, ungated:

        no windowing        visible 1.299   occluded   3.186   re-acquire   1.558 px
        anchored (default)  visible 1.304   occluded   3.426   re-acquire   1.528 px
        previous re-query   visible 199.6   occluded 197.9     re-acquire 278.2  px

    A clip short enough to fit one window never reaches that code, so those three rows are
    one row on every bench in this repo -- which is why it went unmeasured until a
    261-frame plate was rendered and watched.
    """

    def __init__(
        self,
        tool_root: Optional[str] = None,
        device: str = "cuda",
        model_size: str = "base",
        ckpt: Optional[str] = None,
        model_res: Tuple[int, int] = DEFAULT_MODEL_RES,
        query_chunk_size: int = 64,
        window: int = 0,
        # Raised 8 -> 32 with the anchored re-query. The overlap frames are taken from the
        # EARLIER window, where the track had a full run-up behind it, so the overlap is
        # what a window start gets instead of context. Once anchoring stopped the seam
        # destroying track identity, that context became the whole remaining gap: measured
        # on a 261-frame plate at window 120, coverage 56.2% at overlap 8 and 62.1% at 32,
        # against 62.6% for the same checkpoint in one window at twice the VRAM. Overlap 60
        # bought 1.0 more point for 26% more time and is not the default.
        window_overlap: int = 32,
        arch: str = "locotrack",
        anchor_query: bool = True,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        # The ResNet strides down by 8, so a side that is not a multiple of 8 makes the
        # final layer round the size down and the feature grid no longer corresponds to
        # the pixels fed in ("output size is not a multiple of 8" from nets.py). Snap it
        # here rather than letting a silently-shifted grid become a tracking error.
        self.model_res = (int(model_res[0]) // 8 * 8, int(model_res[1]) // 8 * 8)
        if self.model_res != (int(model_res[0]), int(model_res[1])):
            print("[jefftrack] model_res {}x{} snapped to {}x{} (must be a multiple of 8)"
                  .format(model_res[0], model_res[1], *self.model_res))
        self.query_chunk_size = int(query_chunk_size)
        # window=0 means "decide from the clip length and the model resolution". LocoTrack
        # is not causal -- one query at frame f produces the WHOLE clip in one pass -- so
        # windowing here is purely a memory measure, not part of the algorithm.
        self.window = int(window)
        self.window_overlap = int(window_overlap)
        # See track_queries_conf. Prepends the QUERY frame to every later window so each
        # one still holds the appearance the track is defined by. Off reproduces the
        # previous behaviour exactly, which is what an A/B between them needs.
        self.anchor_query = bool(anchor_query)
        self.ckpt = ckpt or DEFAULT_CKPT
        if not os.path.isfile(self.ckpt):
            raise SystemExit("[ERROR] checkpoint not found: {}".format(self.ckpt))
        if arch not in ("locotrack", "jefftrack"):
            raise SystemExit("[ERROR] unknown arch {!r}".format(arch))
        self.arch = arch
        self.model = (_load_locotrack(self.ckpt, model_size, self.device)
                      if arch == "locotrack"
                      else _load_jefftrack(self.ckpt, model_size, self.device))
        self.model_size = model_size

    # ------------------------------------------------------------------ preprocessing
    def _prep_video(self, frames_bgr: np.ndarray) -> torch.Tensor:
        """(T,H,W,3) uint8 BGR -> (1,T,h,w,3) uint8 RGB on device, at model_res.

        The resize happens on the CPU with INTER_AREA rather than inside the vendor's
        inference(), which interpolates the whole clip as float32 on the GPU: at 312
        frames that is ~2 GB of transient VRAM spent to reach a 256x256 tensor.
        """
        h, w = self.model_res
        out = np.empty((frames_bgr.shape[0], h, w, 3), np.uint8)
        for t, fr in enumerate(frames_bgr):
            small = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            out[t] = small[:, :, ::-1]  # BGR -> RGB
        return torch.from_numpy(out).unsqueeze(0).to(self.device)

    def _q_to_model(self, queries: np.ndarray, W: int, H: int) -> torch.Tensor:
        """[frame, x, y] in caller pixels -> (1,N,3) 'tyx' in model_res pixels."""
        q = np.asarray(queries, np.float32).reshape(-1, 3)
        h, w = self.model_res
        t = q[:, 0]
        y = q[:, 2] * (h / float(H))
        x = q[:, 1] * (w / float(W))
        return torch.from_numpy(np.stack([t, y, x], 1)[None]).float().to(self.device)

    # ------------------------------------------------------------------ core inference
    @torch.no_grad()
    def _infer(self, video_u8: torch.Tensor, q_tyx: torch.Tensor):
        """One forward pass. Returns (tracks (N,T,2) xy in model_res px, occ_prob (N,T)).

        Mirrors the vendor's own LocoTrack.inference() (locotrack_model.py:1030-1053)
        line for line, with one deliberate difference: it stops before the
        `pred_occ > 0.5` threshold and hands back the continuous probability, because
        that number is the whole reason to prefer this model over the TAPNext path.
        The uint8 -> [-1,1] normalisation below is the vendor's, not an assumption.
        """
        video = video_u8.float() / 255.0 * 2 - 1
        out = self.model(video, q_tyx, query_chunk_size=self.query_chunk_size)
        tracks = out["tracks"][0]                       # (N, T, 2) as [x, y], raster
        occ = torch.sigmoid(out["occlusion"][0])
        unc = torch.sigmoid(out["expected_dist"][0])
        # vendor: pred_occ = 1 - (1 - sigmoid(occ)) * (1 - sigmoid(expected_dist))
        # i.e. a point counts as unusable if it is EITHER occluded OR uncertain.
        occ_prob = 1.0 - (1.0 - occ) * (1.0 - unc)
        return tracks.float().cpu().numpy(), occ_prob.float().cpu().numpy()

    def _auto_window(self, T: int) -> int:
        if self.window > 0:
            return self.window
        # Cost scales with T times the feature-grid area. Headroom figures for a 16 GB
        # A4000; a caller that proves these wrong on a plate passes window=N explicitly.
        px = self.model_res[0] * self.model_res[1]
        budget = 250 if px <= 256 * 256 else 120 if px <= 384 * 512 else 64
        return min(T, budget)

    # ------------------------------------------------------------------ public surface
    def track_queries_conf(self, frames_bgr, queries, fp16: bool = False):
        """Full-fidelity call: (tracks (T,N,2), vis (T,N) bool, conf (T,N) float 0..1).

        conf is 1 - P(occluded or uncertain). vis is conf >= 0.5, which reproduces the
        vendor's own threshold exactly.
        """
        frames_bgr = np.asarray(frames_bgr)
        T, H, W = frames_bgr.shape[0], frames_bgr.shape[1], frames_bgr.shape[2]
        q = np.asarray(queries, np.float32).reshape(-1, 3)
        N = q.shape[0]
        win = self._auto_window(T)

        tracks = np.zeros((T, N, 2), np.float32)
        conf = np.zeros((T, N), np.float32)
        h, w = self.model_res
        sx, sy = W / float(w), H / float(h)

        if win >= T:
            spans = [(0, T)]
        else:
            # The overlap has to leave real forward progress and stay smaller than the
            # window: overlap == win collapses the step to 1 and turns the clip into one
            # re-query per frame, the worst possible way to run this model.
            overlap = int(min(self.window_overlap, max(1, win // 2)))
            step = max(1, win - overlap)
            spans = []
            for s in range(0, T, step):
                e = min(s + win, T)
                if e > s and (not spans or e > spans[-1][1]):
                    spans.append((s, e))
                if e >= T:
                    break

        qframes = np.unique(q[:, 0].astype(int))   # usually just [0]

        filled = 0   # frames already written; a later window only writes past this
        for wi, (s, e) in enumerate(spans):
            block = frames_bgr[s:e]
            n_anchor = 0
            if wi == 0:
                qq = q.copy()
                qq[:, 0] = np.clip(qq[:, 0] - s, 0, e - s - 1)
            elif self.anchor_query:
                # ANCHORED re-query. Prepend the frames the tracks were QUERIED on to this
                # window's block, and query every track at its own original frame and
                # original position, exactly as window 0 did.
                #
                # The alternative below -- re-querying at frame `s` on the position the
                # previous window predicted there -- defines each track by whatever pixel
                # it happened to be sitting on at the seam. That is fine while the track is
                # healthy and destroys it when it is not: a point that is OCCLUDED at the
                # seam is re-defined as the occluder, and no later frame can undo it,
                # because the appearance the model matches against is now the wrong
                # appearance. It cannot re-acquire something it no longer has a picture of.
                #
                # Measured on a 261-frame plate whose subject is fully hidden across the
                # seam: the same checkpoint scores 34.9% coverage windowed and 62.6% in one
                # window, and the difference is a cliff at the window boundary, not at the
                # occlusion. On the occlusion bench with ground truth it is the difference
                # between a 278 px mean re-acquisition error and a 1.53 px one.
                #
                # Cost is one extra frame of context per window. The prepended frame is
                # temporally discontinuous with the rest of the block, so its own output is
                # discarded -- it is there to be correlated against, not to be tracked.
                anchors = [f for f in qframes if not (s <= f < e)]
                n_anchor = len(anchors)
                if n_anchor:
                    block = np.concatenate([frames_bgr[anchors], block], axis=0)
                local = {int(f): i for i, f in enumerate(anchors)}
                qq = q.copy()
                qq[:, 0] = [local[int(f)] if int(f) in local else int(f) - s + n_anchor
                            for f in q[:, 0]]
            else:
                # Legacy path, kept so the change above can be measured against it.
                # The frame index and the position must refer to the same instant;
                # carrying the previous window's LAST position into local frame 0 instead
                # is an `overlap`-frame mismatch that compounds at every seam. Measured on
                # the selftest, that mistake cost 0.10 px -> 72 px.
                qq = np.concatenate(
                    [np.zeros((N, 1), np.float32), tracks[s].astype(np.float32)], axis=1)

            video = self._prep_video(block)
            tr, occ_p = self._infer(video, self._q_to_model(qq, W, H))
            del video
            if self.device == "cuda":
                torch.cuda.empty_cache()

            tr = np.transpose(tr, (1, 0, 2))         # (N,T',2) -> (T',N,2)
            tr[..., 0] *= sx
            tr[..., 1] *= sy
            cf = np.transpose(1.0 - occ_p, (1, 0))   # (T',N)
            if n_anchor:                             # drop the prepended anchor frames
                tr, cf = tr[n_anchor:], cf[n_anchor:]

            # The overlap belongs to the earlier window: there the point was tracked from
            # a real query with more context behind it. Only write frames not yet filled.
            beg = max(s, filled)
            tracks[beg:e] = tr[beg - s:]
            conf[beg:e] = cf[beg - s:]
            filled = e

        vis = conf >= 0.5
        return tracks, vis, conf

    def track_queries(self, frames_bgr, queries, fp16: bool = False):
        """Bot-compatible call -- identical signature and shapes to
        a TAPNext-style engine surface."""
        tracks, vis, _ = self.track_queries_conf(frames_bgr, queries, fp16=fp16)
        return tracks, vis

    def track_grid(self, frames_bgr, grid_size: int, grid_query_frame: int = 0,
                   segm_mask=None):
        """Grid call -- the TAPNext-style engine surface. Seeds a
        grid_size x grid_size grid on grid_query_frame, keeping only points where
        segm_mask is nonzero when one is given."""
        frames_bgr = np.asarray(frames_bgr)
        H, W = frames_bgr.shape[1], frames_bgr.shape[2]
        ys = np.linspace(0, H - 1, grid_size + 2)[1:-1]
        xs = np.linspace(0, W - 1, grid_size + 2)[1:-1]
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
        if segm_mask is not None:
            m = np.asarray(segm_mask)
            if m.ndim == 3:
                m = m[..., 0]
            keep = m[np.int32(pts[:, 1]), np.int32(pts[:, 0])] > 0
            pts = pts[keep]
            if pts.size == 0:
                raise SystemExit("[ERROR] segm_mask left no grid points")
        q = np.concatenate(
            [np.full((len(pts), 1), float(grid_query_frame), np.float32), pts], 1)
        return self.track_queries(frames_bgr, q)


# --------------------------------------------------------------------------- self-test
def _selftest(model_size: str, model_res, ckpt: Optional[str], window: int = 0) -> int:
    """Track a known rigid translation and check the answer against exact ground truth.

    A wrong axis order, a wrong normalisation, or a transposed tyx query all produce
    tracks that LOOK plausible in an overlay and are quietly wrong -- the same failure
    class tools/check_per_track.py exists to catch. Integer np.roll gives exact ground
    truth with no resampling blur to hide behind.
    """
    rng = np.random.default_rng(0)
    T, H, W = 24, 288, 384
    base = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)  # kill single-pixel noise, keep texture
    dx, dy = 3, -2
    frames = np.stack([np.roll(np.roll(base, t * dy, 0), t * dx, 1) for t in range(T)])

    # Seed inside a margin so the wrap-around seam never enters a correlation window.
    m = 64
    pts = np.array([[x, y] for y in np.linspace(m, H - m, 4)
                    for x in np.linspace(m, W - m, 5)], np.float32)
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)

    eng = JeffTrackEngine(device="cuda", model_size=model_size, model_res=model_res,
                        ckpt=ckpt, window=window)
    tracks, vis, conf = eng.track_queries_conf(frames, q)

    gt = np.stack([pts + np.array([dx * t, dy * t], np.float32) for t in range(T)])
    err = np.linalg.norm(tracks - gt, axis=-1)          # (T,N)
    inb = ((gt[..., 0] >= 0) & (gt[..., 0] < W) & (gt[..., 1] >= 0) & (gt[..., 1] < H))

    print("device={} model={} res={} T={} N={} points  window={}".format(
        eng.device, model_size, eng.model_res, T, len(pts), eng._auto_window(T)))
    print("  mean err {:7.3f} px   median {:7.3f} px   p95 {:7.3f} px   max {:7.3f} px"
          .format(err[inb].mean(), np.median(err[inb]),
                  np.percentile(err[inb], 95), err[inb].max()))
    print("  visible  {:5.1f}%   mean conf {:.3f}".format(
        vis[inb].mean() * 100, conf[inb].mean()))

    # A transposed axis on this motion lands ~5*t px out and blows past the bar instantly.
    # The bar is set by the model's own resolution: at 256x256 on a 384-wide frame one
    # model pixel is 1.5 source px, so under ~4 px is honest tracking, not luck.
    bar = 4.0
    ok = float(np.median(err[inb])) < bar and float(vis[inb].mean()) > 0.9
    print("  {}  (median < {} px and >90% visible)".format("PASS" if ok else "FAIL", bar))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Jeff-Tracker engine (LocoTrack, Apache-2.0)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--model-res", default="256x256",
                    help="model input resolution, HxW (256x256 default; 384x512 is the "
                         "paper's high-accuracy setting)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--window", type=int, default=0,
                    help="force a time window; 0 = auto. Used by --selftest to exercise "
                         "the seam-chaining path, which a short clip never reaches.")
    a = ap.parse_args()
    res = tuple(int(v) for v in a.model_res.lower().split("x"))
    if not a.selftest:
        ap.error("nothing to do -- pass --selftest (this module is a library otherwise)")
    return _selftest(a.model_size, res, a.ckpt, a.window)


if __name__ == "__main__":
    raise SystemExit(main())
