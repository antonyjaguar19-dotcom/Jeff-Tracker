"""Render the README demo GIFs from TAP-Vid DAVIS.

    python tools/make_demo.py --clip horsejump-high --mode grid   --out assets/demo_grid.gif
    python tools/make_demo.py --clip bmx-trees      --mode conf   --out assets/demo_confidence.gif
    python tools/make_demo.py --clip libby          --mode occl   --out assets/demo_occlusion.gif

DAVIS is used rather than any footage of our own because it is what every tracker in this
family demos on, so the pictures are comparable by eye, and because its licence permits
redistribution: the DAVIS videos are **CC BY 4.0** and the TAP-Vid point annotations are
Apache-2.0 (DeepMind). Attribution for both is in assets/README.md, which is a licence
condition and not decoration.

Three modes, because one picture cannot honestly carry this model:

  grid  the hero shot. A dense grid carried through the clip, coloured by track.
  conf  the differentiator. Points coloured by the model's OWN confidence, green through
        red. This is the signal a TAPNext-style path throws away by thresholding
        visibility logits, and it detects the model's own >5 px frames at AUC 0.955.
  occl  the limitation, drawn rather than described. Hollow rings are frames the model
        declines to place. It gaps an occlusion instead of crossing it, and a demo that
        hid that would be advertising rather than evidence.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dbtrack.engine import DBTrackEngine, DEFAULT_CKPT  # noqa: E402
from dbtrack.io import seed_grid, track_colors  # noqa: E402

DEFAULT_PKL = os.path.join(ROOT, "data", "tapvid_davis", "tapvid_davis.pkl")


def load_clip(pkl: str, name: str):
    if not os.path.isfile(pkl):
        raise SystemExit(
            "[ERROR] {} not found.\n"
            "        curl -L -o tapvid_davis.zip "
            "https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip".format(pkl))
    with open(pkl, "rb") as fh:
        data = pickle.load(fh)
    if name not in data:
        raise SystemExit("[ERROR] {!r} not in the pkl. Available: {}".format(
            name, ", ".join(sorted(data))))
    return data[name]["video"]                      # (T, H, W, 3) uint8 RGB


def conf_color(c: float):
    """Green at confident, red at not. Returned BGR, because everything here is OpenCV."""
    c = float(np.clip(c, 0.0, 1.0))
    return (0, int(255 * c), int(255 * (1.0 - c)))


def draw(frame_bgr, tracks, vis, conf, t, colors, mode, tail):
    out = frame_bgr.copy()
    n = tracks.shape[1]
    for i in range(n):
        t0 = max(0, t - tail)
        seg, sv = tracks[t0:t + 1, i], vis[t0:t + 1, i]
        col = colors[i] if mode != "conf" else conf_color(conf[t, i])
        for k in range(1, len(seg)):
            if sv[k] and sv[k - 1]:
                cv2.line(out, tuple(np.int32(seg[k - 1])), tuple(np.int32(seg[k])),
                         col, 1, cv2.LINE_AA)
        p = tuple(np.int32(tracks[t, i]))
        if vis[t, i]:
            cv2.circle(out, p, 3, col, -1, cv2.LINE_AA)
        elif mode in ("occl", "conf"):
            # A frame the model declines to place. Drawn, not dropped -- the gap is the
            # behaviour worth showing.
            cv2.circle(out, p, 4, (60, 60, 255), 1, cv2.LINE_AA)
    return out


def save_gif(path, frames_rgb, fps, colors=96):
    """Write a GIF small enough to sit in a README.

    A straight mimsave of a 50-frame 640px clip lands around 6 MB, which GitHub will
    serve but nobody waits for. Quantising to one adaptive palette built from the middle
    frame -- rather than a fresh palette per frame -- keeps colours stable between frames
    so the inter-frame optimiser has long runs to collapse, and drops it by roughly 4x.
    """
    from PIL import Image                                    # noqa: PLC0415
    mid = Image.fromarray(frames_rgb[len(frames_rgb) // 2])
    pal = mid.quantize(colors=colors, method=Image.MEDIANCUT)
    imgs = [Image.fromarray(f).quantize(palette=pal, dither=Image.FLOYDSTEINBERG)
            for f in frames_rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000.0 / fps)), loop=0, optimize=True, disposal=2)


def banner(img, text):
    h = 26
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="render a README demo GIF")
    ap.add_argument("--clip", default="horsejump-high")
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--mode", default="grid", choices=["grid", "conf", "occl"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--arch", default="dbtrack", choices=["locotrack", "dbtrack"])
    ap.add_argument("--model-res", default="256x256")
    ap.add_argument("--grid", type=int, default=18)
    ap.add_argument("--frames", type=int, default=0, help="0 = the whole clip")
    ap.add_argument("--width", type=int, default=640, help="output width")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--tail", type=int, default=14)
    ap.add_argument("--colors", type=int, default=96,
                    help="GIF palette size; lower is smaller and flatter")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    model_res = tuple(int(v) for v in a.model_res.lower().split("x"))
    rgb = load_clip(a.pkl, a.clip)
    if a.frames:
        rgb = rgb[:a.frames]
    frames = np.stack([f[:, :, ::-1] for f in rgb])          # RGB -> BGR
    T, H, W = frames.shape[:3]
    print("[demo] {} {} frames {}x{}".format(a.clip, T, W, H))

    pts = seed_grid(a.grid, W, H)
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)

    eng = DBTrackEngine(device="cuda", model_size="base", ckpt=a.ckpt,
                        model_res=model_res, arch=a.arch)
    tracks, vis, conf = eng.track_queries_conf(frames, q)
    print("[demo] {} points, {:.1f}% visible, mean conf {:.3f}".format(
        len(pts), 100.0 * vis.mean(), conf.mean()))

    colors = track_colors(tracks.shape[1])
    ow = min(a.width, W)
    oh = int(round(H * ow / float(W)))
    out = []
    for t in range(T):
        img = draw(frames[t], tracks, vis, conf, t, colors, a.mode, a.tail)
        if (ow, oh) != (W, H):
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
        if a.label:
            img = banner(img, a.label)
        out.append(img[:, :, ::-1])                          # BGR -> RGB for the encoder

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    save_gif(a.out, out, a.fps, a.colors)
    print("[demo] wrote {} ({:.1f} MB)".format(
        a.out, os.path.getsize(a.out) / 1048576.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
