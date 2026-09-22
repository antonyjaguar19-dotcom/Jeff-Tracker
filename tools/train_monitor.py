"""A live dashboard for a training run: progress, GPU, CPU, memory, disk. In a browser.

    python tools/train_monitor.py

then open http://localhost:8099. Point it at a different run with --log, or move it off
8099 with --port. Reading is all it does: it never writes to the log, never touches the
checkpoints, and killing it has no effect on the training.

Standard library only, on purpose. This has to start instantly in the middle of a run that
already owns the GPU, and installing anything into the experiment's pydeps mid-run is how a
monitoring tool ends up being the thing that broke training. So:

  * CPU and RAM come from Win32 through ctypes (`GetSystemTimes`, `GlobalMemoryStatusEx`),
    not psutil, which is not in the embeddable runtime.
  * GPU comes from `nvidia-smi`, which ships with the driver.
  * the training state is parsed out of the log the trainer already writes, so nothing had
    to change in train_cross.py to support this.

The CPU and memory panels are Windows-only -- they read Win32 counters that do not
exist elsewhere -- and say so rather than half-working: on another platform those two
show "--" while the training progress, the GPU panel and the chart carry on.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The CPU and memory readers below are Win32. Guarded rather than assumed, so this file
# imports and serves on Linux and macOS with those two panels reading "--" instead of
# raising at startup -- a monitoring tool that cannot start is worse than one missing a row.
IS_WINDOWS = sys.platform == "win32"
_K32 = ctypes.windll.kernel32 if IS_WINDOWS else None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
LOG_DIR = os.path.join(ROOT, "logs")


def newest_log(log_dir: str = LOG_DIR):
    """The most recently written train_*.log.

    Resolved on every poll rather than once at startup, so the page follows whichever run
    is live: starting the next run is enough, the dashboard does not have to be restarted
    and pointed at it by hand. Pinning a run is what --log is for.
    """
    import glob
    logs = glob.glob(os.path.join(log_dir, "train_*.log"))
    if not logs:
        return os.path.join(log_dir, "train.log")
    return max(logs, key=os.path.getmtime)


DEFAULT_LOG = None                      # None means "follow the newest", see sampler()

# [   25/20000] loss 5.3695  lr 1.29e-05  gnorm 3.96 clip 100%  0.55s/step  vram 3.0GB  eta 3.1h
STEP_RE = re.compile(
    r"^\[\s*(\d+)/(\d+)\]\s+loss\s+([\d.]+)\s+lr\s+([\d.eE+-]+)\s+gnorm\s+([\d.]+)"
    r"\s+clip\s+(\d+)%\s+([\d.]+)s/step\s+vram\s+([\d.]+)GB\s+eta\s+([\d.]+)h")


# --------------------------------------------------------------------------- machine


class _CpuSampler:
    """Total CPU load between calls, from GetSystemTimes.

    Sampled as a DIFFERENCE between two reads rather than an instantaneous figure -- the
    counter is cumulative since boot, so a single read says nothing at all about now.
    """

    def __init__(self):
        self._prev = None

    def read(self) -> float:
        if not IS_WINDOWS:
            return -1.0
        idle, kern, user = (ctypes.c_ulonglong(), ctypes.c_ulonglong(),
                            ctypes.c_ulonglong())
        if not _K32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
            return -1.0
        cur = (idle.value, kern.value + user.value)
        if self._prev is None:
            self._prev = cur
            return -1.0
        d_idle = cur[0] - self._prev[0]
        d_tot = cur[1] - self._prev[1]
        self._prev = cur
        if d_tot <= 0:
            return -1.0
        return max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_tot)))


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def read_ram():
    if not IS_WINDOWS:
        return None
    m = _MemStatus()
    m.dwLength = ctypes.sizeof(_MemStatus)
    if not _K32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return None
    return {"used_gb": (m.ullTotalPhys - m.ullAvailPhys) / 1e9,
            "total_gb": m.ullTotalPhys / 1e9, "pct": float(m.dwMemoryLoad)}


def read_disk(path: str):
    if not IS_WINDOWS:
        try:
            u = __import__("shutil").disk_usage(path)
            return {"free_gb": u.free / 1e9, "total_gb": u.total / 1e9}
        except OSError:
            return None
    free = ctypes.c_ulonglong()
    total = ctypes.c_ulonglong()
    ok = _K32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(path), None, ctypes.byref(total), ctypes.byref(free))
    if not ok:
        return None
    return {"free_gb": free.value / 1e9, "total_gb": total.value / 1e9}


_NVIDIA_FIELDS = ("utilization.gpu,memory.used,memory.total,temperature.gpu,"
                  "power.draw,power.limit,clocks.sm,fan.speed")


def read_gpu():
    """nvidia-smi, with its window hidden so a 2 s poll does not flash a console."""
    try:
        kw = {}
        if IS_WINDOWS:                       # keep a 2 s poll from flashing a console
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kw["startupinfo"] = si
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + _NVIDIA_FIELDS,
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=8, **kw).decode()
    except Exception:
        return None
    row = out.strip().splitlines()
    if not row:
        return None

    def num(x):
        x = x.strip()
        try:
            return float(x)
        except ValueError:
            return None

    p = [num(v) for v in row[0].split(",")]
    while len(p) < 8:
        p.append(None)
    return {"util": p[0], "mem_used_gb": (p[1] or 0) / 1024.0,
            "mem_total_gb": (p[2] or 0) / 1024.0, "temp": p[3], "power": p[4],
            "power_limit": p[5], "clock": p[6], "fan": p[7]}


# --------------------------------------------------------------------------- the run


def read_log(path: str, tail_bytes: int = 400_000):
    """Parse the trainer's own log. Only the tail is read, so a long run stays cheap."""
    if not os.path.exists(path):
        return {"state": "missing", "path": path}
    size = os.path.getsize(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        if size > tail_bytes:
            fh.seek(size - tail_bytes)
            fh.readline()
        lines = fh.readlines()

    steps, last = [], None
    for ln in lines:
        m = STEP_RE.match(ln.strip())
        if m:
            last = m
            steps.append({"step": int(m.group(1)), "loss": float(m.group(3)),
                          "gnorm": float(m.group(5))})

    head = [ln.rstrip() for ln in lines
            if ln.startswith(("[model]", "[loss]", "[sched]", "[ema]", "[data]", "  "))
            and not ln.startswith("  [")][:8]

    err = [ln.rstrip() for ln in lines
           if ("Traceback" in ln or "Error" in ln or "error:" in ln
               or "out of memory" in ln.lower())]

    done = any(ln.startswith("[done]") for ln in lines)
    state = "error" if err else ("finished" if done else
                                 ("running" if last else "starting"))

    out = {"state": state, "path": path, "header": head,
           "errors": err[-6:], "mtime": os.path.getmtime(path),
           "age_s": time.time() - os.path.getmtime(path)}
    if last:
        total = int(last.group(2))
        step = int(last.group(1))
        out.update({
            "step": step, "total": total, "pct": 100.0 * step / max(total, 1),
            "loss": float(last.group(3)), "lr": float(last.group(4)),
            "gnorm": float(last.group(5)), "clip": float(last.group(6)),
            "s_per_step": float(last.group(7)), "vram_gb": float(last.group(8)),
            "eta_h": float(last.group(9)),
            # One point per ~200 steps keeps the chart readable over a 20k run.
            "curve": steps[-400:],
        })
    return out


STATE = {"lock": threading.Lock(), "data": {}}


def sampler(log_path, disk: str, period: float = 2.0):
    cpu = _CpuSampler()
    cpu.read()
    while True:
        path = log_path or newest_log()
        snap = {"t": time.time(), "cpu": cpu.read(), "ram": read_ram(),
                "gpu": read_gpu(), "disk": read_disk(disk),
                "following": "newest run" if log_path is None else "pinned",
                "run": read_log(path)}
        with STATE["lock"]:
            STATE["data"] = snap
        time.sleep(period)


# --------------------------------------------------------------------------- serving


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Jeff-Tracker training</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{
    --bg:#0e1116; --panel:#161b22; --line:#262d36; --tx:#e6edf3; --dim:#8b949e;
    --ok:#3fb950; --warn:#d29922; --bad:#f85149; --accent:#58a6ff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--tx);
       font:14px/1.5 "Segoe UI",system-ui,sans-serif;padding:20px}
  h1{font-size:17px;margin:0 0 2px;font-weight:600}
  .sub{color:var(--dim);font-size:12px;margin-bottom:18px}
  .grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
        max-width:1200px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
  .card.wide{grid-column:1/-1}
  .lbl{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em;
       margin-bottom:8px}
  .big{font-size:30px;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1}
  .unit{font-size:14px;color:var(--dim);font-weight:400}
  .bar{height:9px;background:#0b0e13;border-radius:5px;overflow:hidden;margin:10px 0 4px}
  .bar i{display:block;height:100%;background:var(--accent);border-radius:5px;
         transition:width .4s ease}
  .row{display:flex;justify-content:space-between;font-size:12px;color:var(--dim);
       font-variant-numeric:tabular-nums}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  td{padding:4px 0;font-size:13px}
  td:last-child{text-align:right;font-weight:600}
  .pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;
        font-weight:600;letter-spacing:.04em;text-transform:uppercase}
  .s-running{background:rgba(63,185,80,.15);color:var(--ok)}
  .s-starting{background:rgba(210,153,34,.15);color:var(--warn)}
  .s-finished{background:rgba(88,166,255,.15);color:var(--accent)}
  .s-error,.s-missing,.s-stalled{background:rgba(248,81,73,.15);color:var(--bad)}
  svg{display:block;width:100%;height:110px}
  .err{color:var(--bad);font-family:Consolas,monospace;font-size:12px;
       white-space:pre-wrap;margin-top:8px}
  .foot{color:var(--dim);font-size:11px;margin-top:16px}
  .note{color:var(--dim);font-size:11.5px;line-height:1.55;margin-top:10px;
        border-top:1px solid var(--line);padding-top:10px;max-width:70ch}
  .note b{color:var(--tx)}
</style></head><body>
<h1>Jeff-Tracker &mdash; training <span id="runname" class="pill s-finished"></span>
  <span id="state" class="pill s-starting">starting</span></h1>
<div class="sub" id="what">reading the log&hellip;</div>

<div class="grid">
  <div class="card wide">
    <div class="lbl">Progress</div>
    <div class="big"><span id="pct">0.0</span><span class="unit">%</span>
      <span class="unit" id="steps" style="margin-left:14px"></span></div>
    <div class="bar"><i id="pbar" style="width:0%"></i></div>
    <div class="row"><span id="elapsed">&nbsp;</span><span id="eta">&nbsp;</span></div>
  </div>

  <div class="card">
    <div class="lbl">Graphics card</div>
    <div class="big"><span id="gutil">--</span><span class="unit">% busy</span></div>
    <div class="bar"><i id="gbar" style="width:0%"></i></div>
    <table><tbody>
      <tr><td>Card memory</td><td id="gmem">--</td></tr>
      <tr><td>Temperature</td><td id="gtemp">--</td></tr>
      <tr><td>Power</td><td id="gpow">--</td></tr>
    </tbody></table>
  </div>

  <div class="card">
    <div class="lbl">Processor &amp; memory</div>
    <div class="big"><span id="cpu">--</span><span class="unit">% busy</span></div>
    <div class="bar"><i id="cbar" style="width:0%"></i></div>
    <table><tbody>
      <tr><td>System memory</td><td id="ram">--</td></tr>
      <tr><td>Disk D: free</td><td id="disk">--</td></tr>
    </tbody></table>
  </div>

  <div class="card">
    <div class="lbl">This run &mdash; health, not accuracy</div>
    <table><tbody>
      <tr><td>Practice score</td><td id="loss">--</td></tr>
      <tr><td>Seconds per step</td><td id="sps">--</td></tr>
      <tr><td>Learning rate</td><td id="lr">--</td></tr>
      <tr><td>Gradient size</td><td id="gn">--</td></tr>
      <tr><td>Capped steps</td><td id="clip">--</td></tr>
    </tbody></table>
  </div>

  <div class="card wide">
    <div class="lbl">Practice score over time &mdash; trend down is what matters; wobble is normal</div>
    <svg id="chart" viewBox="0 0 800 110" preserveAspectRatio="none"></svg>
    <div class="row"><span id="cmin">&nbsp;</span><span id="cmax">&nbsp;</span></div>
    <div class="note">The practice score is how wrong the model is on the shots it is
      studying right now. It is <b>not pixels</b>, and it cannot be compared to CoTracker3
      or to a run trained on different shots &mdash; a model can score better here and be
      worse on real footage. It answers one question only: is this run healthy? Going down
      is healthy. Whether the model actually improved is decided afterwards, by the bench
      scores in pixels.</div>
  </div>

  <div class="card wide" id="errcard" style="display:none">
    <div class="lbl">Problem</div><div class="err" id="errtx"></div>
  </div>
</div>
<div class="foot" id="foot">&nbsp;</div>

<script>
const $ = i => document.getElementById(i);
function hms(s){ if(!isFinite(s)||s<0) return "--";
  const h=Math.floor(s/3600), m=Math.floor(s%3600/60);
  return h? h+"h "+m+"m" : m+"m "+Math.floor(s%60)+"s"; }
function col(el,v,warn,bad){ el.style.background = v>=bad? "var(--bad)"
  : v>=warn? "var(--warn)" : "var(--accent)"; }

let t0=null;
async function tick(){
  let d; try{ d = await (await fetch("/data",{cache:"no-store"})).json(); }
  catch(e){ $("state").className="pill s-error"; $("state").textContent="monitor lost";
            return; }
  const r = d.run||{};
  let st = r.state||"missing";
  if(st==="running" && r.age_s>180) st="stalled";
  $("state").className = "pill s-"+st;
  $("state").textContent = st==="stalled" ? "no output for "+hms(r.age_s) : st;

  if(r.step!=null){
    $("pct").textContent = r.pct.toFixed(1);
    $("steps").textContent = r.step.toLocaleString()+" / "+r.total.toLocaleString()+" steps";
    $("pbar").style.width = r.pct+"%";
    $("eta").textContent = "about "+r.eta_h.toFixed(1)+" h left";
    if(t0===null) t0 = Date.now() - r.step*r.s_per_step*1000;
    $("elapsed").textContent = hms((Date.now()-t0)/1000)+" elapsed";
    $("loss").textContent = r.loss.toFixed(4);
    $("sps").textContent  = r.s_per_step.toFixed(2)+" s";
    $("lr").textContent   = r.lr.toExponential(2);
    $("gn").textContent   = r.gnorm.toFixed(2);
    $("clip").textContent = r.clip.toFixed(0)+"%";
  }
  $("what").textContent = (r.header&&r.header.length? r.header.join("   ") : r.path||"");

  const g = d.gpu;
  if(g){
    $("gutil").textContent = g.util==null? "--" : g.util.toFixed(0);
    $("gbar").style.width = (g.util||0)+"%"; col($("gbar"), g.util||0, 101, 102);
    $("gmem").textContent  = g.mem_used_gb.toFixed(1)+" / "+g.mem_total_gb.toFixed(1)+" GB";
    $("gtemp").textContent = g.temp==null? "--" : g.temp.toFixed(0)+" °C";
    $("gpow").textContent  = g.power==null? "--" :
        g.power.toFixed(0)+(g.power_limit? " / "+g.power_limit.toFixed(0):"")+" W";
    $("gtemp").style.color = (g.temp||0)>=84? "var(--bad)"
                           : (g.temp||0)>=78? "var(--warn)" : "";
  }
  if(d.cpu!=null && d.cpu>=0){
    $("cpu").textContent = d.cpu.toFixed(0);
    $("cbar").style.width = d.cpu+"%"; col($("cbar"), d.cpu, 85, 95);
  }
  if(d.ram) $("ram").textContent = d.ram.used_gb.toFixed(1)+" / "+d.ram.total_gb.toFixed(1)+" GB";
  if(d.disk){ $("disk").textContent = d.disk.free_gb.toFixed(0)+" GB";
    $("disk").style.color = d.disk.free_gb<20? "var(--bad)"
                          : d.disk.free_gb<60? "var(--warn)" : ""; }

  const c = r.curve||[];
  if(c.length>1){
    const ys = c.map(p=>p.loss), lo=Math.min(...ys), hi=Math.max(...ys);
    const sp = (hi-lo)||1;
    const pts = c.map((p,i)=>[i/(c.length-1)*800, 105-(p.loss-lo)/sp*100]);
    const dd = pts.map((p,i)=>(i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
    $("chart").innerHTML =
      '<path d="'+dd+' L800 110 L0 110 Z" fill="rgba(88,166,255,.10)"/>'+
      '<path d="'+dd+'" fill="none" stroke="var(--accent)" stroke-width="1.6"/>';
    $("cmin").textContent = "step "+c[0].step.toLocaleString();
    $("cmax").textContent = "low "+lo.toFixed(2)+"   high "+hi.toFixed(2);
  }
  if(r.errors && r.errors.length){
    $("errcard").style.display=""; $("errtx").textContent = r.errors.join("\n");
  }
  const runname = (r.path||"").split(/[\/]/).pop().replace(/^train_|\.log$/g,"");
  $("runname").textContent = runname ? "run "+runname : "";
  $("foot").textContent = "updated "+new Date().toLocaleTimeString()+
      "  ·  reading "+(r.path||"")+"  ·  this page only reads, it cannot affect the run";
}
tick(); setInterval(tick, 2000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):                                    # noqa: N802
        if self.path.startswith("/data"):
            with STATE["lock"]:
                body = json.dumps(STATE["data"], default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                           # noqa: A003
        pass                                             # a request line per 2 s is noise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", default=None,
                    help="pin one training log. Default: follow whichever train_*.log "
                         "was written to most recently, so the page moves to the next "
                         "run on its own")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--disk", default=ROOT)
    ap.add_argument("--once", action="store_true",
                    help="print one snapshot as json and exit -- checks the readers "
                         "without starting a server")
    a = ap.parse_args()

    if a.once:
        cpu = _CpuSampler()
        cpu.read()
        time.sleep(0.4)
        print(json.dumps({"cpu": cpu.read(), "ram": read_ram(), "gpu": read_gpu(),
                          "disk": read_disk(a.disk),
                          "run": {k: v for k, v in read_log(a.log or newest_log()).items()
                                  if k != "curve"}},
                         indent=2, default=str))
        return 0

    threading.Thread(target=sampler, args=(a.log, a.disk), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print("  training monitor  ->  http://localhost:{}".format(a.port))
    print("  reading           : {}".format(a.log or "newest train_*.log "
                                        "(follows the live run)"))
    print("  Ctrl+C to stop. This only reads; the training is unaffected.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
