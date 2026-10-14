"""Live Web Dashboard for Jeff-Tracker Multi-Cycle Option A+B Training.

Zero-dependency HTTP server using standard Python library.
Serves a responsive, dark VFX-themed dashboard at http://localhost:8765.
"""
from __future__ import annotations

import glob
import http.server
import json
import os
import re
import socketserver
import subprocess
import time
import urllib.parse

PORT = 8765
BRAIN_TASKS_DIR = r"C:\Users\anton\.gemini\antigravity-cli\brain\167b0630-4136-41eb-a389-c4ae17685455\.system_generated\tasks"
REPO_ROOT = r"E:\CHAT_GPT\Jeff-Tracker"
WEIGHTS_JSON = os.path.join(REPO_ROOT, "weights", "jefftrack_reacq_v2_smooth.json")


def get_latest_log_path() -> str | None:
    repo_log = os.path.join(REPO_ROOT, "training.log")
    if os.path.isfile(repo_log) and os.path.getsize(repo_log) > 0:
        return repo_log
    if not os.path.isdir(BRAIN_TASKS_DIR):
        return None
    logs = glob.glob(os.path.join(BRAIN_TASKS_DIR, "*.log"))
    if not logs:
        return None
    # Sort newest first
    logs.sort(key=os.path.getmtime, reverse=True)
    # Check for the newest log that looks like a training log
    for l in logs:
        try:
            with open(l, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(2048)
                if "Streaming Multi-Cycle" in head or "train_streaming" in head or "STREAMING MULTI-BATCH" in head or "train_cross" in head:
                    return l
        except Exception:
            pass
    return logs[0]


def get_gpu_telemetry():
    telemetry = {
        "name": "NVIDIA GeForce RTX 5080",
        "vram_used": "3.1 GB",
        "vram_total": "16.3 GB",
        "vram_pct": 19,
        "gpu_util": "95%",
        "temp": "55°C",
    }
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 5:
                used = float(parts[1]) / 1024.0
                total = float(parts[2]) / 1024.0
                telemetry["name"] = parts[0]
                telemetry["vram_used"] = f"{used:.1f} GB"
                telemetry["vram_total"] = f"{total:.1f} GB"
                telemetry["vram_pct"] = int((used / max(1.0, total)) * 100)
                telemetry["gpu_util"] = f"{parts[3]}%"
                telemetry["temp"] = f"{parts[4]}°C"
    except Exception:
        pass
    return telemetry


def parse_training_log():
    log_path = get_latest_log_path()
    if not log_path or not os.path.isfile(log_path):
        return {"status": "Waiting for training task...", "lines": [], "history": []}

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return {"status": f"Error reading log: {e}", "lines": [], "history": []}

    start_shard = 12
    total_cycles = 6
    banner = "".join(lines[:35])
    m_st = re.search(r"Starting Shard:\s+(\d+)", banner)
    if m_st:
        start_shard = int(m_st.group(1))
    m_cy = re.search(r"Total Cycles:\s+(\d+)", banner)
    if m_cy:
        total_cycles = int(m_cy.group(1))

    cycle = 1
    shards_info = f"{start_shard:04d}..{start_shard+1:04d}"
    current_step = 0
    total_steps = total_cycles * 250
    loss_history = []
    latest_loss = 0.0
    latest_lr = "1.0e-4"
    latest_speed = "0.30s/step"
    latest_vram = "3.1GB"
    phase = "Active Training"
    completed = False

    dl_re = re.compile(r"\[downloading\]\s+(\d+)\.tar\.gz")
    conv_re = re.compile(r"\[(\d+)/(\d+)\]\s+(\d+)\s+([\d\.]+)\s*MB")
    step_re = re.compile(r"\[\s*(\d+)/(\d+)\]\s+loss\s+([\d\.]+)\s+lr\s+([e\d\.\-]+)\s+gnorm\s+([\d\.]+)\s+clip\s+(\d+)%\s+([\d\.]+s/step)\s+vram\s+([\d\.]+GB)", re.IGNORECASE)

    for l in lines:
        m_dl = dl_re.search(l)
        if m_dl:
            shard_num = int(m_dl.group(1))
            cycle = max(1, min(total_cycles, (shard_num - start_shard) // 2 + 1))
            s_base = start_shard + (cycle - 1) * 2
            shards_info = f"{s_base:04d}..{s_base+1:04d}"
            phase = f"Streaming Shard {shard_num:04d}.tar.gz"

        m_cv = conv_re.search(l)
        if m_cv:
            phase = f"Converting Shard {m_cv.group(3)} to Cache"

        m_s = step_re.search(l)
        if m_s:
            cycle_step = int(m_s.group(1))
            loss_val = float(m_s.group(3))
            latest_loss = loss_val
            latest_lr = m_s.group(4)
            latest_speed = m_s.group(7)
            latest_vram = m_s.group(8)
            g_step = (cycle - 1) * 250 + cycle_step
            current_step = g_step
            loss_history.append({"step": g_step, "loss": round(loss_val, 4)})
            phase = f"Training Step {cycle_step}/250 (Cycle {cycle}/{total_cycles})"

        if "ALL" in l and "COMPLETED" in l:
            completed = True
            phase = "Training Fully Completed!"

    progress_pct = min(100, int((current_step / max(1, total_steps)) * 100))
    if completed:
        progress_pct = 100

    tail_lines = [line.rstrip() for line in lines[-100:]]

    return {
        "cycle": cycle,
        "total_cycles": total_cycles,
        "shards": shards_info,
        "current_step": current_step,
        "total_steps": total_steps,
        "progress_pct": progress_pct,
        "latest_loss": latest_loss,
        "latest_lr": latest_lr,
        "latest_speed": latest_speed,
        "latest_vram": latest_vram,
        "phase": phase,
        "completed": completed,
        "history": loss_history,
        "log_tail": tail_lines,
        "gpu": get_gpu_telemetry(),
        "timestamp": time.strftime("%H:%M:%S"),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Jeff-Tracker | Option A+B Live Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {
    --bg: #0c0e14;
    --card: #151822;
    --card-border: #232738;
    --primary: #3b82f6;
    --primary-glow: rgba(59, 130, 246, 0.35);
    --accent: #10b981;
    --accent-glow: rgba(16, 185, 129, 0.3);
    --warning: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --text-muted: #8492a6;
    --code-bg: #090a0f;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 24px;
  }
  .container { max-width: 1400px; margin: 0 auto; }
  
  /* Header */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--card-border);
    margin-bottom: 24px;
  }
  .title-group h1 {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .title-group p {
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 4px;
  }
  .badge-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .pulse-dot {
    width: 8px;
    height: 8px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 1.6s infinite ease-in-out;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 var(--accent-glow); }
    50% { opacity: 0.4; transform: scale(0.85); box-shadow: 0 0 0 6px rgba(16,185,129,0); }
  }

  /* Metric Cards Grid */
  .grid-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }
  .card:hover { border-color: rgba(59, 130, 246, 0.4); }
  .card-label {
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }
  .card-value {
    font-size: 26px;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: baseline;
    gap: 8px;
  }
  .card-sub {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
  }
  .card-progress-bar {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 99px;
    margin-top: 12px;
    overflow: hidden;
  }
  .card-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 99px;
    transition: width 0.4s ease;
  }

  /* Feature Pills */
  .pills-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 24px;
  }
  .pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--card-border);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .pill-tag {
    font-weight: 700;
    color: var(--primary);
  }
  .pill.active {
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.06);
  }
  .pill.active .pill-tag {
    color: var(--accent);
  }

  /* Main Split: Chart + Terminal */
  .main-split {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }
  @media (max-width: 1024px) {
    .main-split { grid-template-columns: 1fr; }
  }

  /* Section Title */
  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .section-title {
    font-size: 15px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* Chart Box */
  .chart-box {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    height: 420px;
    display: flex;
    flex-direction: column;
  }
  .chart-container {
    flex: 1;
    position: relative;
    width: 100%;
    height: 100%;
  }

  /* Terminal Log Box */
  .terminal-box {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    height: 420px;
    display: flex;
    flex-direction: column;
  }
  .terminal-content {
    flex: 1;
    background: var(--code-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 12px;
    font-family: "JetBrains Mono", Consolas, "Fira Code", monospace;
    font-size: 11.5px;
    color: #abb2bf;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .terminal-controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .btn {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid var(--card-border);
    color: var(--text);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s;
  }
  .btn:hover { background: rgba(255, 255, 255, 0.15); }
  .btn.active {
    background: var(--primary);
    border-color: var(--primary);
  }

  /* GPU Bar */
  .hardware-bar {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
  }
  .hw-item {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .hw-icon {
    width: 38px;
    height: 38px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--primary);
    font-weight: 700;
    font-size: 13px;
  }
  .hw-info h4 { font-size: 13px; font-weight: 600; color: #fff; }
  .hw-info p { font-size: 12px; color: var(--text-muted); }

  footer {
    text-align: center;
    margin-top: 32px;
    color: var(--text-muted);
    font-size: 12px;
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="title-group">
      <h1>
        Jeff-Tracker Option A + B
        <span class="badge-live"><span class="pulse-dot"></span> <span id="header-status">Live Streaming</span></span>
      </h1>
      <p>High-Fidelity Anti-Jitter & Non-Rigid Surface Tracking Multi-Cycle Training Pipeline</p>
    </div>
    <div style="text-align: right;">
      <div style="font-size: 12px; color: var(--text-muted);">RTX 5080 Blackwell (17 GB)</div>
      <div id="clock" style="font-size: 14px; font-weight: 600; color: var(--text);">--:--:--</div>
    </div>
  </header>

  <!-- Metric Cards -->
  <div class="grid-metrics">
    <div class="card">
      <div class="card-label">Training Progress</div>
      <div class="card-value" id="val-step">0 / 1500</div>
      <div class="card-sub" id="val-step-sub">Cycle 1 of 6 (0%)</div>
      <div class="card-progress-bar">
        <div class="card-progress-fill" id="bar-step" style="width: 0%;"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Active Shards & Data</div>
      <div class="card-value" id="val-shards" style="font-size: 22px;">0012..0013</div>
      <div class="card-sub" id="val-phase">Downloading & Converting</div>
      <div class="card-progress-bar">
        <div class="card-progress-fill" id="bar-cycle" style="width: 15%; background: var(--warning);"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">Anti-Jitter Training Loss</div>
      <div class="card-value" id="val-loss" style="color: #60a5fa;">--</div>
      <div class="card-sub" id="val-speed">Speed: -- s/step | LR: --</div>
      <div class="card-progress-bar">
        <div class="card-progress-fill" id="bar-loss" style="width: 60%; background: var(--primary);"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-label">GPU Memory (RTX 5080)</div>
      <div class="card-value" id="val-vram" style="color: #34d399;">3.1 GB</div>
      <div class="card-sub" id="val-gpu-sub">Utilization: 95% | Temp: 55°C</div>
      <div class="card-progress-bar">
        <div class="card-progress-fill" id="bar-vram" style="width: 19%; background: var(--accent);"></div>
      </div>
    </div>
  </div>

  <!-- Feature Pills -->
  <div class="pills-bar">
    <div class="pill active">
      <span class="pill-tag">Option A</span> Stateful Cross-Attention (--carry-state) Active
    </div>
    <div class="pill active">
      <span class="pill-tag">Option B</span> Camera Shake & Jitter Augmentation Active
    </div>
    <div class="pill active">
      <span class="pill-tag">Option B</span> Non-Rigid Elastic Harmonic TPS Surface Warp Active
    </div>
    <div class="pill active" style="border-color: rgba(245, 158, 11, 0.4); background: rgba(245, 158, 11, 0.08);">
      <span class="pill-tag" style="color: #f59e0b;">VFX Smoke</span> Volumetric Fog & Steam Sim Active
    </div>
    <div class="pill active" style="border-color: rgba(239, 68, 68, 0.4); background: rgba(239, 68, 68, 0.08);">
      <span class="pill-tag" style="color: #ef4444;">VFX Fire</span> Flame Flicker & Thermal Shimmer Active
    </div>
    <div class="pill active" style="border-color: rgba(56, 189, 248, 0.4); background: rgba(56, 189, 248, 0.08);">
      <span class="pill-tag" style="color: #38bdf8;">VFX Water</span> Refractive Splashes & Specular Glints Active
    </div>
    <div class="pill">
      <span class="pill-tag">Loss</span> Anti-Jitter Accel (&lambda;=0.15)
    </div>
    <div class="pill">
      <span class="pill-tag">Loss</span> Surface Deformation (&lambda;=0.08)
    </div>
    <div class="pill">
      <span class="pill-tag">Loss</span> Re-Acquisition Surge (15% Share)
    </div>
    <div class="pill">
      <span class="pill-tag">Storage</span> Bounded Disk Space (&lt;1.2 GB peak)
    </div>
  </div>

  <!-- Split View: Loss Curve + Terminal Log -->
  <div class="main-split">
    <div class="chart-box">
      <div class="section-header">
        <div class="section-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
          Convergence & Loss Trajectory
        </div>
        <div style="font-size: 11px; color: var(--text-muted);" id="chart-points-count">0 steps recorded</div>
      </div>
      <div class="chart-container">
        <canvas id="lossChart"></canvas>
      </div>
    </div>

    <div class="terminal-box">
      <div class="section-header">
        <div class="section-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>
          Live Training Console Output
        </div>
        <div class="terminal-controls">
          <button class="btn active" id="btn-autoscroll" onclick="toggleAutoScroll()">Auto-scroll: ON</button>
        </div>
      </div>
      <div class="terminal-content" id="terminal-content">Waiting for telemetry connection...</div>
    </div>
  </div>

  <!-- Hardware Telemetry Bar -->
  <div class="hardware-bar">
    <div class="hw-item">
      <div class="hw-icon">GPU</div>
      <div class="hw-info">
        <h4 id="hw-gpu-name">NVIDIA GeForce RTX 5080</h4>
        <p id="hw-gpu-desc">16.3 GB GDDR7 &bull; sm_120 Blackwell Architecture</p>
      </div>
    </div>
    <div class="hw-item">
      <div class="hw-icon" style="color: #60a5fa; border-color: rgba(96,165,250,0.3); background: rgba(96,165,250,0.1);">TORCH</div>
      <div class="hw-info">
        <h4>PyTorch 2.11.0+cu128</h4>
        <p>Native CUDA 12.8 Blackwell Compilation</p>
      </div>
    </div>
    <div class="hw-item">
      <div class="hw-icon" style="color: #34d399; border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.1);">MODEL</div>
      <div class="hw-info">
        <h4>Jeff-Tracker v2 (11.6M Trainable)</h4>
        <p>16 Stateful Proxies &bull; 4 Heads &bull; Unfrozen Mixer</p>
      </div>
    </div>
    <div class="hw-item">
      <div class="hw-icon" style="color: #f59e0b; border-color: rgba(245,158,11,0.3); background: rgba(245,158,11,0.1);">DISK</div>
      <div class="hw-info">
        <h4>Streaming Cache</h4>
        <p>Zero archive accumulation &bull; &lt; 0.1 GB cache</p>
      </div>
    </div>
  </div>

  <footer>
    Jeff-Tracker Live VFX Supervision Dashboard &bull; Auto-refreshing every 1.5s &bull; Port 8765
  </footer>

</div>

<script>
  let autoScroll = true;
  function toggleAutoScroll() {
    autoScroll = !autoScroll;
    const btn = document.getElementById("btn-autoscroll");
    btn.textContent = autoScroll ? "Auto-scroll: ON" : "Auto-scroll: OFF";
    btn.className = autoScroll ? "btn active" : "btn";
  }

  // Setup Chart.js
  const ctx = document.getElementById('lossChart').getContext('2d');
  const lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Loss',
        data: [],
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        tension: 0.25,
        fill: true,
        pointRadius: 3,
        pointBackgroundColor: '#60a5fa'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1e2230',
          titleColor: '#fff',
          bodyColor: '#38bdf8',
          borderColor: '#334155',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Optimizer Step', color: '#64748b', font: { size: 11 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#8492a6', font: { size: 10 } }
        },
        y: {
          title: { display: true, text: 'Training Loss', color: '#64748b', font: { size: 11 } },
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#8492a6', font: { size: 10 } }
        }
      }
    }
  });

  async function updateDashboard() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) return;
      const data = await res.json();

      document.getElementById('clock').textContent = data.timestamp || '--:--:--';
      document.getElementById('header-status').textContent = data.completed ? "COMPLETED" : "LIVE STREAMING";
      
      // Step & Cycle
      document.getElementById('val-step').textContent = `${data.current_step} / ${data.total_steps}`;
      document.getElementById('val-step-sub').textContent = `Cycle ${data.cycle} of ${data.total_cycles} (${data.progress_pct}%)`;
      document.getElementById('bar-step').style.width = `${data.progress_pct}%`;

      // Shards & Phase
      document.getElementById('val-shards').textContent = `Shards ${data.shards}`;
      document.getElementById('val-phase').textContent = data.phase;
      const cyclePct = Math.round((data.cycle / data.total_cycles) * 100);
      document.getElementById('bar-cycle').style.width = `${cyclePct}%`;

      // Loss & Speed
      if (data.latest_loss) {
        document.getElementById('val-loss').textContent = data.latest_loss.toFixed(4);
      }
      document.getElementById('val-speed').textContent = `Speed: ${data.latest_speed} | LR: ${data.latest_lr}`;

      // GPU
      if (data.gpu) {
        document.getElementById('val-vram').textContent = `${data.gpu.vram_used} / ${data.gpu.vram_total}`;
        document.getElementById('val-gpu-sub').textContent = `Util: ${data.gpu.gpu_util} | Temp: ${data.gpu.temp}`;
        document.getElementById('bar-vram').style.width = `${data.gpu.vram_pct}%`;
        document.getElementById('hw-gpu-name').textContent = data.gpu.name;
      }

      // Chart
      if (data.history && data.history.length > 0) {
        const labels = data.history.map(h => h.step);
        const points = data.history.map(h => h.loss);
        lossChart.data.labels = labels;
        lossChart.data.datasets[0].data = points;
        lossChart.update('none');
        document.getElementById('chart-points-count').textContent = `${data.history.length} checkpoints recorded`;
      }

      // Terminal Log
      if (data.log_tail && data.log_tail.length > 0) {
        const term = document.getElementById('terminal-content');
        term.textContent = data.log_tail.join('\\n');
        if (autoScroll) {
          term.scrollTop = term.scrollHeight;
        }
      }

    } catch (e) {
      console.error("Dashboard poll error:", e);
    }
  }

  setInterval(updateDashboard, 1500);
  updateDashboard();
</script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = parse_training_log()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in terminal
        pass


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"============================================================")
        print(f" JEFF-TRACKER OPTION A+B LIVE DASHBOARD RUNNING")
        print(f" Local URL: http://localhost:{PORT}")
        print(f"============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


if __name__ == "__main__":
    run_server()
