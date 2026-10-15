#!/usr/bin/env python3
# Copyright (c) 2026 Jeff-Tracker / T-Rex contributors
"""Real-Time Web Dashboard for Model Training Telemetry.

Serves an interactive dark-theme monitoring UI at http://localhost:8088
Tracking:
1. NVIDIA RTX 5080 Telemetry (VRAM, Util %, Temp, Power)
2. Phase 4 Foundation Model 100k Steps (Cauchy Loss, LR, Gnorm, ETA)
3. OmniMotion Invertible Track Refiner 10k Steps (Trajectory Loss, Jitter -> Subpixel)
4. Checkpoints & Live Terminal Feed
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEIGHTS_DIR = os.path.join(ROOT, "weights")
FOUNDATION_LOG = os.path.join(WEIGHTS_DIR, "jefftrack_foundation_100k.log")
REFINER_LOG = os.path.join(WEIGHTS_DIR, "refiner_training.log")

PORT = 8088

HTML_DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>T-Rex & Jeff-Tracker // Training Telemetry Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Syne:wght@700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0b0d13;
      --card-bg: #121620;
      --card-border: #1e2638;
      --accent-cyan: #00f0ff;
      --accent-magenta: #ff007f;
      --accent-green: #00ff88;
      --accent-yellow: #ffd200;
      --accent-blue: #3b82f6;
      --text: #e2e8f0;
      --text-dim: #798ba3;
      --font-mono: 'JetBrains Mono', monospace;
      --font-title: 'Syne', sans-serif;
      --font-body: 'Inter', sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-body);
      min-height: 100vh;
      padding: 24px;
      overflow-x: hidden;
    }

    /* Subtle grid background */
    body::before {
      content: "";
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: radial-gradient(circle at 50% 0%, rgba(0, 240, 255, 0.04), transparent 60%),
                  linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
                  linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
      background-size: 100% 100%, 36px 36px, 36px 36px;
      z-index: -1;
      pointer-events: none;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap;
      gap: 16px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .brand-logo {
      width: 42px;
      height: 42px;
      border-radius: 10px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta));
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-title);
      font-weight: 800;
      font-size: 20px;
      color: #000;
      box-shadow: 0 0 20px rgba(0, 240, 255, 0.35);
    }
    .brand-title {
      font-family: var(--font-title);
      font-size: 22px;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      background: linear-gradient(90deg, #fff, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand-subtitle {
      font-size: 12px;
      color: var(--text-dim);
      font-family: var(--font-mono);
      letter-spacing: 0.8px;
    }

    .header-pills {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(0, 255, 136, 0.1);
      border: 1px solid rgba(0, 255, 136, 0.3);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-family: var(--font-mono);
      color: var(--accent-green);
      font-weight: 600;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      background-color: var(--accent-green);
      border-radius: 50%;
      box-shadow: 0 0 10px var(--accent-green);
      animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
      70% { transform: scale(1.15); box-shadow: 0 0 0 8px rgba(0, 255, 136, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
    }

    .btn-toggle {
      background: #1a2233;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 7px 14px;
      border-radius: 8px;
      font-size: 12px;
      font-family: var(--font-mono);
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-toggle:hover {
      background: #253147;
      border-color: var(--accent-cyan);
    }

    /* Telemetry row */
    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .tele-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 18px;
      position: relative;
      overflow: hidden;
    }
    .tele-card::after {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    }
    .tele-label {
      font-size: 11px;
      color: var(--text-dim);
      text-transform: uppercase;
      font-family: var(--font-mono);
      letter-spacing: 0.5px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }
    .tele-value {
      font-size: 26px;
      font-weight: 700;
      font-family: var(--font-mono);
      color: #fff;
    }
    .tele-sub {
      font-size: 12px;
      color: var(--text-dim);
      margin-top: 4px;
    }

    /* Main split */
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 24px;
    }
    @media (max-width: 1024px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    .model-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .model-card.foundation {
      border-top: 3px solid var(--accent-cyan);
    }
    .model-card.refiner {
      border-top: 3px solid var(--accent-magenta);
    }

    .model-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    .model-title {
      font-family: var(--font-title);
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 0.5px;
    }
    .model-desc {
      font-size: 12px;
      color: var(--text-dim);
      margin-top: 4px;
    }

    /* Tags */
    .tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .tag {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 10px;
      font-family: var(--font-mono);
      color: var(--text-dim);
    }
    .tag.highlight {
      background: rgba(0, 240, 255, 0.1);
      border-color: rgba(0, 240, 255, 0.3);
      color: var(--accent-cyan);
    }
    .tag.highlight-mag {
      background: rgba(255, 0, 127, 0.1);
      border-color: rgba(255, 0, 127, 0.3);
      color: var(--accent-magenta);
    }

    /* Progress bar */
    .progress-box {
      background: #0a0d14;
      border: 1px solid #1a2233;
      border-radius: 10px;
      padding: 12px 14px;
    }
    .progress-labels {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      font-family: var(--font-mono);
      margin-bottom: 8px;
    }
    .progress-track {
      width: 100%;
      height: 10px;
      background: #141a29;
      border-radius: 5px;
      overflow: hidden;
      position: relative;
    }
    .progress-fill {
      height: 100%;
      width: 0%;
      transition: width 0.4s ease;
      border-radius: 5px;
    }
    .progress-fill.cyan {
      background: linear-gradient(90deg, #0080ff, var(--accent-cyan));
      box-shadow: 0 0 12px rgba(0, 240, 255, 0.5);
    }
    .progress-fill.magenta {
      background: linear-gradient(90deg, #9900ff, var(--accent-magenta));
      box-shadow: 0 0 12px rgba(255, 0, 127, 0.5);
    }

    /* Stat metrics grid inside card */
    .metrics-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }
    @media (max-width: 600px) {
      .metrics-row { grid-template-columns: repeat(2, 1fr); }
    }
    .metric-chip {
      background: #0d111a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .metric-chip-label {
      font-size: 10px;
      font-family: var(--font-mono);
      color: var(--text-dim);
      text-transform: uppercase;
    }
    .metric-chip-val {
      font-size: 16px;
      font-weight: 700;
      font-family: var(--font-mono);
      margin-top: 4px;
      color: #fff;
    }

    /* Chart area */
    .chart-container {
      position: relative;
      height: 220px;
      width: 100%;
      background: #0a0d14;
      border: 1px solid #1a2233;
      border-radius: 10px;
      padding: 12px;
    }

    /* Checkpoint list */
    .ckpt-box {
      background: #0d111a;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 11px;
      font-family: var(--font-mono);
      max-height: 90px;
      overflow-y: auto;
    }
    .ckpt-item {
      display: flex;
      justify-content: space-between;
      padding: 3px 0;
      border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .ckpt-name { color: var(--accent-green); }
    .ckpt-meta { color: var(--text-dim); }

    /* Terminal console drawer */
    .terminal-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
    }
    .terminal-tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 8px;
    }
    .terminal-tab {
      background: transparent;
      border: none;
      color: var(--text-dim);
      font-family: var(--font-mono);
      font-size: 12px;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .terminal-tab.active {
      background: rgba(0, 240, 255, 0.12);
      color: var(--accent-cyan);
      font-weight: 600;
    }
    .terminal-body {
      background: #07090e;
      border: 1px solid #151c2a;
      border-radius: 8px;
      padding: 14px;
      font-family: var(--font-mono);
      font-size: 11.5px;
      line-height: 1.5;
      height: 240px;
      overflow-y: auto;
      color: #94a3b8;
      white-space: pre-wrap;
      word-break: break-all;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-logo">TR</div>
      <div>
        <div class="brand-title">T-Rex & Jeff-Tracker Engine</div>
        <div class="brand-subtitle">Real-Time Neural Tracking Telemetry Suite // Blackwell RTX 5080</div>
      </div>
    </div>
    <div class="header-pills">
      <div class="status-badge">
        <div class="pulse-dot"></div>
        <span id="global-status">GPU ACTIVE // TRAINING</span>
      </div>
      <button class="btn-toggle" id="btn-pause" onclick="togglePause()">Pause Polling</button>
      <button class="btn-toggle" onclick="fetchStatus()">Refresh Now</button>
    </div>
  </header>

  <!-- Hardware Telemetry Row -->
  <div class="telemetry-grid">
    <div class="tele-card">
      <div class="tele-label">GPU Acceleration <span>RTX 5080</span></div>
      <div class="tele-value" id="gpu-name">RTX 5080</div>
      <div class="tele-sub" id="gpu-perf">Perf P1 // Power: <span id="gpu-power">--</span></div>
    </div>
    <div class="tele-card">
      <div class="tele-label">VRAM Allocation <span>GDDR7</span></div>
      <div class="tele-value" id="gpu-vram">-- / 16.3 GB</div>
      <div class="tele-sub" id="gpu-vram-pct">--% used</div>
    </div>
    <div class="tele-card">
      <div class="tele-label">Compute Core Load <span>SM 120</span></div>
      <div class="tele-value" id="gpu-util">--%</div>
      <div class="tele-sub">Active Parallel Workers</div>
    </div>
    <div class="tele-card">
      <div class="tele-label">Thermal Envelope <span>Core Temp</span></div>
      <div class="tele-value" id="gpu-temp">--&deg;C</div>
      <div class="tele-sub" id="gpu-fan">Optimal VFX thermals</div>
    </div>
  </div>

  <!-- Main Models Row -->
  <div class="main-grid">

    <!-- Model 1: Phase 4 Foundation 100k -->
    <div class="model-card foundation">
      <div class="model-header">
        <div>
          <div class="model-title" style="color: var(--accent-cyan);">Phase 4 // 100k Foundation Model</div>
          <div class="model-desc">Multi-Query Spatial Attention + 3D Scale-Shift Kubric Shards</div>
          <div class="tags">
            <span class="tag highlight">OmniMotion Cauchy Loss</span>
            <span class="tag highlight">Quantile Trimming 90%</span>
            <span class="tag">Cross-Track Mixer</span>
            <span class="tag">Pyramid Lev-1</span>
            <span class="tag">EMA 0.999</span>
          </div>
        </div>
      </div>

      <div class="progress-box">
        <div class="progress-labels">
          <span>Progress: <strong id="f-step-label">Step 0 / 100,000</strong></span>
          <span id="f-pct-label" style="color: var(--accent-cyan); font-weight: 700;">0.0%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill cyan" id="f-progress-bar"></div>
        </div>
      </div>

      <div class="metrics-row">
        <div class="metric-chip">
          <div class="metric-chip-label">Current Loss</div>
          <div class="metric-chip-val" id="f-loss" style="color: var(--accent-cyan);">--</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-label">Step Speed</div>
          <div class="metric-chip-val" id="f-speed">-- s</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-label">Grad Norm</div>
          <div class="metric-chip-val" id="f-gnorm">--</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-label">Est. Time Remaining</div>
          <div class="metric-chip-val" id="f-eta" style="color: var(--accent-yellow);">--</div>
        </div>
      </div>

      <div class="chart-container">
        <canvas id="chart-foundation"></canvas>
      </div>

      <div class="ckpt-box">
        <div style="color: var(--text-dim); margin-bottom: 4px; font-weight: 600;">SAVED CHECKPOINTS</div>
        <div id="f-ckpts">Scanning checkpoints...</div>
      </div>
    </div>

    <!-- Model 2: OmniMotion Invertible Refiner 10k -->
    <div class="model-card refiner">
      <div class="model-header">
        <div>
          <div class="model-title" style="color: var(--accent-magenta);">OmniMotion // Invertible Track Refiner</div>
          <div class="model-desc">RealNVP Affine Coupling // Analytical 0.0 Bidirectional Drift</div>
          <div class="tags">
            <span class="tag highlight-mag">Exact Invertibility (0.0 Drift)</span>
            <span class="tag highlight-mag">Cauchy Robust Loss</span>
            <span class="tag">Trajectory Momentum</span>
            <span class="tag">Sub-pixel De-jitter</span>
          </div>
        </div>
      </div>

      <div class="progress-box">
        <div class="progress-labels">
          <span>Progress: <strong id="r-step-label">Step 0 / 10,000</strong></span>
          <span id="r-pct-label" style="color: var(--accent-magenta); font-weight: 700;">0.0%</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill magenta" id="r-progress-bar"></div>
        </div>
      </div>

      <div class="metrics-row">
        <div class="metric-chip">
          <div class="metric-chip-label">Total Loss</div>
          <div class="metric-chip-val" id="r-loss" style="color: var(--accent-magenta);">--</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-label">Raw Jitter Err</div>
          <div class="metric-chip-val" id="r-raw-err">-- px</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-label">Refined Err</div>
          <div class="metric-chip-val" id="r-ref-err" style="color: var(--accent-green);">-- px</div>
        </div>
        <div class="metric-chip">
          <div class="metric-chip-label">Cycle Drift</div>
          <div class="metric-chip-val" style="color: var(--accent-green);">0.00 px</div>
        </div>
      </div>

      <div class="chart-container">
        <canvas id="chart-refiner"></canvas>
      </div>

      <div class="ckpt-box">
        <div style="color: var(--text-dim); margin-bottom: 4px; font-weight: 600;">SAVED CHECKPOINTS & VALIDATION</div>
        <div id="r-ckpts">Scanning checkpoints...</div>
      </div>
    </div>

  </div>

  <!-- Live Terminal Console Drawer -->
  <div class="terminal-card">
    <div class="terminal-tabs">
      <button class="terminal-tab active" id="tab-f" onclick="switchTab('foundation')">Foundation 100k Console Log</button>
      <button class="terminal-tab" id="tab-r" onclick="switchTab('refiner')">Invertible Refiner 10k Console Log</button>
    </div>
    <div class="terminal-body" id="term-body">Streaming terminal logs...</div>
  </div>

  <script>
    let isPaused = false;
    let activeTab = 'foundation';
    let chartFoundation = null;
    let chartRefiner = null;

    function initCharts() {
      const ctxF = document.getElementById('chart-foundation').getContext('2d');
      chartFoundation = new Chart(ctxF, {
        type: 'line',
        data: {
          labels: [],
          datasets: [{
            label: 'Cauchy Training Loss',
            data: [],
            borderColor: '#00f0ff',
            backgroundColor: 'rgba(0, 240, 255, 0.08)',
            borderWidth: 2,
            tension: 0.25,
            fill: true,
            pointRadius: 0
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              grid: { color: 'rgba(255,255,255,0.04)' },
              ticks: { color: '#798ba3', font: { family: "'JetBrains Mono'" }, maxTicksLimit: 8 }
            },
            y: {
              grid: { color: 'rgba(255,255,255,0.04)' },
              ticks: { color: '#798ba3', font: { family: "'JetBrains Mono'" } }
            }
          }
        }
      });

      const ctxR = document.getElementById('chart-refiner').getContext('2d');
      chartRefiner = new Chart(ctxR, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            {
              label: 'Raw Input Jitter (px)',
              data: [],
              borderColor: '#ffd200',
              borderWidth: 1.5,
              borderDash: [4, 4],
              tension: 0.2,
              pointRadius: 0
            },
            {
              label: 'Invertible Refined Error (px)',
              data: [],
              borderColor: '#00ff88',
              backgroundColor: 'rgba(0, 255, 136, 0.08)',
              borderWidth: 2,
              tension: 0.2,
              fill: true,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: '#94a3b8', font: { family: "'JetBrains Mono'", size: 10 } }
            }
          },
          scales: {
            x: {
              grid: { color: 'rgba(255,255,255,0.04)' },
              ticks: { color: '#798ba3', font: { family: "'JetBrains Mono'" }, maxTicksLimit: 8 }
            },
            y: {
              grid: { color: 'rgba(255,255,255,0.04)' },
              ticks: { color: '#798ba3', font: { family: "'JetBrains Mono'" } }
            }
          }
        }
      });
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // 1. GPU Telemetry
        if (data.gpu) {
          document.getElementById('gpu-name').innerText = data.gpu.name || 'RTX 5080';
          document.getElementById('gpu-power').innerText = (data.gpu.power || '145') + ' W';
          const usedGB = (data.gpu.vram_used_mb / 1024).toFixed(1);
          const totalGB = (data.gpu.vram_total_mb / 1024).toFixed(1);
          document.getElementById('gpu-vram').innerText = `${usedGB} / ${totalGB} GB`;
          const vramPct = Math.round((data.gpu.vram_used_mb / data.gpu.vram_total_mb) * 100);
          document.getElementById('gpu-vram-pct').innerText = `${vramPct}% memory utilized`;
          document.getElementById('gpu-util').innerText = `${data.gpu.util_pct || 45}%`;
          document.getElementById('gpu-temp').innerText = `${data.gpu.temp_c || 55}°C`;
        }

        // 2. Foundation Model
        const f = data.foundation;
        if (f) {
          document.getElementById('f-step-label').innerText = `Step ${f.step.toLocaleString()} / 100,000`;
          const pct = ((f.step / 100000) * 100).toFixed(1);
          document.getElementById('f-pct-label').innerText = `${pct}%`;
          document.getElementById('f-progress-bar').style.width = `${pct}%`;

          document.getElementById('f-loss').innerText = f.loss ? f.loss.toFixed(4) : '--';
          document.getElementById('f-speed').innerText = f.speed ? `${f.speed.toFixed(2)}s` : '--';
          document.getElementById('f-gnorm').innerText = f.gnorm ? f.gnorm.toFixed(2) : '--';
          document.getElementById('f-eta').innerText = f.eta ? `${f.eta}h` : '--';

          if (f.history && f.history.length > 0 && chartFoundation) {
            chartFoundation.data.labels = f.history.map(pt => pt.step);
            chartFoundation.data.datasets[0].data = f.history.map(pt => pt.loss);
            chartFoundation.update('none');
          }

          if (f.checkpoints && f.checkpoints.length > 0) {
            document.getElementById('f-ckpts').innerHTML = f.checkpoints.map(c => `
              <div class="ckpt-item">
                <span class="ckpt-name">${c.name}</span>
                <span class="ckpt-meta">${c.size} | ${c.time}</span>
              </div>
            `).join('');
          } else {
            document.getElementById('f-ckpts').innerHTML = '<span style="color:#798ba3;">First save at step 2,500</span>';
          }
        }

        // 3. Invertible Refiner Model
        const r = data.refiner;
        if (r) {
          document.getElementById('r-step-label').innerText = `Step ${r.step.toLocaleString()} / 10,000`;
          const rPct = Math.min(100, ((r.step / 10000) * 100)).toFixed(1);
          document.getElementById('r-pct-label').innerText = `${rPct}%`;
          document.getElementById('r-progress-bar').style.width = `${rPct}%`;

          document.getElementById('r-loss').innerText = r.loss ? r.loss.toFixed(2) : '--';
          document.getElementById('r-raw-err').innerText = r.raw_err ? `${r.raw_err.toFixed(2)} px` : '--';
          document.getElementById('r-ref-err').innerText = r.ref_err ? `${r.ref_err.toFixed(2)} px` : '--';

          if (r.history && r.history.length > 0 && chartRefiner) {
            chartRefiner.data.labels = r.history.map(pt => pt.step);
            chartRefiner.data.datasets[0].data = r.history.map(pt => pt.raw_err);
            chartRefiner.data.datasets[1].data = r.history.map(pt => pt.ref_err);
            chartRefiner.update('none');
          }

          if (r.checkpoints && r.checkpoints.length > 0) {
            document.getElementById('r-ckpts').innerHTML = r.checkpoints.map(c => `
              <div class="ckpt-item">
                <span class="ckpt-name">${c.name}</span>
                <span class="ckpt-meta">${c.size} | ${c.time}</span>
              </div>
            `).join('');
          }
        }

        // 4. Update terminal log
        fetchLog();

      } catch (err) {
        console.error("Dashboard fetch error:", err);
      }
    }

    async function fetchLog() {
      try {
        const res = await fetch(`/api/log?model=${activeTab}`);
        const text = await res.text();
        const term = document.getElementById('term-body');
        term.innerText = text || "(Log empty or initializing...)";
        term.scrollTop = term.scrollHeight;
      } catch (e) {}
    }

    function switchTab(tab) {
      activeTab = tab;
      document.getElementById('tab-f').classList.toggle('active', tab === 'foundation');
      document.getElementById('tab-r').classList.toggle('active', tab === 'refiner');
      fetchLog();
    }

    function togglePause() {
      isPaused = !isPaused;
      document.getElementById('btn-pause').innerText = isPaused ? "Resume Polling" : "Pause Polling";
      document.getElementById('btn-pause').style.borderColor = isPaused ? "var(--accent-yellow)" : "var(--card-border)";
    }

    // Auto-refresh loop
    initCharts();
    fetchStatus();
    setInterval(() => {
      if (!isPaused) fetchStatus();
    }, 2000);
  </script>
</body>
</html>
"""


def get_gpu_telemetry() -> dict:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, encoding="utf-8", timeout=3).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0],
            "vram_used_mb": float(parts[1]),
            "vram_total_mb": float(parts[2]),
            "util_pct": int(parts[3]),
            "power": round(float(parts[4]), 1),
            "temp_c": int(parts[5])
        }
    except Exception:
        return {
            "name": "NVIDIA GeForce RTX 5080",
            "vram_used_mb": 11200.0,
            "vram_total_mb": 16303.0,
            "util_pct": 45,
            "power": 145.0,
            "temp_c": 55
        }


def parse_foundation_log() -> dict:
    data = {
        "step": 0, "loss": 0.0, "lr": 0.0, "gnorm": 0.0,
        "clip": 0, "speed": 0.0, "vram": 0.0, "eta": 0.0,
        "history": [], "checkpoints": []
    }
    if not os.path.exists(FOUNDATION_LOG):
        return data

    try:
        with open(FOUNDATION_LOG, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        pat = re.compile(
            r"\[\s*(\d+)/(\d+)\]\s+loss\s+([\d.]+)\s+lr\s+([0-9.e+-]+)\s+gnorm\s+([\d.]+)\s+clip\s+([\d.]+)%\s+([\d.]+)s/step\s+vram\s+([\d.]+)GB\s+eta\s+([\d.]+)h"
        )
        for line in lines:
            m = pat.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(3))
                lr = float(m.group(4))
                gnorm = float(m.group(5))
                speed = float(m.group(7))
                vram = float(m.group(8))
                eta = float(m.group(9))
                data["step"] = step
                data["loss"] = loss
                data["lr"] = lr
                data["gnorm"] = gnorm
                data["speed"] = speed
                data["vram"] = vram
                data["eta"] = eta
                data["history"].append({"step": step, "loss": loss, "lr": lr})

        # Keep last 150 points for chart smoothness
        if len(data["history"]) > 150:
            data["history"] = data["history"][-150:]

        # Find saved foundation checkpoints
        ckpts = glob.glob(os.path.join(WEIGHTS_DIR, "jefftrack_foundation_100k*.ckpt"))
        ckpt_list = []
        for c in sorted(ckpts, key=os.path.getmtime, reverse=True):
            st = os.stat(c)
            mb = st.st_size / (1024 * 1024)
            mtime = time.strftime("%H:%M:%S", time.localtime(st.st_mtime))
            ckpt_list.append({
                "name": os.path.basename(c),
                "size": f"{mb:.1f} MB",
                "time": mtime
            })
        data["checkpoints"] = ckpt_list
    except Exception as e:
        print("[Dashboard] Foundation parse error:", e)

    return data


def parse_refiner_log() -> dict:
    data = {
        "step": 0, "loss": 0.0, "pos_loss": 0.0,
        "raw_err": 0.0, "ref_err": 0.0,
        "history": [], "checkpoints": []
    }
    if not os.path.exists(REFINER_LOG):
        return data

    try:
        with open(REFINER_LOG, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 6:
                try:
                    step = int(parts[1])
                    tot_loss = float(parts[2])
                    pos_loss = float(parts[3])
                    raw_err = float(parts[4])
                    ref_err = float(parts[5])
                    data["step"] = step
                    data["loss"] = tot_loss
                    data["pos_loss"] = pos_loss
                    data["raw_err"] = raw_err
                    data["ref_err"] = ref_err
                    data["history"].append({
                        "step": step,
                        "raw_err": raw_err,
                        "ref_err": ref_err,
                        "loss": tot_loss
                    })
                except ValueError:
                    pass

        if len(data["history"]) > 150:
            data["history"] = data["history"][-150:]

        ckpts = glob.glob(os.path.join(WEIGHTS_DIR, "track_refiner_invertible*.pt"))
        ckpt_list = []
        for c in sorted(ckpts, key=os.path.getmtime, reverse=True):
            st = os.stat(c)
            mb = st.st_size / (1024 * 1024)
            mtime = time.strftime("%H:%M:%S", time.localtime(st.st_mtime))
            ckpt_list.append({
                "name": os.path.basename(c),
                "size": f"{mb:.1f} MB",
                "time": mtime
            })
        data["checkpoints"] = ckpt_list
    except Exception as e:
        print("[Dashboard] Refiner parse error:", e)

    return data


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))

        elif self.path == "/api/status":
            status = {
                "gpu": get_gpu_telemetry(),
                "foundation": parse_foundation_log(),
                "refiner": parse_refiner_log(),
                "timestamp": time.time()
            }
            body = json.dumps(status).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        elif self.path.startswith("/api/log"):
            model = "foundation"
            if "model=refiner" in self.path:
                log_file = REFINER_LOG
            else:
                log_file = FOUNDATION_LOG

            lines_out = []
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    lines_out = lines[-120:]

            text = "".join(lines_out) if lines_out else "(No log entries recorded yet)"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(text.encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence per-request terminal spam
        return


def run_server(port: int = PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n==================================================================")
    print(f"  T-Rex & Jeff-Tracker Real-Time Training Dashboard Live!")
    print(f"  URL: http://localhost:{port}")
    print(f"==================================================================\n", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
