import os
import sys
import time
import math
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

FFMPEG_EXE = r"C:\Users\PUBLIC.SURAJ-TOMAR\.unveo\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v9.0.2.exe"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VERIQ - Cinematic Intro (8s)</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body, html {
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    background: #02040b;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
  }

  #canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 1920px;
    height: 1080px;
    display: block;
  }
</style>
</head>
<body>
<canvas id="canvas" width="1920" height="1080"></canvas>

<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Pre-create floating cards data
const cards = [
  { x: 380,  y: 340, z: 120, w: 340, h: 180, rot: -0.06, type: "expired", label: "Claim: Latest iPhone model", claim: "The latest iPhone is the iPhone 17...", turnGreenAt: 3.0 },
  { x: 1480, y: 310, z: -80, w: 320, h: 170, rot: 0.05,  type: "normal",  label: "Claim: GST Tax Slabs", claim: "India GST has 5%, 12%, 18%, 28%...", turnGreenAt: 2.2 },
  { x: 420,  y: 720, z: -40, w: 300, h: 160, rot: 0.04,  type: "normal",  label: "Claim: OpenAI Model", claim: "GPT-4 is OpenAI's latest model...", turnGreenAt: 2.6 },
  { x: 1450, y: 740, z: 100, w: 330, h: 175, rot: -0.04, type: "normal",  label: "Claim: DeepSeek V3 Specs", claim: "671B MoE architecture with MLA...", turnGreenAt: 3.4 },
  { x: 960,  y: 220, z: -150, w: 310, h: 160, rot: 0.02, type: "normal",  label: "Claim: US Federal Rate", claim: "Target federal funds rate at 5.25%...", turnGreenAt: 3.7 },
];

// Pre-create dust particles for expired card
const dustParticles = [];
for (let i = 0; i < 90; i++) {
  dustParticles.push({
    rx: (Math.random() - 0.5) * 280,
    ry: (Math.random() - 0.5) * 120,
    speedY: Math.random() * 60 + 30,
    speedX: (Math.random() - 0.5) * 40,
    size: Math.random() * 2.5 + 1.0,
    alpha: Math.random() * 0.7 + 0.3,
    seed: Math.random() * 10
  });
}

// Background stars/dust
const bgStars = [];
for (let i = 0; i < 140; i++) {
  bgStars.push({
    x: Math.random() * 1920,
    y: Math.random() * 1080,
    size: Math.random() * 1.8 + 0.5,
    alpha: Math.random() * 0.6 + 0.2
  });
}

// Helper: ease in out cubic
function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

// Main frame render function driven by timestamp t (0.0 to 8.0s)
window.renderFrame = function(t) {
  ctx.save();
  ctx.clearRect(0, 0, 1920, 1080);

  // 1. Deep Space Navy Background
  let bgGradient = ctx.createRadialGradient(960, 540, 50, 960, 540, 1050);
  bgGradient.addColorStop(0, "#080c26");
  bgGradient.addColorStop(0.5, "#040616");
  bgGradient.addColorStop(1, "#02030a");
  ctx.fillStyle = bgGradient;
  ctx.fillRect(0, 0, 1920, 1080);

  // Background stars
  ctx.fillStyle = "#ffffff";
  for (const s of bgStars) {
    ctx.globalAlpha = s.alpha * Math.min(1, Math.max(0, (t - 0.4) * 2));
    ctx.fillRect(s.x, s.y, s.size, s.size);
  }
  ctx.globalAlpha = 1.0;

  // 2. Violet Volumetric Rays (0.4s to 8.0s)
  if (t > 0.3) {
    const rayAlpha = Math.min(0.25, (t - 0.3) * 0.2);
    ctx.save();
    ctx.globalAlpha = rayAlpha;
    const rayGrad1 = ctx.createRadialGradient(350, 200, 10, 350, 200, 700);
    rayGrad1.addColorStop(0, "rgba(168, 85, 247, 0.45)");
    rayGrad1.addColorStop(1, "transparent");
    ctx.fillStyle = rayGrad1;
    ctx.beginPath();
    ctx.arc(350, 200, 700, 0, Math.PI * 2);
    ctx.fill();

    const rayGrad2 = ctx.createRadialGradient(1550, 850, 10, 1550, 850, 750);
    rayGrad2.addColorStop(0, "rgba(52, 211, 153, 0.35)");
    rayGrad2.addColorStop(1, "transparent");
    ctx.fillStyle = rayGrad2;
    ctx.beginPath();
    ctx.arc(1550, 850, 750, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // 3. PHASE 0-2s: Initial Violet Sweep (0.2s - 1.0s)
  if (t >= 0.2 && t <= 1.3) {
    const sweepProg = clamp((t - 0.2) / 0.8, 0, 1);
    const beamX = sweepProg * 2200 - 100;

    ctx.save();
    ctx.shadowBlur = 35;
    ctx.shadowColor = "#c084fc";
    ctx.strokeStyle = "rgba(216, 180, 254, 0.95)";
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.moveTo(beamX, 0);
    ctx.lineTo(beamX + 180, 1080);
    ctx.stroke();

    // Beam glow streak
    const streakGrad = ctx.createLinearGradient(beamX - 120, 0, beamX + 220, 0);
    streakGrad.addColorStop(0, "transparent");
    streakGrad.addColorStop(0.5, "rgba(168, 85, 247, 0.45)");
    streakGrad.addColorStop(1, "transparent");
    ctx.fillStyle = streakGrad;
    ctx.fillRect(beamX - 120, 0, 340, 1080);
    ctx.restore();
  }

  // Camera Zoom / Dolly calculation (Push forward from 4.0s to 6.0s)
  let camScale = 1.0;
  let camOffsetY = 0;
  if (t > 4.0 && t <= 6.2) {
    const p = clamp((t - 4.0) / 2.0, 0, 1);
    const ep = easeInOutCubic(p);
    camScale = 1.0 + ep * 0.45;
    camOffsetY = ep * -20;
  } else if (t > 6.2) {
    camScale = 1.45;
    camOffsetY = -20;
  }

  ctx.save();
  ctx.translate(960, 540 + camOffsetY);
  ctx.scale(camScale, camScale);
  ctx.translate(-960, -540);

  // 4. FLOATING GLASS CARDS (Visible from 0.6s to ~5.4s)
  if (t > 0.5 && t < 5.8) {
    const cardsAlpha = t < 1.0 ? (t - 0.5) / 0.5 : (t > 5.0 ? clamp((5.6 - t) / 0.6, 0, 1) : 1.0);
    
    // Check if cards are snapping together into shield (t=4.6 to 5.4)
    const snapProgress = t < 4.6 ? 0 : clamp((t - 4.6) / 0.7, 0, 1);
    const snapEase = easeInOutCubic(snapProgress);

    for (let idx = 0; idx < cards.length; idx++) {
      const card = cards[idx];
      
      // Calculate position
      let curX = card.x;
      let curY = card.y;
      let curW = card.w;
      let curH = card.h;
      let curRot = card.rot;

      if (snapProgress > 0) {
        // Converge toward center (960, 540)
        curX = card.x + (960 - card.x) * snapEase;
        curY = card.y + (540 - card.y) * snapEase;
        curW = card.w * (1 - snapEase * 0.85);
        curH = card.h * (1 - snapEase * 0.85);
        curRot = card.rot * (1 - snapEase);
      }

      ctx.save();
      ctx.translate(curX, curY);
      ctx.rotate(curRot);
      ctx.globalAlpha = cardsAlpha * (1 - snapEase * 0.9);

      // Determine card state: expired (red/amber) vs scanned fresh (green) vs normal
      let isExpiredCard = (card.type === "expired");
      let turnedGreen = (t >= card.turnGreenAt);
      
      let borderGrad = ctx.createLinearGradient(-curW/2, -curH/2, curW/2, curH/2);
      let cardBg = "rgba(10, 16, 38, 0.75)";
      let glowColor = "rgba(168, 85, 247, 0.25)";
      let textColor = "#e2e8f0";
      let statusTag = "UNVERIFIED";
      let statusColor = "#94a3b8";

      if (turnedGreen) {
        // Transformed into fresh green
        borderGrad.addColorStop(0, "#34d399");
        borderGrad.addColorStop(1, "#059669");
        cardBg = "rgba(6, 32, 24, 0.82)";
        glowColor = "rgba(52, 211, 153, 0.6)";
        statusTag = "VERIFIED FRESH";
        statusColor = "#34d399";
      } else if (isExpiredCard) {
        if (t >= 1.0 && t < 1.4) {
          // Warning amber
          borderGrad.addColorStop(0, "#fbbf24");
          borderGrad.addColorStop(1, "#b45309");
          cardBg = "rgba(35, 20, 6, 0.82)";
          glowColor = "rgba(245, 158, 11, 0.5)";
          statusTag = "STALE WARNING";
          statusColor = "#f59e0b";
        } else if (t >= 1.4) {
          // Warning red with flicker
          const flicker = Math.sin(t * 30) > 0 ? 1.0 : 0.65;
          borderGrad.addColorStop(0, "#f87171");
          borderGrad.addColorStop(1, "#dc2626");
          cardBg = "rgba(38, 8, 12, 0.85)";
          glowColor = `rgba(239, 68, 68, ${0.7 * flicker})`;
          statusTag = "EXPIRED (STALE)";
          statusColor = "#ef4444";
        } else {
          borderGrad.addColorStop(0, "rgba(255, 255, 255, 0.25)");
          borderGrad.addColorStop(1, "rgba(168, 85, 247, 0.3)");
        }
      } else {
        borderGrad.addColorStop(0, "rgba(255, 255, 255, 0.2)");
        borderGrad.addColorStop(1, "rgba(147, 197, 253, 0.25)");
      }

      // Draw Glass Card Body
      ctx.shadowBlur = turnedGreen ? 24 : (isExpiredCard && t >= 1.4 ? 28 : 12);
      ctx.shadowColor = glowColor;
      ctx.fillStyle = cardBg;
      ctx.strokeStyle = borderGrad;
      ctx.lineWidth = turnedGreen ? 2.5 : 1.8;

      ctx.beginPath();
      ctx.roundRect(-curW/2, -curH/2, curW, curH, 14);
      ctx.fill();
      ctx.stroke();

      // Card Content (only if not completely shrunken)
      if (curW > 120) {
        ctx.shadowBlur = 0;
        // Header pill
        ctx.fillStyle = statusColor;
        ctx.font = "bold 11px Inter, sans-serif";
        ctx.fillText(statusTag, -curW/2 + 20, -curH/2 + 30);

        // Claim Title
        ctx.fillStyle = textColor;
        ctx.font = "bold 13px Inter, sans-serif";
        ctx.fillText(card.label, -curW/2 + 20, -curH/2 + 55);

        // Claim text body
        ctx.fillStyle = "#94a3b8";
        ctx.font = "12px Inter, sans-serif";
        ctx.fillText(card.claim, -curW/2 + 20, -curH/2 + 82);

        // Metric badge
        ctx.fillStyle = turnedGreen ? "rgba(52, 211, 153, 0.2)" : "rgba(255, 255, 255, 0.08)";
        ctx.beginPath();
        ctx.roundRect(-curW/2 + 20, -curH/2 + 105, 120, 24, 6);
        ctx.fill();

        ctx.fillStyle = turnedGreen ? "#34d399" : "#cbd5e1";
        ctx.font = "bold 11px Inter, sans-serif";
        ctx.fillText(turnedGreen ? "Score: 100% Fresh" : "Trust: Pending", -curW/2 + 28, -curH/2 + 121);
      }

      // Expired Card: Embers / Dust particles falling down (t=1.4 to 3.0)
      if (isExpiredCard && t >= 1.4 && t < 3.2 && snapProgress === 0) {
        ctx.save();
        const expireProg = (t - 1.4);
        for (const dp of dustParticles) {
          const dy = (dp.ry + expireProg * dp.speedY) % 180;
          const dx = dp.rx + Math.sin(expireProg * 4 + dp.seed) * 20;
          ctx.beginPath();
          ctx.arc(dx, dy + curH/4, dp.size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(239, 68, 68, ${dp.alpha * Math.max(0, 1 - dy / 180)})`;
          ctx.shadowBlur = 8;
          ctx.shadowColor = "#ef4444";
          ctx.fill();
        }
        ctx.restore();
      }

      // Soft shockwave ring on green lock
      if (t >= card.turnGreenAt && t <= card.turnGreenAt + 0.6) {
        const shockP = (t - card.turnGreenAt) / 0.6;
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(-curW/2 - shockP * 40, -curH/2 - shockP * 40, curW + shockP * 80, curH + shockP * 80, 20);
        ctx.strokeStyle = `rgba(52, 211, 153, ${1 - shockP})`;
        ctx.lineWidth = 3 * (1 - shockP);
        ctx.shadowBlur = 18;
        ctx.shadowColor = "#34d399";
        ctx.stroke();
        ctx.restore();
      }

      ctx.restore();
    }
  }

  // 5. PHASE 2-4s: Mint-Green Scanning Beam (t=2.0s to 3.9s)
  if (t >= 2.0 && t <= 4.0) {
    const scanProg = (t - 2.0) / 1.9;
    const scanX = scanProg * 1800 + 60;

    ctx.save();
    // Laser line
    ctx.shadowBlur = 32;
    ctx.shadowColor = "#34d399";
    ctx.strokeStyle = "#6ee7b7";
    ctx.lineWidth = 3.8;
    ctx.beginPath();
    ctx.moveTo(scanX, 80);
    ctx.lineTo(scanX, 1000);
    ctx.stroke();

    // Laser glow aura
    const scanGlow = ctx.createLinearGradient(scanX - 90, 0, scanX + 90, 0);
    scanGlow.addColorStop(0, "transparent");
    scanGlow.addColorStop(0.5, "rgba(52, 211, 153, 0.45)");
    scanGlow.addColorStop(1, "transparent");
    ctx.fillStyle = scanGlow;
    ctx.fillRect(scanX - 90, 80, 180, 920);
    ctx.restore();
  }

  // 6. TWO LIGHT TRAILS: Spiraling Double Helix (Violet & Mint) - (t=2.0s to 5.4s)
  if (t >= 2.0 && t <= 5.4) {
    const helixAlpha = t < 2.5 ? (t - 2.0) / 0.5 : (t > 4.8 ? clamp((5.4 - t) / 0.6, 0, 1) : 1.0);
    ctx.save();
    ctx.globalAlpha = helixAlpha;

    const points = 70;
    const baseRadius = t > 4.4 ? 140 * Math.max(0, (5.4 - t) / 1.0) : 140; // collapse inward into shield
    const length = 1100;
    const startX = 960 - length / 2;

    // Trail 1: Violet (Builder 1)
    ctx.beginPath();
    for (let i = 0; i <= points; i++) {
      const prog = i / points;
      const x = startX + prog * length;
      const angle = prog * Math.PI * 6 + t * 4.5;
      const y = 540 + Math.sin(angle) * baseRadius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#c084fc";
    ctx.lineWidth = 4.0;
    ctx.shadowBlur = 24;
    ctx.shadowColor = "#a855f7";
    ctx.stroke();

    // Trail 2: Mint (Builder 2)
    ctx.beginPath();
    for (let i = 0; i <= points; i++) {
      const prog = i / points;
      const x = startX + prog * length;
      const angle = prog * Math.PI * 6 + t * 4.5 + Math.PI; // 180 deg opposite
      const y = 540 + Math.sin(angle) * baseRadius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 4.0;
    ctx.shadowBlur = 24;
    ctx.shadowColor = "#10b981";
    ctx.stroke();

    ctx.restore();
  }

  // 7. PHASE 4-6s: SHIELD EMBLEM & SEMICIRCULAR GAUGE (t=4.6s to 6.4s)
  if (t >= 4.6 && t <= 6.5) {
    const shieldEnterP = clamp((t - 4.6) / 0.7, 0, 1);
    const shieldEase = easeInOutCubic(shieldEnterP);
    const shieldExitP = t > 5.9 ? clamp((t - 5.9) / 0.5, 0, 1) : 0;
    const shieldAlpha = shieldEase * (1 - shieldExitP);

    ctx.save();
    ctx.translate(960, 540);
    ctx.scale(shieldEase * (1 + shieldExitP * 0.3), shieldEase * (1 + shieldExitP * 0.3));
    ctx.globalAlpha = shieldAlpha;

    // A. Semicircular Freshness Gauge
    const gaugeProg = clamp((t - 4.8) / 0.8, 0, 1); // 0 to 1 sweeps red -> amber -> green
    const gaugeAngle = Math.PI + gaugeProg * Math.PI; // from Math.PI (180deg) to 2*Math.PI (360deg)

    // Gauge Track
    ctx.lineWidth = 14;
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.beginPath();
    ctx.arc(0, -10, 150, Math.PI, 2 * Math.PI, false);
    ctx.stroke();

    // Gauge Active Sweep
    const gaugeGrad = ctx.createLinearGradient(-150, -10, 150, -10);
    gaugeGrad.addColorStop(0, "#ef4444");
    gaugeGrad.addColorStop(0.5, "#f59e0b");
    gaugeGrad.addColorStop(1, "#10b981");

    ctx.strokeStyle = gaugeGrad;
    ctx.shadowBlur = 26;
    ctx.shadowColor = gaugeProg > 0.8 ? "#34d399" : (gaugeProg > 0.4 ? "#f59e0b" : "#ef4444");
    ctx.beginPath();
    ctx.arc(0, -10, 150, Math.PI, gaugeAngle, false);
    ctx.stroke();

    // B. Shield Shape
    ctx.shadowBlur = 40;
    ctx.shadowColor = "rgba(52, 211, 153, 0.75)";
    const shieldGrad = ctx.createLinearGradient(0, -80, 0, 120);
    shieldGrad.addColorStop(0, "rgba(168, 85, 247, 0.35)");
    shieldGrad.addColorStop(0.5, "rgba(52, 211, 153, 0.28)");
    shieldGrad.addColorStop(1, "rgba(6, 78, 59, 0.4)");
    ctx.fillStyle = shieldGrad;

    const strokeGrad = ctx.createLinearGradient(-70, -80, 70, 120);
    strokeGrad.addColorStop(0, "#c084fc");
    strokeGrad.addColorStop(0.5, "#6ee7b7");
    strokeGrad.addColorStop(1, "#34d399");
    ctx.strokeStyle = strokeGrad;
    ctx.lineWidth = 4.5;

    ctx.beginPath();
    ctx.moveTo(0, -80);
    ctx.lineTo(65, -50);
    ctx.lineTo(65, 25);
    ctx.bezierCurveTo(65, 85, 0, 120, 0, 120);
    ctx.bezierCurveTo(0, 120, -65, 85, -65, 25);
    ctx.lineTo(-65, -50);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Inner checkmark icon
    if (t >= 5.3) {
      const checkP = clamp((t - 5.3) / 0.4, 0, 1);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 5.0;
      ctx.shadowBlur = 20;
      ctx.shadowColor = "#34d399";
      ctx.beginPath();
      ctx.moveTo(-24, 15);
      ctx.lineTo(-6 + checkP * 0, 32);
      if (checkP > 0.4) {
        const p2 = (checkP - 0.4) / 0.6;
        ctx.lineTo(-6 + p2 * 34, 32 - p2 * 42);
      }
      ctx.stroke();
    }

    // Lock shockwave pulse at t=5.5s
    if (t >= 5.5 && t <= 6.2) {
      const pwave = (t - 5.5) / 0.7;
      ctx.beginPath();
      ctx.arc(0, 0, 80 + pwave * 140, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(52, 211, 153, ${1 - pwave})`;
      ctx.lineWidth = 3.5 * (1 - pwave);
      ctx.shadowBlur = 25;
      ctx.shadowColor = "#34d399";
      ctx.stroke();
    }

    ctx.restore();
  }

  // 8. PHASE 6-8s: 3D TITLE "VERIQ" & CREDITS REVEAL (t=5.9s to 8.0s)
  if (t >= 5.9) {
    const revealProg = clamp((t - 5.9) / 0.7, 0, 1);
    const revEase = easeInOutCubic(revealProg);

    ctx.save();
    ctx.translate(960, 500);
    ctx.scale(0.85 + revEase * 0.15, 0.85 + revEase * 0.15);
    ctx.globalAlpha = revEase;

    // Violet Ambient Nebula Glow behind VERIQ
    const nebulaGrad = ctx.createRadialGradient(0, -30, 20, 0, -30, 480);
    nebulaGrad.addColorStop(0, "rgba(168, 85, 247, 0.42)");
    nebulaGrad.addColorStop(0.5, "rgba(52, 211, 153, 0.18)");
    nebulaGrad.addColorStop(1, "transparent");
    ctx.fillStyle = nebulaGrad;
    ctx.beginPath();
    ctx.arc(0, -30, 480, 0, Math.PI * 2);
    ctx.fill();

    // 3D VERIQ Typography
    // "VER" in pristine white, "IQ" in luminous mint green
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    // Measure widths to place "VER" and "IQ" side by side
    ctx.font = "900 132px 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    const verText = "VER";
    const iqText = "IQ";
    const verWidth = ctx.measureText(verText).width;
    const iqWidth = ctx.measureText(iqText).width;
    const totalWidth = verWidth + iqWidth;
    const startX = -totalWidth / 2;

    const verCenterX = startX + verWidth / 2;
    const iqCenterX = startX + verWidth + iqWidth / 2;
    const textY = -40;

    // Subtle 3D shadow layers (extrusion effect)
    for (let depth = 6; depth >= 1; depth--) {
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(4, 8, 24, 0.9)";
      ctx.fillText(verText, verCenterX, textY + depth * 1.5);
      ctx.fillText(iqText, iqCenterX, textY + depth * 1.5);
    }

    // Top face of "VER" (Radiant White with soft metallic gradient)
    ctx.save();
    const verGrad = ctx.createLinearGradient(verCenterX, textY - 60, verCenterX, textY + 60);
    verGrad.addColorStop(0, "#ffffff");
    verGrad.addColorStop(0.7, "#f1f5f9");
    verGrad.addColorStop(1, "#cbd5e1");
    ctx.shadowBlur = 35;
    ctx.shadowColor = "rgba(255, 255, 255, 0.75)";
    ctx.fillStyle = verGrad;
    ctx.fillText(verText, verCenterX, textY);
    ctx.restore();

    // Top face of "IQ" (Vibrant Mint Green)
    ctx.save();
    const iqGrad = ctx.createLinearGradient(iqCenterX, textY - 60, iqCenterX, textY + 60);
    iqGrad.addColorStop(0, "#a7f3d0");
    iqGrad.addColorStop(0.5, "#34d399");
    iqGrad.addColorStop(1, "#059669");
    ctx.shadowBlur = 45;
    ctx.shadowColor = "#34d399";
    ctx.fillStyle = iqGrad;
    ctx.fillText(iqText, iqCenterX, textY);
    ctx.restore();

    // Specular Glint across VERIQ (sweeps at t=6.8s to 7.4s)
    if (t >= 6.7 && t <= 7.6) {
      const glintP = (t - 6.7) / 0.9;
      const glintX = startX + glintP * (totalWidth + 120) - 60;
      ctx.save();
      ctx.beginPath();
      ctx.rect(startX - 20, textY - 80, totalWidth + 40, 160);
      ctx.clip();

      const glintGrad = ctx.createLinearGradient(glintX - 40, textY, glintX + 40, textY);
      glintGrad.addColorStop(0, "transparent");
      glintGrad.addColorStop(0.5, "rgba(255, 255, 255, 0.85)");
      glintGrad.addColorStop(1, "transparent");
      ctx.fillStyle = glintGrad;
      ctx.fillRect(glintX - 40, textY - 80, 80, 160);
      ctx.restore();
    }

    // Subtitle: "Stale-Answer Detector"
    ctx.save();
    ctx.font = "600 24px 'Inter', sans-serif";
    ctx.letterSpacing = "6px";
    ctx.shadowBlur = 18;
    ctx.shadowColor = "rgba(168, 85, 247, 0.6)";
    ctx.fillStyle = "#ffffff";
    ctx.fillText("STALE-ANSWER DETECTOR", 0, textY + 95);
    ctx.restore();

    // Bottom Duo Credits: "Archita Jain  ·  Purab Tomar  ·  Tech Duo"
    ctx.save();
    ctx.font = "500 16px 'Inter', sans-serif";
    ctx.letterSpacing = "4px";
    ctx.fillStyle = "rgba(203, 213, 225, 0.9)";
    ctx.shadowBlur = 12;
    ctx.shadowColor = "rgba(52, 211, 153, 0.5)";
    ctx.fillText("ARCHITA JAIN   ·   PURAB TOMAR   ·   TECH DUO", 0, textY + 155);
    ctx.restore();

    ctx.restore();
  }

  ctx.restore(); // restore camera transform
  ctx.restore(); // restore base
};
</script>
</body>
</html>
"""

def generate_cinematic_audio(output_audio_path: Path, duration_s: float = 8.0):
    """
    Synthesizes the exact audio storyboard matching:
    - 0-2s: Deep cinematic sub-bass rumble (42Hz + 55Hz lowpassed) + subtle laser ignite sweep
    - 1.4-1.8s: Soft digital warning flutter for the expired card
    - 2-4s: Soft digital ticks for each card turning green (sine bursts at t=2.2, 2.6, 3.0, 3.4, 3.7)
    - 3-5.5s: Rising synth swell during camera push
    - 5.3s: Crisp whoosh as cards snap together into the shield emblem
    - 6.0s: Confident chime + short warm bass hit on the VERIQ title reveal
    - 6-8s: Resolving warm pad chord sustaining with gentle tail
    """
    print("[Audio] Synthesizing 8.0s cinematic audio track via FFmpeg filtergraph...")

    filter_complex = (
        # 1. Deep Sub-bass rumble (0s - 3s)
        "sine=f=42:d=3.5[sub1]; "
        "sine=f=58:d=3.5[sub2]; "
        "[sub1][sub2]amix=inputs=2[sub_mix]; "
        "[sub_mix]volume=0.38,afade=t=in:st=0:d=0.8,afade=t=out:st=2.5:d=1.0[sub_rumble]; "

        # 2. Subtle laser sweep whoosh (0.3s - 1.0s)
        "anoisesrc=d=0.9:c=pink:r=48000,bandpass=f=1200:width_type=h:w=600,volume=0.18,afade=t=in:st=0:d=0.2,afade=t=out:st=0.5:d=0.4,adelay=300|300[sweep_whoosh]; "

        # 3. Soft digital ticks when cards turn green (t=2.2s, 2.6s, 3.0s, 3.4s, 3.7s)
        "sine=f=2400:d=0.06,afade=t=out:st=0.01:d=0.05,volume=0.22,adelay=2200|2200[tick1]; "
        "sine=f=2700:d=0.06,afade=t=out:st=0.01:d=0.05,volume=0.22,adelay=2600|2600[tick2]; "
        "sine=f=3100:d=0.06,afade=t=out:st=0.01:d=0.05,volume=0.25,adelay=3000|3000[tick3]; "
        "sine=f=2900:d=0.06,afade=t=out:st=0.01:d=0.05,volume=0.22,adelay=3400|3400[tick4]; "
        "sine=f=3400:d=0.06,afade=t=out:st=0.01:d=0.05,volume=0.22,adelay=3700|3700[tick5]; "
        "[tick1][tick2][tick3][tick4][tick5]amix=inputs=5[ticks]; "

        # 4. Rising synth swell during camera push (t=3.2s to 5.4s)
        "sine=f=220:d=2.6,afade=t=in:st=0:d=1.2,afade=t=out:st=2.0:d=0.6,volume=0.25,adelay=3200|3200[swell1]; "
        "sine=f=330:d=2.6,afade=t=in:st=0:d=1.2,afade=t=out:st=2.0:d=0.6,volume=0.20,adelay=3200|3200[swell2]; "
        "[swell1][swell2]amix=inputs=2[swell]; "

        # 5. Crisp snap whoosh into shield (t=5.2s)
        "anoisesrc=d=0.7:c=white:r=48000,bandpass=f=2200:width_type=h:w=1200,volume=0.28,afade=t=in:st=0:d=0.1,afade=t=out:st=0.25:d=0.45,adelay=5200|5200[snap_whoosh]; "

        # 6. VERIQ Reveal Chime (t=6.0s) - Tubular Bell Harmonic Strike (1760Hz, 2637Hz, 3520Hz)
        "sine=f=1760:d=2.5,afade=t=in:st=0:d=0.005,afade=t=out:st=0.05:d=2.4,volume=0.36,adelay=6000|6000[chime1]; "
        "sine=f=2637:d=2.2,afade=t=in:st=0:d=0.005,afade=t=out:st=0.05:d=2.1,volume=0.24,adelay=6000|6000[chime2]; "
        "sine=f=3520:d=1.8,afade=t=in:st=0:d=0.005,afade=t=out:st=0.05:d=1.7,volume=0.15,adelay=6000|6000[chime3]; "
        "[chime1][chime2][chime3]amix=inputs=3[chime_full]; "

        # 7. Warm bass hit on title reveal (t=6.0s)
        "sine=f=65:d=1.5,volume=0.45,afade=t=in:st=0:d=0.01,afade=t=out:st=0.1:d=1.4,adelay=6000|6000[bass_punch]; "

        # 8. Resolving warm pad chord (t=6.0s to 8.0s)
        "sine=f=261.63:d=2.2,volume=0.18,afade=t=in:st=0:d=0.2,afade=t=out:st=1.6:d=0.6,adelay=6000|6000[pad_c]; "
        "sine=f=329.63:d=2.2,volume=0.16,afade=t=in:st=0:d=0.2,afade=t=out:st=1.6:d=0.6,adelay=6000|6000[pad_e]; "
        "sine=f=392.00:d=2.2,volume=0.14,afade=t=in:st=0:d=0.2,afade=t=out:st=1.6:d=0.6,adelay=6000|6000[pad_g]; "
        "[pad_c][pad_e][pad_g]amix=inputs=3[warm_pad]; "

        # Final Master Mix
        "[sub_rumble][sweep_whoosh][ticks][swell][snap_whoosh][chime_full][bass_punch][warm_pad]amix=inputs=8:weights=1.0 0.8 1.1 0.9 1.1 1.3 1.2 0.9[aout]"
    )

    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-t", str(duration_s),
        "-c:a", "aac", "-b:a", "192k",
        str(output_audio_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"[Audio] Cinematic sound design complete: {output_audio_path}")


def main():
    out_dir = Path("unveo-out")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = out_dir / "veriq_intro_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    html_file = out_dir / "veriq_intro.html"
    html_file.write_text(HTML_TEMPLATE, encoding="utf-8")

    audio_file = out_dir / "veriq_intro_audio.aac"
    final_video_file = out_dir / "veriq-cinematic-intro-8s.mp4"
    title_frame_file = out_dir / "veriq-intro-title-frame.png"

    # Step 1: Synthesize Audio
    generate_cinematic_audio(audio_file, duration_s=8.0)

    # Step 2: Frame-by-Frame deterministic capture at 24fps
    # 8.0 seconds @ 24fps = 192 frames (from 0 to 191)
    fps = 24
    total_frames = int(8.0 * fps)
    print(f"[Render] Capturing {total_frames} frames @ {fps}fps (1920x1080) for rock-solid 24fps film motion...")

    start_render = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(html_file.resolve().as_uri(), wait_until="networkidle")

        for frame_idx in range(total_frames):
            t_sec = frame_idx / float(fps)
            page.evaluate(f"window.renderFrame({t_sec})")
            frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
            page.screenshot(path=str(frame_path), type="png")

            if frame_idx % 24 == 0:
                print(f"  > Progress: {frame_idx}/{total_frames} frames ({t_sec:.1f}s)...")

        browser.close()

    render_dur = time.time() - start_render
    print(f"[Render] All {total_frames} frames rendered in {render_dur:.1f}s!")

    # Copy frame at t=7.2s as the showcase title frame
    title_src = frames_dir / f"frame_{int(7.2 * fps):04d}.png"
    if title_src.exists():
        import shutil
        shutil.copyfile(title_src, title_frame_file)
        print(f"[Snapshot] Title showcase frame saved: {title_frame_file}")

    # Step 3: Encode video with FFmpeg at 24fps and mux with audio
    print(f"[Muxing] Encoding 1920x1080 24fps MP4 with ultra-sharp crf 16 and muxing audio...")
    cmd = [
        FFMPEG_EXE, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-i", str(audio_file),
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-t", "8.0",
        str(final_video_file)
    ]
    subprocess.run(cmd, check=True)
    print(f"\n[SUCCESS] Final 8-second cinematic intro delivered: {final_video_file}")

    # Clean up individual frame PNGs to save disk space
    print("[Cleanup] Cleaning up intermediate PNG frames...")
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    print("[Cleanup] Done.")

if __name__ == "__main__":
    main()
