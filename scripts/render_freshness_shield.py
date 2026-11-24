import os
import sys
import time
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

FFMPEG_EXE = r"C:\Users\PUBLIC.SURAJ-TOMAR\.unveo\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v9.0.2.exe"

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Freshness Gauge & Shield Convergence</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body, html {
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    background: #02040a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }

  .stage {
    position: relative;
    width: 1920px;
    height: 1080px;
    background: radial-gradient(circle at 50% 45%, #0f0a26 0%, #060919 50%, #02040a 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    perspective: 1400px;
  }

  /* Volumetric Violet & Mint Glows */
  .glow-violet-left {
    position: absolute;
    top: 10%;
    left: 15%;
    width: 700px;
    height: 700px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.12) 0%, transparent 70%);
    filter: blur(80px);
    pointer-events: none;
  }

  .glow-mint-right {
    position: absolute;
    bottom: 15%;
    right: 15%;
    width: 750px;
    height: 750px;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.1) 0%, transparent 70%);
    filter: blur(90px);
    pointer-events: none;
  }

  /* Camera Dolly-Out Rig */
  .camera-rig {
    position: relative;
    width: 1000px;
    height: 800px;
    display: flex;
    align-items: center;
    justify-content: center;
    transform-style: preserve-3d;
    animation: cameraDollyOut 10s cubic-bezier(0.2, 0.05, 0.2, 1) forwards;
  }

  @keyframes cameraDollyOut {
    0% {
      transform: scale(1.28) translateZ(80px) rotateX(4deg);
    }
    100% {
      transform: scale(0.92) translateZ(-40px) rotateX(1deg);
    }
  }

  /* Reflective Ground Plane */
  .reflection-plane {
    position: absolute;
    bottom: 60px;
    left: 50%;
    transform: translateX(-50%) rotateX(80deg);
    width: 1200px;
    height: 400px;
    background: linear-gradient(180deg, rgba(52, 211, 153, 0.08) 0%, transparent 60%);
    filter: blur(12px);
    pointer-events: none;
  }

  /* Central Assembly */
  .center-group {
    position: relative;
    width: 480px;
    height: 540px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* Gauge Ring SVG */
  .gauge-svg {
    position: absolute;
    width: 440px;
    height: 440px;
    transform: rotate(-90deg);
    transition: opacity 1.2s ease;
  }

  .gauge-track {
    fill: none;
    stroke: rgba(255, 255, 255, 0.05);
    stroke-width: 14;
  }

  .gauge-sweep {
    fill: none;
    stroke-width: 16;
    stroke-linecap: round;
    stroke-dasharray: 1005;
    stroke-dashoffset: 1005;
    animation: 
      sweepGauge 4.5s cubic-bezier(0.4, 0, 0.2, 1) forwards,
      pulseLockedGreen 2.5s ease-in-out 4.5s infinite alternate;
  }

  @keyframes sweepGauge {
    0% {
      stroke-dashoffset: 1005;
      stroke: #ef4444;
      filter: drop-shadow(0 0 16px rgba(239, 68, 68, 0.8));
    }
    30% {
      stroke: #ef4444;
      filter: drop-shadow(0 0 20px rgba(239, 68, 68, 0.9));
    }
    60% {
      stroke: #f59e0b;
      filter: drop-shadow(0 0 24px rgba(245, 158, 11, 0.9));
    }
    85% {
      stroke: #34d399;
      filter: drop-shadow(0 0 30px rgba(52, 211, 153, 1));
    }
    100% {
      stroke-dashoffset: 250; /* Locked in green 75-100 zone */
      stroke: #10b981;
      filter: drop-shadow(0 0 36px rgba(16, 185, 129, 1));
    }
  }

  @keyframes pulseLockedGreen {
    0% { filter: drop-shadow(0 0 24px rgba(52, 211, 153, 0.8)); }
    100% { filter: drop-shadow(0 0 48px rgba(110, 231, 183, 1)); }
  }

  /* Gauge Needle Indicator */
  .gauge-indicator {
    position: absolute;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #6ee7b7;
    box-shadow: 0 0 25px #34d399, 0 0 50px #10b981;
    opacity: 0;
    animation: indicatorAppear 1s ease 4.4s forwards;
  }

  @keyframes indicatorAppear {
    0% { opacity: 0; transform: scale(0.2); }
    100% { opacity: 1; transform: scale(1); }
  }

  /* Soft Light Pulse Wave on Lock */
  .green-lock-pulse {
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    border: 2px solid rgba(52, 211, 153, 0.8);
    background: radial-gradient(circle, rgba(52, 211, 153, 0.25) 0%, transparent 75%);
    opacity: 0;
    pointer-events: none;
    animation: lockPulseWave 1.8s cubic-bezier(0.1, 0.8, 0.2, 1) 4.5s forwards;
  }

  @keyframes lockPulseWave {
    0% {
      transform: scale(0.5);
      opacity: 0.9;
      filter: blur(2px);
    }
    60% {
      opacity: 0.5;
    }
    100% {
      transform: scale(2.6);
      opacity: 0;
      filter: blur(14px);
    }
  }

  /* Clean Glowing Shield */
  .shield-container {
    position: absolute;
    width: 320px;
    height: 380px;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0;
    animation: shieldMaterialize 2.4s cubic-bezier(0.16, 1, 0.3, 1) 5.2s forwards;
  }

  @keyframes shieldMaterialize {
    0% {
      opacity: 0;
      transform: scale(0.85);
      filter: blur(12px);
    }
    100% {
      opacity: 1;
      transform: scale(1);
      filter: blur(0px);
    }
  }

  .shield-svg {
    width: 260px;
    height: 310px;
    filter: drop-shadow(0 0 30px rgba(52, 211, 153, 0.7)) drop-shadow(0 0 60px rgba(168, 85, 247, 0.4));
  }

  .shield-path {
    fill: url(#shieldGrad);
    stroke: url(#shieldStroke);
    stroke-width: 4;
    stroke-linejoin: round;
  }

  .shield-inner-emblem {
    fill: none;
    stroke: #6ee7b7;
    stroke-width: 3;
    stroke-linecap: round;
    stroke-linejoin: round;
    filter: drop-shadow(0 0 12px #34d399);
    stroke-dasharray: 300;
    stroke-dashoffset: 300;
    animation: drawCheck 1.2s ease 6.2s forwards;
  }

  @keyframes drawCheck {
    to { stroke-dashoffset: 0; }
  }

  /* Particle Canvas */
  #particleCanvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 1920px;
    height: 1080px;
    pointer-events: none;
  }
</style>
</head>
<body>

<div class="stage">
  <div class="glow-violet-left"></div>
  <div class="glow-mint-right"></div>
  
  <div class="camera-rig">
    <div class="center-group">
      <!-- Freshness Circular Gauge -->
      <svg class="gauge-svg" viewBox="0 0 360 360">
        <circle class="gauge-track" cx="180" cy="180" r="160"></circle>
        <circle class="gauge-sweep" cx="180" cy="180" r="160"></circle>
      </svg>
      
      <!-- Lock Pulse -->
      <div class="green-lock-pulse"></div>

      <!-- Shield Container -->
      <div class="shield-container">
        <svg class="shield-svg" viewBox="0 0 200 240">
          <defs>
            <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="rgba(168, 85, 247, 0.28)"/>
              <stop offset="60%" stop-color="rgba(16, 185, 129, 0.22)"/>
              <stop offset="100%" stop-color="rgba(5, 150, 105, 0.35)"/>
            </linearGradient>
            <linearGradient id="shieldStroke" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#c084fc"/>
              <stop offset="50%" stop-color="#6ee7b7"/>
              <stop offset="100%" stop-color="#34d399"/>
            </linearGradient>
          </defs>
          <path class="shield-path" d="M 100 15 L 175 50 L 175 125 C 175 185 100 225 100 225 C 100 225 25 185 25 125 L 25 50 Z" />
          <path class="shield-inner-emblem" d="M 65 120 L 90 145 L 140 95" />
        </svg>
      </div>
    </div>

    <div class="reflection-plane"></div>
  </div>

  <canvas id="particleCanvas"></canvas>
</div>

<script>
  const canvas = document.getElementById("particleCanvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 1920;
  canvas.height = 1080;

  // Particle Convergence Flow
  const particles = [];
  const targetX = 1920 / 2;
  const targetY = 1080 / 2;
  const startTime = performance.now();

  for (let i = 0; i < 180; i++) {
    const fromLeft = Math.random() > 0.5;
    const startX = fromLeft ? -100 - Math.random() * 400 : 2020 + Math.random() * 400;
    const startY = 100 + Math.random() * 880;
    particles.push({
      startX,
      startY,
      x: startX,
      y: startY,
      targetRadius: Math.random() * 110 + 20,
      targetAngle: Math.random() * Math.PI * 2,
      size: Math.random() * 3.5 + 1.2,
      color: Math.random() > 0.4 ? "rgba(52, 211, 153, " : "rgba(192, 132, 252, ",
      delay: 4.2 + Math.random() * 2.5, // Enter between 4.2s and 6.7s
      speed: Math.random() * 0.025 + 0.015,
      progress: 0,
      alpha: Math.random() * 0.6 + 0.4
    });
  }

  function render(now) {
    const elapsed = (now - startTime) / 1000;
    ctx.clearRect(0, 0, 1920, 1080);

    for (const p of particles) {
      if (elapsed > p.delay) {
        p.progress = Math.min(1, p.progress + p.speed);
        // EaseInOutCubic
        const t = p.progress;
        const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

        const destX = targetX + Math.cos(p.targetAngle) * p.targetRadius * (1 - ease * 0.5);
        const destY = targetY + Math.sin(p.targetAngle) * p.targetRadius * (1 - ease * 0.5);

        p.x = p.startX + (destX - p.startX) * ease;
        p.y = p.startY + (destY - p.startY) * ease;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * (1 - ease * 0.3), 0, Math.PI * 2);
        ctx.fillStyle = p.color + (p.alpha * Math.min(1, p.progress * 2)) + ")";
        ctx.shadowBlur = 14;
        ctx.shadowColor = p.color + "0.8)";
        ctx.fill();
      }
    }

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
</script>

</body>
</html>
"""

def generate_warm_synth_and_chime(output_audio_path: Path, duration_s: float = 10.0):
    """
    Synthesizes the audio track matching:
    - Confident, warm synth chord: Fmaj9 / Cmaj7 pad (F3=174.6Hz, A3=220Hz, C4=261.6Hz, E4=329.6Hz)
    - Soft chime resolving at t=4.5s: Pristine 1760Hz bell strike with decaying harmonics.
    """
    print("[Audio] Synthesizing confident, warm synth chord with resolving chime...")
    
    # Layer 1: Warm synth chord (F3, A3, C4, E4) with soft attack and decay
    chord_filter = (
        "sine=f=174.61:d=8.5[f3]; "
        "sine=f=220.00:d=8.5[a3]; "
        "sine=f=261.63:d=8.5[c4]; "
        "sine=f=329.63:d=8.5[e4]; "
        "[f3][a3][c4][e4]amix=inputs=4:weights=0.35 0.3 0.25 0.2[raw_chord]; "
        "[raw_chord]volume=0.45,afade=t=in:st=0:d=1.8,afade=t=out:st=6.5:d=2.0,lowpass=f=450[warm_pad]; "
        # Layer 2: Pristine chime resolving at t=4.5s
        "sine=f=1760:d=4.0,afade=t=in:st=0:d=0.01,afade=t=out:st=0.08:d=3.8,volume=0.35,adelay=4500|4500[chime1]; "
        "sine=f=2637:d=3.5,afade=t=in:st=0:d=0.01,afade=t=out:st=0.05:d=3.2,volume=0.22,adelay=4500|4500[chime2]; "
        "[chime1][chime2]amix=inputs=2[chime]; "
        # Layer 3: Subtle sub warmth
        "sine=f=87.3:d=8.5,volume=0.2,afade=t=in:st=0:d=2.0,afade=t=out:st=6.5:d=2.0[sub]; "
        # Master Mix
        "[warm_pad][chime][sub]amix=inputs=3:weights=1.0 1.2 0.7[aout]"
    )

    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex", chord_filter,
        "-map", "[aout]",
        "-t", str(duration_s),
        "-c:a", "aac", "-b:a", "192k",
        str(output_audio_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"[Audio] Audio generated: {output_audio_path}")


def main():
    out_dir = Path("unveo-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_video_dir = out_dir / "raw_shield"
    raw_video_dir.mkdir(parents=True, exist_ok=True)

    html_file = out_dir / "freshness_shield.html"
    html_file.write_text(HTML_CONTENT, encoding="utf-8")

    audio_file = out_dir / "freshness_shield_audio.aac"
    final_video_file = out_dir / "freshness-gauge-shield.mp4"

    # 1. Synthesize audio
    generate_warm_synth_and_chime(audio_file, duration_s=10.0)

    # 2. Record animation with Playwright at 1920x1080
    print("[Video] Capturing 10s freshness gauge & shield animation at 1920x1080...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(raw_video_dir),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.goto(html_file.resolve().as_uri(), wait_until="networkidle")

        time.sleep(10.2)

        video = page.video
        context.close()
        browser.close()

        raw_video_path = Path(video.path())
        print(f"[Video] Raw video recorded to: {raw_video_path}")

    # 3. Combine video + audio
    print(f"[Muxing] Re-encoding into {final_video_file}...")
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-i", str(raw_video_path),
        "-i", str(audio_file),
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-t", "10.0",
        str(final_video_file)
    ], check=True)

    print(f"[Done] Freshness Gauge & Shield Video generated at: {final_video_file}")
    return str(final_video_file)

if __name__ == "__main__":
    main()
