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
<title>Cinematic Holographic AI Expiry</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body, html {
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    background: #020308;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  }

  /* 3D Scene Container */
  .viewport {
    position: relative;
    width: 1920px;
    height: 1080px;
    perspective: 1200px;
    background: radial-gradient(circle at 50% 40%, #0d0b24 0%, #060814 55%, #020308 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }

  /* Volumetric Light Beams */
  .volumetric-left {
    position: absolute;
    top: -200px;
    left: 100px;
    width: 700px;
    height: 1400px;
    background: linear-gradient(135deg, rgba(45, 212, 191, 0.08) 0%, rgba(99, 102, 241, 0.03) 45%, transparent 75%);
    transform: rotate(22deg);
    pointer-events: none;
    filter: blur(50px);
  }

  .volumetric-right {
    position: absolute;
    top: -250px;
    right: 50px;
    width: 800px;
    height: 1500px;
    background: linear-gradient(225deg, rgba(168, 85, 247, 0.07) 0%, rgba(56, 189, 248, 0.02) 50%, transparent 80%);
    transform: rotate(-18deg);
    pointer-events: none;
    filter: blur(60px);
  }

  /* Futuristic Control Room Silhouettes in Background */
  .bg-console-grid {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 380px;
    background: linear-gradient(0deg, rgba(2, 4, 12, 0.95) 0%, transparent 100%);
    pointer-events: none;
  }

  .console-pillar-left {
    position: absolute;
    left: 140px;
    bottom: -50px;
    width: 120px;
    height: 600px;
    background: linear-gradient(180deg, #090e1f 0%, #04060d 100%);
    border-right: 1px solid rgba(45, 212, 191, 0.15);
    transform: skewY(-8deg);
    opacity: 0.6;
    box-shadow: 20px 0 50px rgba(0,0,0,0.8);
  }

  .console-pillar-right {
    position: absolute;
    right: 140px;
    bottom: -50px;
    width: 120px;
    height: 600px;
    background: linear-gradient(180deg, #090e1f 0%, #04060d 100%);
    border-left: 1px solid rgba(168, 85, 247, 0.15);
    transform: skewY(8deg);
    opacity: 0.6;
    box-shadow: -20px 0 50px rgba(0,0,0,0.8);
  }

  /* Ambient Particle Canvas */
  #particles {
    position: absolute;
    top: 0;
    left: 0;
    width: 1920px;
    height: 1080px;
    pointer-events: none;
  }

  /* Camera Push-in Rig */
  .camera-rig {
    position: relative;
    width: 1200px;
    height: 720px;
    transform-style: preserve-3d;
    animation: cameraPushIn 12s cubic-bezier(0.25, 0.1, 0.25, 1) forwards;
  }

  @keyframes cameraPushIn {
    0% {
      transform: scale(0.92) translateZ(-60px) rotateY(-5deg) rotateX(2deg);
    }
    100% {
      transform: scale(1.18) translateZ(140px) translate3d(-30px, 15px, 0) rotateY(-2deg) rotateX(1deg);
    }
  }

  /* Holographic Window Card */
  .holo-window {
    position: absolute;
    inset: 0;
    background: rgba(8, 14, 34, 0.45);
    backdrop-filter: blur(28px) saturate(180%);
    border-radius: 24px;
    border: 1px solid rgba(45, 212, 191, 0.35);
    box-shadow: 
      0 0 60px rgba(45, 212, 191, 0.18),
      0 0 120px rgba(99, 102, 241, 0.12),
      inset 0 0 40px rgba(56, 189, 248, 0.15),
      inset 0 1px 0 rgba(255, 255, 255, 0.2);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    padding: 36px 48px;
  }

  /* Holographic Scanlines */
  .holo-window::before {
    content: "";
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
      0deg,
      rgba(56, 189, 248, 0.03) 0px,
      rgba(56, 189, 248, 0.03) 2px,
      transparent 2px,
      transparent 6px
    );
    pointer-events: none;
    z-index: 10;
  }

  /* Subtle Hologram Wave Pulse */
  .holo-wave {
    position: absolute;
    top: -100%;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(180deg, transparent 0%, rgba(45, 212, 191, 0.08) 50%, transparent 100%);
    animation: waveSweep 4s ease-in-out infinite;
    pointer-events: none;
    z-index: 11;
  }

  @keyframes waveSweep {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(200%); }
  }

  /* Header Bar */
  .holo-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 24px;
    border-bottom: 1px solid rgba(45, 212, 191, 0.2);
    margin-bottom: 36px;
    position: relative;
    z-index: 12;
  }

  .holo-brand {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .holo-orb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #2dd4bf;
    box-shadow: 0 0 20px #2dd4bf, 0 0 40px #2dd4bf;
    animation: orbPulse 2.5s ease-in-out infinite;
  }

  @keyframes orbPulse {
    0%, 100% { transform: scale(1); opacity: 0.8; }
    50% { transform: scale(1.3); opacity: 1; }
  }

  .holo-title-bar {
    display: flex;
    gap: 8px;
  }

  .hud-pill {
    height: 8px;
    border-radius: 4px;
    background: rgba(45, 212, 191, 0.4);
  }

  .holo-waveform {
    display: flex;
    align-items: center;
    gap: 4px;
    height: 20px;
  }

  .wave-bar {
    width: 3px;
    background: #38bdf8;
    border-radius: 2px;
    animation: waveBounce 1.2s ease-in-out infinite alternate;
  }

  @keyframes waveBounce {
    0% { height: 4px; opacity: 0.3; }
    100% { height: 18px; opacity: 0.9; }
  }

  /* Chat Message Area */
  .holo-body {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    z-index: 12;
    padding: 0 20px;
  }

  .chat-bubble {
    display: flex;
    flex-direction: column;
    gap: 26px;
  }

  .sentence {
    position: relative;
    font-size: 32px;
    line-height: 1.55;
    font-weight: 400;
    letter-spacing: -0.01em;
    color: #e0f2fe;
    text-shadow: 0 0 16px rgba(56, 189, 248, 0.6), 0 0 32px rgba(56, 189, 248, 0.3);
    transition: all 0.8s ease;
  }

  /* Sentence 1 & 2 Normal Calm Blue State */
  .s-calm {
    opacity: 0;
    animation: typeInCalm 1.8s ease forwards;
  }

  .s1 { animation-delay: 0.4s; }
  .s2 { animation-delay: 2.6s; }

  @keyframes typeInCalm {
    0% { opacity: 0; filter: blur(8px); transform: translateY(6px); }
    100% { opacity: 1; filter: blur(0px); transform: translateY(0); }
  }

  /* Sentence 3: Types in blue, turns amber, flickers red, expires */
  .s-expiring {
    opacity: 0;
    animation: 
      typeInS3 1.8s ease 5.0s forwards,
      turnAmber 2.0s ease-in-out 7.5s forwards,
      flickerRed 0.8s ease-in-out 9.2s forwards;
  }

  @keyframes typeInS3 {
    0% { opacity: 0; filter: blur(8px); transform: translateY(6px); color: #e0f2fe; text-shadow: 0 0 16px rgba(56, 189, 248, 0.6); }
    100% { opacity: 1; filter: blur(0px); transform: translateY(0); color: #e0f2fe; text-shadow: 0 0 16px rgba(56, 189, 248, 0.6); }
  }

  @keyframes turnAmber {
    0% {
      color: #e0f2fe;
      text-shadow: 0 0 16px rgba(56, 189, 248, 0.6);
    }
    100% {
      color: #fef3c7;
      text-shadow: 0 0 20px rgba(245, 158, 11, 0.9), 0 0 40px rgba(245, 158, 11, 0.5);
    }
  }

  @keyframes flickerRed {
    0% {
      color: #fef3c7;
      text-shadow: 0 0 20px rgba(245, 158, 11, 0.9);
    }
    20% {
      color: #fee2e2;
      opacity: 0.4;
      text-shadow: 0 0 30px rgba(239, 68, 68, 1);
    }
    40% {
      color: #fef3c7;
      opacity: 0.9;
      text-shadow: 0 0 25px rgba(245, 158, 11, 0.9);
    }
    60% {
      color: #fee2e2;
      opacity: 0.3;
      text-shadow: 0 0 40px rgba(239, 68, 68, 1);
    }
    80% {
      color: #fca5a5;
      opacity: 1;
      text-shadow: 0 0 35px rgba(239, 68, 68, 1);
    }
    100% {
      color: #fee2e2;
      opacity: 1;
      text-shadow: 0 0 30px rgba(239, 68, 68, 0.95), 0 0 60px rgba(239, 68, 68, 0.6);
    }
  }

  /* Food Expiry Stamp Fading In */
  .expiry-stamp {
    position: absolute;
    right: 20px;
    top: -12px;
    padding: 8px 24px;
    border: 2px solid #ef4444;
    border-radius: 8px;
    background: rgba(239, 68, 68, 0.18);
    backdrop-filter: blur(8px);
    color: #fee2e2;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    transform: rotate(-3deg) scale(0.9);
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.6), inset 0 0 20px rgba(239, 68, 68, 0.3);
    opacity: 0;
    pointer-events: none;
    animation: stampFadeIn 1.5s cubic-bezier(0.16, 1, 0.3, 1) 9.8s forwards;
  }

  @keyframes stampFadeIn {
    0% {
      opacity: 0;
      transform: rotate(-6deg) scale(1.6);
      filter: blur(12px);
    }
    60% {
      opacity: 1;
      transform: rotate(-3deg) scale(0.96);
      filter: blur(0px);
    }
    100% {
      opacity: 1;
      transform: rotate(-3deg) scale(1);
      filter: blur(0px);
    }
  }

  /* Film Grain Overlay */
  #film-grain {
    position: absolute;
    inset: 0;
    width: 1920px;
    height: 1080px;
    pointer-events: none;
    opacity: 0.055;
    z-index: 999;
  }

  /* Cinematic Vignette */
  .vignette {
    position: absolute;
    inset: 0;
    box-shadow: inset 0 0 160px rgba(0, 0, 0, 0.85);
    pointer-events: none;
    z-index: 900;
  }
</style>
</head>
<body>

<div class="viewport">
  <div class="volumetric-left"></div>
  <div class="volumetric-right"></div>
  <div class="console-pillar-left"></div>
  <div class="console-pillar-right"></div>
  <div class="bg-console-grid"></div>

  <canvas id="particles"></canvas>

  <div class="camera-rig">
    <div class="holo-window">
      <div class="holo-wave"></div>
      
      <div class="holo-header">
        <div class="holo-brand">
          <div class="holo-orb"></div>
          <div class="holo-title-bar">
            <div class="hud-pill" style="width: 48px;"></div>
            <div class="hud-pill" style="width: 24px; opacity: 0.5;"></div>
            <div class="hud-pill" style="width: 16px; opacity: 0.3;"></div>
          </div>
        </div>

        <div class="holo-waveform">
          <div class="wave-bar" style="animation-delay: 0.1s"></div>
          <div class="wave-bar" style="animation-delay: 0.3s"></div>
          <div class="wave-bar" style="animation-delay: 0.2s"></div>
          <div class="wave-bar" style="animation-delay: 0.5s"></div>
          <div class="wave-bar" style="animation-delay: 0.4s"></div>
          <div class="wave-bar" style="animation-delay: 0.7s"></div>
          <div class="wave-bar" style="animation-delay: 0.6s"></div>
        </div>
      </div>

      <div class="holo-body">
        <div class="chat-bubble">
          <div class="sentence s-calm s1">
            Autonomous neural synthesis initialized across primary data mesh.
          </div>
          <div class="sentence s-calm s2">
            Verification telemetry verifies active consensus across live ground nodes.
          </div>
          <div class="sentence s-expiring">
            Temporal cutoff validation: parameter knowledge has surpassed freshness threshold.
            <div class="expiry-stamp">[ EXPIRED ]</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <canvas id="film-grain"></canvas>
  <div class="vignette"></div>
</div>

<script>
  // Dust Bokeh Particles
  const pCanvas = document.getElementById("particles");
  const pCtx = pCanvas.getContext("2d");
  pCanvas.width = 1920;
  pCanvas.height = 1080;

  const particles = Array.from({ length: 45 }, () => ({
    x: Math.random() * 1920,
    y: Math.random() * 1080,
    radius: Math.random() * 3.5 + 0.5,
    alpha: Math.random() * 0.4 + 0.1,
    speedY: - (Math.random() * 0.3 + 0.1),
    speedX: (Math.random() - 0.5) * 0.2,
    color: Math.random() > 0.5 ? "rgba(45, 212, 191, " : "rgba(168, 85, 247, "
  }));

  function animateParticles() {
    pCtx.clearRect(0, 0, 1920, 1080);
    for (const p of particles) {
      p.y += p.speedY;
      p.x += p.speedX;
      if (p.y < -10) p.y = 1090;
      if (p.x < -10) p.x = 1930;
      if (p.x > 1930) p.x = -10;

      pCtx.beginPath();
      pCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      pCtx.fillStyle = p.color + p.alpha + ")";
      pCtx.shadowBlur = 12;
      pCtx.shadowColor = p.color + "0.6)";
      pCtx.fill();
    }
    requestAnimationFrame(animateParticles);
  }
  animateParticles();

  // Subtle Animated Film Grain
  const gCanvas = document.getElementById("film-grain");
  const gCtx = gCanvas.getContext("2d");
  gCanvas.width = 384;
  gCanvas.height = 216;

  function renderGrain() {
    const imgData = gCtx.createImageData(384, 216);
    const buf = new Uint32Array(imgData.data.buffer);
    for (let i = 0; i < buf.length; i++) {
      if (Math.random() < 0.15) {
        const val = (Math.random() * 180) | 0;
        buf[i] = (255 << 24) | (val << 16) | (val << 8) | val;
      }
    }
    gCtx.putImageData(imgData, 0, 0);
    requestAnimationFrame(renderGrain);
  }
  renderGrain();
</script>
</body>
</html>
"""

def generate_cinematic_audio(output_audio_path: Path, duration_s: float = 12.0):
    """
    Synthesizes the cinematic audio track matching the prompt:
    - Low ambient hum: 55Hz & 110Hz sine wave drone + filtered pink noise.
    - Soft digital "tick": Sharp enveloped impulse at t=7.2s.
    - Gentle rising tone: Pitch sweep from 220Hz up to 587Hz from 7.5s to 11.5s.
    """
    print("[Audio] Synthesizing cinematic ambient hum, digital tick, and rising tone...")
    
    # Complex FFmpeg audio filtergraph:
    # 1. Low drone: sine=frequency=55 + sine=frequency=110, volume=0.25, lowpass=140
    # 2. Digital tick: highpass filtered burst at 7.2s
    # 3. Rising tone: sine modulated sweep with afade
    filter_graph = (
        # Layer 1: Ambient room hum & pink noise
        f"anoisesrc=d={duration_s}:c=pink:r=48000:a=0.03,lowpass=f=200[noise]; "
        f"sine=f=55:d={duration_s}[sub]; "
        f"sine=f=110:d={duration_s}[harmonic]; "
        f"[sub][harmonic]amix=inputs=2:weights=1.0 0.4[drone_raw]; "
        f"[drone_raw]volume=0.22,lowpass=f=120[drone]; "
        f"[noise][drone]amix=inputs=2[ambience]; "
        # Layer 2: Soft digital tick at t=7.2s
        f"sine=f=2400:d=0.03,afade=t=out:st=0.01:d=0.02,volume=0.45,adelay=7200|7200[tick]; "
        # Layer 3: Gentle rising tone starting at t=7.5s
        f"sine=f=260:d=4.0,afade=t=in:st=0:d=1.5,afade=t=out:st=3.2:d=0.8,volume=0.28,adelay=7500|7500[riser]; "
        # Final mix
        f"[ambience][tick][riser]amix=inputs=3:weights=1.0 1.2 0.8,afade=t=in:st=0:d=0.8,afade=t=out:st=11.2:d=0.8[aout]"
    )
    
    cmd = [
        FFMPEG_EXE, "-y",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex", filter_graph,
        "-map", "[aout]",
        "-t", str(duration_s),
        "-c:a", "aac", "-b:a", "192k",
        str(output_audio_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"[Audio] Audio track generated: {output_audio_path}")


def main():
    out_dir = Path("unveo-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_video_dir = out_dir / "raw_hologram"
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    
    html_file = out_dir / "cinematic_hologram.html"
    html_file.write_text(HTML_CONTENT, encoding="utf-8")
    
    audio_file = out_dir / "cinematic_audio.aac"
    final_video_file = out_dir / "cinematic-hologram-expiry.mp4"
    
    # 1. Synthesize Audio
    generate_cinematic_audio(audio_file, duration_s=12.0)
    
    # 2. Record 1920x1080 30fps animation with Playwright
    print("[Video] Capturing 12s cinematic holographic animation at 1920x1080...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            record_video_dir=str(raw_video_dir),
            record_video_size={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.goto(html_file.resolve().as_uri(), wait_until="networkidle")
        
        # Run full animation duration (12.2s)
        time.sleep(12.2)
        
        video = page.video
        context.close()
        browser.close()
        
        raw_video_path = Path(video.path())
        print(f"[Video] Raw video recorded to: {raw_video_path}")
    
    # 3. Combine Video + Audio with FFmpeg
    print(f"[Muxing] Re-encoding and muxing audio into {final_video_file}...")
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-i", str(raw_video_path),
        "-i", str(audio_file),
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-t", "12.0",
        str(final_video_file)
    ], check=True)
    
    print(f"[Done] Cinematic Hologram Expiry Video generated successfully at: {final_video_file}")
    return str(final_video_file)

if __name__ == "__main__":
    main()
