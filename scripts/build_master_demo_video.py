import os
import sys
import subprocess
from pathlib import Path

FFMPEG_EXE = r"C:\Users\PUBLIC.SURAJ-TOMAR\.unveo\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v9.0.2.exe"

def main():
    out_dir = Path("unveo-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / "edit_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    voiceover_path = Path(r"c:\Users\PUBLIC.SURAJ-TOMAR\Downloads\prb realme 5\Generated Audio November 21, 2026 - 10_15AM.wav")
    demo_video_path = out_dir / "demo-video-80s.mp4"
    hologram_video_path = out_dir / "cinematic-hologram-expiry.mp4"
    shield_video_path = out_dir / "freshness-gauge-shield.mp4"

    master_output_path = out_dir / "veric-product-demo-master.mp4"

    print("[1/5] Checking source video and audio assets...")
    for p in [voiceover_path, demo_video_path, hologram_video_path, shield_video_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required asset: {p}")
        print(f"  ✓ Found: {p.name}")

    # Step 2: Extract & trim the exact video clips according to the master EDL
    print("[2/5] Trimming high-resolution video segments to match voiceover narrative...")

    # Clip 1: Cinematic Intro (10.5s)
    # Covers: "AI answers sound confident... Prices change, APIs change, laws change, and chatbots don't always notice."
    clip1 = temp_dir / "clip1_intro.mp4"
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-ss", "00:00:00.800", "-i", str(hologram_video_path),
        "-t", "10.5",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-an",
        str(clip1)
    ], check=True)
    print("  ✓ Clip 1 (Cinematic Hologram Expiry Hook): 10.5s")

    # Clip 2: Paste answer & Check Freshness (11.0s)
    # Covers: "This is Veric. Paste any AI answer, and Veric finds the claims that can go stale, checks each one against the live web..."
    clip2 = temp_dir / "clip2_input_audit.mp4"
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-ss", "00:00:04.200", "-i", str(demo_video_path),
        "-t", "11.0",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-an",
        str(clip2)
    ], check=True)
    print("  ✓ Clip 2 (Input Studio & Live Freshness Stepper): 11.0s")

    # Clip 3: Annotated text & Stale claim highlights (9.0s)
    # Covers: "Here, the GPT-4 claim is outdated, the GST slabs changed in September 2025, and the iPhone claim is stale too."
    clip3 = temp_dir / "clip3_annotated.mp4"
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-ss", "00:00:16.500", "-i", str(demo_video_path),
        "-t", "9.0",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-an",
        str(clip3)
    ], check=True)
    print("  ✓ Clip 3 (Annotated Answer & Stale Highlights): 9.0s")

    # Clip 4: Claim cards, primary sources & diff view (8.5s)
    # Covers: "Every verdict shows what is actually true today, with official sources marked primary, and how many independent sources agree."
    clip4 = temp_dir / "clip4_sources_diff.mp4"
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-ss", "00:00:28.000", "-i", str(demo_video_path),
        "-t", "8.5",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-an",
        str(clip4)
    ], check=True)
    print("  ✓ Clip 4 (Claimed vs Actually, Primary Sources, & Diff): 8.5s")

    # Clip 5: Bifrost Gateway outage simulation & Fallback (9.0s)
    # Covers: "All model calls run through the Bifrost gateway. When one model fails, Bifrost switches to a backup automatically..."
    clip5 = temp_dir / "clip5_gateway_fallback.mp4"
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-ss", "00:00:41.000", "-i", str(demo_video_path),
        "-t", "9.0",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-an",
        str(clip5)
    ], check=True)
    print("  ✓ Clip 5 (Bifrost Gateway & Outage Circuit Breaker): 9.0s")

    # Clip 6: Ground Truth Benchmark metrics (7.5s)
    # Covers: "A built-in benchmark measures accuracy, precision, and recall. And when the evidence is thin, Veric says unverifiable..."
    clip6 = temp_dir / "clip6_benchmark.mp4"
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-ss", "00:00:56.000", "-i", str(demo_video_path),
        "-t", "7.5",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-an",
        str(clip6)
    ], check=True)
    print("  ✓ Clip 6 (Agent Lab & Benchmark Accuracy/Recall): 7.5s")

    # Clip 7: Cinematic Outro: Gauge locking green & Shield formation (5.5s)
    # Covers: "Veric. Trust the answer only while it's still true."
    clip7 = temp_dir / "clip7_shield_outro.mp4"
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-ss", "00:00:04.200", "-i", str(shield_video_path),
        "-t", "5.5",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-an",
        str(clip7)
    ], check=True)
    print("  ✓ Clip 7 (Cinematic Gauge Lock & Resolving Shield): 5.5s")

    # Step 3: Concatenate video segments
    print("[3/5] Seamlessly stitching 7 story clips into continuous 1080p visual track...")
    concat_list_file = temp_dir / "concat_list.txt"
    clips = [clip1, clip2, clip3, clip4, clip5, clip6, clip7]
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{c.resolve().as_posix()}'\n")

    stitched_video = temp_dir / "stitched_visuals.mp4"
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        str(stitched_video)
    ], check=True)
    print("  ✓ Stitched video track ready.")

    # Step 4: Master Audio Mixing
    # Mix user's voiceover with subtle cinematic ambient bed & ending chime swell
    print("[4/5] Mastering audio track: leveling voiceover & blending cinematic score...")
    mastered_audio = temp_dir / "mastered_audio.aac"
    
    # Audio filtergraph:
    # 1. Voiceover: normalized and brought to broadcast standard (-14 LUFS / clean speech)
    # 2. Add subtle ambient bed during intro (0-10s) and warm chord/chime during outro (55-61s)
    audio_filter = (
        # Voiceover input
        "[0:a]aresample=48000,volume=1.35,alimiter=limit=0.95[voice]; "
        # Ambient bed for opening
        "anoisesrc=d=11:c=pink:r=48000:a=0.015,lowpass=f=180,afade=t=out:st=9.5:d=1.5,volume=0.18[intro_bed]; "
        # Resolving chime for outro
        "sine=f=1760:d=4.5,afade=t=in:st=0:d=0.01,afade=t=out:st=0.08:d=4.2,volume=0.22,adelay=55500|55500[outro_chime]; "
        # Mix tracks
        "[voice][intro_bed]amix=inputs=2:weights=1.0 0.35:dropout_transition=2[mix1]; "
        "[mix1][outro_chime]amix=inputs=2:weights=1.0 0.6[a_out]"
    )

    subprocess.run([
        FFMPEG_EXE, "-y",
        "-i", str(voiceover_path),
        "-filter_complex", audio_filter,
        "-map", "[a_out]",
        "-c:a", "aac", "-b:a", "256k",
        str(mastered_audio)
    ], check=True)
    print("  ✓ Mastered audio track ready.")

    # Step 5: Final Muxing with high-efficiency h264 profile
    print("[5/5] Exporting final mastered demo video...")
    subprocess.run([
        FFMPEG_EXE, "-y",
        "-i", str(stitched_video),
        "-i", str(mastered_audio),
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        str(master_output_path)
    ], check=True)

    print(f"\n=======================================================")
    print(f"SUCCESS! Master demo video generated:")
    print(f"Path: {master_output_path.resolve()}")
    print(f"=======================================================")
    return str(master_output_path)

if __name__ == "__main__":
    main()
