"""
Kritiq MAE — Visual Analysis Module

Extracts frames from performance video and sends to Claude Vision
for visual assessment.

Phase 5: two analysis modes.
  - mode="presence"  (vocals selected): stage presence / visual performance
    assessment — the original behavior.
  - mode="technique" (no vocals selected): instrument-technique-focused
    analysis only. Hand position, finger placement, instrument hold, bow
    technique, equipment setup and care. NO eye contact scoring, NO stage
    presence, NO performance energy.

Both modes return the same JSON shape so the frontend and DB writes stay
uniform; `analysis_type` distinguishes them and the `visual_kind` column on
submissions records which ran. The score key stays `presence_score` for
back-compat with the frontend and the 193 archived rows — for technique
mode, the value in that key IS the technique visual score.
"""

import os
import subprocess
import tempfile
import base64
import json
import logging
import requests as http_requests

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# Sensitivity rules — same as feedback.py
SENSITIVITY_RULES = """
VISUAL FEEDBACK BOUNDARIES (strictly enforced):
IN-BOUNDS (you may comment on): grooming, attire, framing, lighting, gear care, 
stage presence, posture related to playing technique, eye contact with camera/audience,
performance energy, movement that affects sound quality, camera angle, background setup.

OUT-OF-BOUNDS (never comment on): body shape, weight, physical features, skin,
facial features, age appearance, physical disability, gender expression, 
ethnicity-related features, anything about the person's physical body that is 
not directly related to their musical performance technique.
"""

# Technique mode narrows the in-bounds list further: no presence/energy/eye
# contact commentary at all.
TECHNIQUE_SENSITIVITY_RULES = """
VISUAL FEEDBACK BOUNDARIES (strictly enforced):
IN-BOUNDS (you may comment on): hand position, finger placement, instrument
hold/positioning, bow technique (for bowed instruments), fretting/picking
technique, pedal or hardware technique, equipment setup and care, framing and
lighting ONLY as they affect visibility of technique.

OUT-OF-BOUNDS (never comment on): eye contact, stage presence, performance
energy, engagement, charisma, emotional expression, body shape, weight,
physical features, skin, facial features, age appearance, physical disability,
gender expression, ethnicity-related features, anything about the person's
physical body that is not directly related to instrument technique.
"""


def analyze_video(video_path: str, num_frames: int = 3, instrument: str = "",
                  mode: str = "presence") -> dict:
    """
    Extract frames from video, send to Claude Vision for visual analysis.

    Args:
        video_path: path to the uploaded video file
        num_frames: how many evenly-spaced frames to extract (default 3)
        instrument: what the user is playing (e.g. "Guitar", "Vocals")
        mode: "presence" (vocals selected) or "technique" (instrument-only)

    Returns:
        {
            "presence_score": float (1-10),   # technique score when mode="technique"
            "visual_feedback": [{"point": str, "detail": str}],
            "summary": str,
            "analysis_type": "presence" | "technique",
        }
        or None if analysis fails
    """
    if not ANTHROPIC_API_KEY:
        logger.info("No ANTHROPIC_API_KEY — skipping visual analysis")
        return None

    if mode not in ("presence", "technique"):
        mode = "presence"

    try:
        # Step 1: Extract frames
        frames = _extract_frames(video_path, num_frames)
        if not frames:
            logger.warning("No frames extracted from video")
            return None

        # Step 2: Send to Claude Vision
        result = _analyze_frames(frames, instrument=instrument, mode=mode)

        # Step 3: Cleanup
        for f in frames:
            try:
                os.unlink(f)
            except:
                pass

        return result

    except Exception as e:
        logger.exception("Visual analysis failed")
        return None


def _extract_frames(video_path: str, num_frames: int = 3) -> list:
    """Extract evenly-spaced frames from video as JPEG files."""
    # Get video duration first
    probe = subprocess.run(
        ["ffmpeg", "-i", video_path],
        capture_output=True, text=True, timeout=10,
    )
    # Parse duration from ffmpeg output (stderr)
    duration = None
    for line in probe.stderr.split("\n"):
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            duration = float(h) * 3600 + float(m) * 60 + float(s)
            break

    if not duration or duration < 1:
        logger.warning(f"Could not determine video duration")
        return []

    # Calculate timestamps for evenly-spaced frames
    # Skip first and last 10% to avoid black frames
    start = duration * 0.1
    end = duration * 0.9
    interval = (end - start) / (num_frames + 1)
    timestamps = [start + interval * (i + 1) for i in range(num_frames)]

    frames = []
    for i, ts in enumerate(timestamps):
        output_path = tempfile.mktemp(suffix=f"_frame_{i}.jpg")
        result = subprocess.run(
            [
                "ffmpeg", "-ss", str(round(ts, 2)),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",  # high quality JPEG
                "-y", output_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            # Verify file has content
            if os.path.getsize(output_path) > 1000:
                frames.append(output_path)
            else:
                os.unlink(output_path)

    logger.info(f"Extracted {len(frames)} frames from {duration:.1f}s video")
    return frames


def _presence_prompt(instrument_note: str) -> str:
    """Original stage-presence prompt (vocals-present submissions)."""
    return f"Analyze these frames from a musician's cover performance video.{instrument_note}" + """

Score their visual presence on a scale of 1-10 where:
- 9.0-10: Multiple camera angles, professional/intentional lighting, studio or venue environment, quality microphone visible. Score 9.0 if ANY TWO of these are present.
- 8.0-8.9: Dedicated home setup with visible mic, intentional lighting, clean framing.
- 6.0-7.9: Basic but functional — phone recording with some thought to framing.
- 4.0-5.9: Minimal effort — poor lighting, awkward angle, cluttered background.
- Below 4.0: Barely visible, extremely poor setup.

CRITICAL: If you see evidence of multiple camera angles across the frames (different shots/perspectives), the score MUST be 9.0 or higher. Multiple cameras = professional production, period. Do not score multi-camera setups below 9.0.

Respond ONLY with valid JSON:
{
    "presence_score": <float 1-10>,
    "visual_feedback": [
        {"point": "Brief headline", "detail": "Specific observation and actionable advice"},
    ],
    "summary": "1-2 sentence visual assessment"
}

Include 3-5 items in visual_feedback covering: framing/camera angle, lighting, 
posture/technique, energy/engagement, and gear/environment.
Do NOT include any text outside the JSON."""


def _technique_prompt(instrument: str, instrument_note: str) -> str:
    """Phase 5 technique-only prompt (instrument-only submissions)."""
    inst = instrument or "their instrument"
    return f"Analyze these frames from a musician's instrumental performance video.{instrument_note}" + f"""

This is a TECHNIQUE-FOCUSED analysis. The performer selected {inst} with no
vocals. Evaluate ONLY instrument technique visible on camera. Do NOT evaluate
stage presence, eye contact, performance energy, engagement, or charisma —
those are out of scope for this analysis.

Evaluate technique markers appropriate to {inst}, such as (where visible):
- Hand position and finger placement
- Instrument hold / positioning / strap or stand setup
- Bow technique (bowed strings), picking/fretting (guitar family),
  hand posture at the keys (piano/keyboard), stick grip (percussion),
  embouchure visibility and horn position (brass/woodwind)
- Equipment setup and care (tuning pegs, cable routing, stand stability,
  instrument condition)
- Framing and lighting ONLY insofar as they help or hurt the viewer's
  ability to see the technique

Score their visible technique on a scale of 1-10 where:
- 9.0-10: Textbook positioning and control clearly visible; setup is
  professional and deliberate.
- 8.0-8.9: Strong fundamentals visible with minor inefficiencies.
- 6.0-7.9: Functional technique; some visible habits worth correcting.
- 4.0-5.9: Clear technical issues visible (grip, posture at the instrument,
  positioning) that likely affect the sound.
- Below 4.0: Technique not really assessable or fundamentally problematic.

If the frames don't show enough of the instrument to assess technique
(hands out of frame, instrument cropped), say so in the summary, score
conservatively toward the middle, and make the FIRST feedback item a
framing recommendation so future recordings capture the technique.

Respond ONLY with valid JSON:
{{
    "presence_score": <float 1-10>,
    "visual_feedback": [
        {{"point": "Brief headline", "detail": "Specific observation and actionable advice"}}
    ],
    "summary": "1-2 sentence technique assessment"
}}

Include 3-5 items in visual_feedback, all technique- or setup-focused.
Do NOT include any text outside the JSON."""


def _analyze_frames(frame_paths: list, instrument: str = "",
                    mode: str = "presence") -> dict:
    """Send frames to Claude Vision and get visual analysis."""

    # Build message content with images
    content = []
    for i, path in enumerate(frame_paths):
        with open(path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_data,
            }
        })
        content.append({
            "type": "text",
            "text": f"Frame {i+1} of {len(frame_paths)} from the performance video.",
        })

    instrument_note = ""
    if instrument:
        instrument_note = f"\n\nIMPORTANT: The performer is playing {instrument}. Only comment on this instrument. Do NOT assume they are playing other instruments that may be visible in the frame."

    if mode == "technique":
        prompt_text = _technique_prompt(instrument, instrument_note)
        system_prompt = f"""You are evaluating a musician's visible instrument technique from video frames.
Be constructive and specific. Focus on what the performer can control and improve.
You are NOT evaluating stage presence, charisma, or performance energy.

{TECHNIQUE_SENSITIVITY_RULES}"""
    else:
        prompt_text = _presence_prompt(instrument_note)
        system_prompt = f"""You are evaluating a musician's visual performance presence from video frames.
Be constructive and specific. Focus on what the performer can control and improve.

{SENSITIVITY_RULES}"""

    content.append({
        "type": "text",
        "text": prompt_text,
    })

    resp = http_requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": CLAUDE_MODEL,
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    raw_text = ""
    for block in result.get("content", []):
        if block.get("type") == "text":
            raw_text += block["text"]

    # Parse JSON
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
        return {
            "presence_score": parsed.get("presence_score", 5.0),
            "visual_feedback": parsed.get("visual_feedback", []),
            "summary": parsed.get("summary", ""),
            "analysis_type": mode,
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse visual analysis JSON: {e}")
        return {
            "presence_score": 5.0,
            "visual_feedback": [],
            "summary": text[:300],
            "parse_error": str(e),
            "analysis_type": mode,
        }
