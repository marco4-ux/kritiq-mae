"""
Kritiq MAE — Claude Feedback Generation

Takes pre-calculated scores + raw Librosa metrics and generates
structured feedback cards via the Claude API.

Claude NEVER generates scores — it receives them and writes feedback to match.
"""

import json
import logging
import requests as http_requests
import os

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ─── Genre list (for prompt context) ─────────────────────────────────

GENRE_LIST = [
    "Pop", "Rock", "R&B / Soul", "Country", "Hip-Hop / Rap",
    "Singer-Songwriter / Folk / Acoustic", "Alternative / Indie Rock", "Blues",
    "Jazz", "Funk / Disco", "Gospel / Spiritual", "Classical", "Musical Theatre",
    "Americana / Bluegrass", "Reggae / Ska", "Grunge / Post-Punk",
    "Synth-Pop / 80s Retro", "Afrobeats / Dancehall", "Latin Pop / Reggaeton",
    "K-Pop", "Bedroom Pop / DIY", "Lofi / Chill", "EDM / House / Techno",
    "Hyperpop / Glitch", "Dark Academia / Orchestral", "Metal / Progressive", "Other"
]

# ─── Sensitivity layer (hard-coded) ──────────────────────────────────

SENSITIVITY_RULES = """
VISUAL FEEDBACK BOUNDARIES (strictly enforced):
IN-BOUNDS (you may comment on): grooming, attire, framing, lighting, gear care, 
stage presence, posture related to playing technique, eye contact with camera/audience,
performance energy, movement that affects sound quality.

OUT-OF-BOUNDS (never comment on): body shape, weight, physical features, skin,
facial features, age appearance, physical disability, gender expression, 
ethnicity-related features, anything about the person's physical body that is 
not directly related to their musical performance technique.
"""

# ─── Intensity presets ───────────────────────────────────────────────

INTENSITY_PRESETS = {
    "Supportive Producer": {
        "persona": "You are a supportive music producer reviewing a student's cover. Use the sandwich method: lead with genuine praise, give constructive criticism, end with encouragement. Be warm but honest.",
        "tone": "encouraging, constructive, warm",
    },
    "Brutal": {
        "persona": "You are an honest industry veteran who has produced hits for 20 years. You respect the artist enough to give them the unfiltered truth. No sugarcoating, no filler. If something is bad, say it plainly. If something is good, acknowledge it briefly and move on.",
        "tone": "direct, blunt, no-nonsense, respectful but unsparing",
    },
    "Militant": {
        "persona": "You are a pure data analyst reviewing musical performance metrics. You speak only in terms of measurable quantities: pitch deviation percentages, timing drift in milliseconds, dynamic range in dB. No emotional language, no encouragement, no personality. Just the numbers and what they mean.",
        "tone": "clinical, metric-driven, impersonal",
    },
}

# ─── Score descriptions (for Claude context) ─────────────────────────

def _score_description(score: float) -> str:
    if score >= 9.0:
        return "world class — exceptional, professional-grade"
    elif score >= 8.0:
        return "very strong — polished and impressive"
    elif score >= 7.0:
        return "strong — solid skill with room to grow"
    elif score >= 6.0:
        return "above average — competent with notable areas for improvement"
    elif score >= 5.0:
        return "average — functional but clearly developing"
    elif score >= 4.0:
        return "below average — significant issues need work"
    elif score >= 3.0:
        return "weak — fundamental problems present"
    else:
        return "very weak — basic technique needs substantial development"


# ─── Main feedback generation ────────────────────────────────────────

def generate_feedback(
    scores: dict,
    analysis: dict,
    song_context: dict,
    artist_context: dict,
    progress_context: dict = None,
    visual_analysis: dict = None,
) -> dict:
    """
    Generate feedback cards from pre-calculated scores via Claude API.
    
    Args:
        scores: from scoring.calculate_scores()
        analysis: raw Librosa metrics
        song_context: {"title", "artist", "chord_reference", "original_audio_source"}
        artist_context: {"skill_level", "harshness", "style", "genre", "environment",
                        "intentional_choices", "influence"}
        progress_context: {"previous_submissions": int, "previous_feedback": {...}} or None
        visual_analysis: from Claude Vision or None
    
    Returns:
        {
            "what_worked": [{"point": str, "timestamp": str or null, "detail": str}],
            "needs_improvement": [{"point": str, "timestamp": str or null, "detail": str}],
            "summary": str,
            "raw_response": str,
        }
    """
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not set"}
    
    harshness = artist_context.get("harshness", "Supportive Producer")
    intensity = INTENSITY_PRESETS.get(harshness, INTENSITY_PRESETS["Supportive Producer"])
    
    system_prompt = _build_system_prompt(intensity, artist_context)
    user_prompt = _build_user_prompt(scores, analysis, song_context, artist_context, progress_context, visual_analysis)
    
    try:
        resp = http_requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 2000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        
        raw_text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                raw_text += block["text"]
        
        # Parse structured JSON from Claude's response
        feedback = _parse_feedback(raw_text)
        feedback["raw_response"] = raw_text
        return feedback
    
    except Exception as e:
        logger.exception("Claude feedback generation failed")
        return {"error": str(e)}


# ─── Prompt builders ─────────────────────────────────────────────────

def _build_system_prompt(intensity: dict, artist_context: dict) -> str:
    return f"""{intensity["persona"]}

You are reviewing a musician's cover performance. You will receive pre-calculated scores 
and raw audio analysis metrics. Your job is to write feedback that MATCHES these scores.

CRITICAL RULES:
1. NEVER generate your own scores. The scores are pre-calculated and final.
2. Your feedback must be CONSISTENT with the scores provided. A 4.0 technical score 
   means there are real problems — say so. A 9.0 means it's exceptional — reflect that.
3. Include specific timestamps from the analysis whenever possible.
4. Reference the actual metrics (pitch data, timing data, dynamics) to support your points.
5. Your tone is: {intensity["tone"]}
6. ONLY reference the instrument the performer specified. Do NOT assume they are playing other instruments visible in the video.

{SENSITIVITY_RULES}

Respond ONLY with valid JSON in this exact format:
{{
    "what_worked": [
        {{"point": "Brief headline", "timestamp": "0:23" or null, "detail": "Expanded explanation referencing specific metrics"}},
    ],
    "needs_improvement": [
        {{"point": "Brief headline", "timestamp": "1:12" or null, "detail": "Expanded explanation with specific actionable advice"}},
    ],
    "summary": "2-3 sentence overall assessment that reflects the scores"
}}

Include 3-5 items in what_worked and 3-5 items in needs_improvement.
Always include at least 2 timestamps in each section.
Timestamps MUST be in mm:ss format (e.g. "0:23", "1:12", "2:05"). Never use decimal seconds.
Do NOT include any text outside the JSON object."""


def _seconds_to_mmss(seconds: float) -> str:
    """Convert seconds (e.g. 4.99) to mm:ss format (e.g. '0:05')."""
    total = int(round(seconds))
    m = total // 60
    s = total % 60
    return f"{m}:{s:02d}"


def _build_user_prompt(
    scores: dict,
    analysis: dict,
    song_context: dict,
    artist_context: dict,
    progress_context: dict = None,
    visual_analysis: dict = None,
) -> str:
    
    overall = scores.get("overall", 0)
    technical = scores.get("technical", 0)
    emotional = scores.get("emotional", 0)
    
    prompt = f"""## PRE-CALCULATED SCORES (do not modify)
- Overall: {overall}/10 ({_score_description(overall)})
- Technical: {technical}/10 ({_score_description(technical)})
- Emotional: {emotional}/10 ({_score_description(emotional)})
- Pitch Accuracy: {scores.get('pitch_accuracy', 'N/A')}
- Timing Consistency: {scores.get('timing_consistency', 'N/A')}
- Chord Accuracy: {scores.get('chord_accuracy', 'N/A')}
- Dynamic Control: {scores.get('dynamic_control', 'N/A')}
- Tonal Clarity: {scores.get('tonal_clarity', 'N/A')}
- Scoring Mode: {scores.get('mode', 'creative')}

## SONG CONTEXT
- Song: "{song_context.get('title', 'Unknown')}" by {song_context.get('artist', 'Unknown')}

## ARTIST CONTEXT
- Instrument: {artist_context.get('instrument', 'Not specified')}
- Skill Level: {artist_context.get('skill_level', 'Not specified')}
- Style: {artist_context.get('style', 'Not specified')}
- Genre: {artist_context.get('genre', 'Not specified')}
- Environment: {artist_context.get('environment', 'Not specified')}"""

    if artist_context.get("intentional_choices"):
        prompt += f"\n- Intentional Choices: {artist_context['intentional_choices']}"
    
    if artist_context.get("influence"):
        prompt += f"\n- Influence/Inspiration: {artist_context['influence']}"

    # Raw metrics for Claude to reference
    prompt += f"""

## RAW AUDIO METRICS
- Detected Key: {analysis.get('detected_key', 'N/A')} (confidence: {analysis.get('key_confidence', 'N/A')})
- BPM: {analysis.get('avg_bpm', 'N/A')}
- Duration: {analysis.get('duration_seconds', 'N/A')}s
- Onset Count: {analysis.get('onset_count', 'N/A')}
- Average RMS: {analysis.get('avg_rms', 'N/A')}
- Dynamic Range: {analysis.get('dynamic_range', 'N/A')}
- Spectral Brightness: {analysis.get('avg_brightness', 'N/A')}
- Detected Technique: {analysis.get('technique', 'N/A')}"""

    # Technique details
    tech_details = analysis.get("technique_details")
    if tech_details:
        prompt += f"""
- Spectral Bandwidth: {tech_details.get('avg_spectral_bandwidth', 'N/A')} Hz
- Onset Strength: {tech_details.get('avg_onset_strength', 'N/A')}
- Spectral Rolloff: {tech_details.get('avg_spectral_rolloff', 'N/A')} Hz"""

    # Vocal-instrument coordination
    if analysis.get("has_vocals"):
        coord = analysis.get("coordination_score")
        if coord is not None:
            prompt += f"""
- Vocals Detected: Yes
- Vocal-Instrument Coordination: {coord} (1.0 = perfectly synchronized, 0.0 = completely independent)"""
    else:
        prompt += "\n- Vocals Detected: No (instrumental only)"

    # Notable pitch moments (first 20 for context)
    pitches = analysis.get("pitches_per_second", [])[:20]
    if pitches:
        pitch_str = ", ".join([f"{_seconds_to_mmss(p['time'])}:{p['note']}" for p in pitches])
        prompt += f"\n- Pitch Timeline (first 20s): {pitch_str}"
    
    # Notable onset timestamps
    onsets = analysis.get("onset_timestamps", [])[:20]
    if onsets:
        onset_mmss = [_seconds_to_mmss(t) for t in onsets]
        prompt += f"\n- Onset Timestamps (first 20): {onset_mmss}"

    # Progress context (previous submission comparison)
    if progress_context and progress_context.get("previous_submissions", 0) > 0:
        prev = progress_context.get("previous_feedback", {})
        prompt += f"""

## PROGRESS CONTEXT
This user has submitted this song {progress_context['previous_submissions']} time(s) before.
Previous scores: {json.dumps(progress_context.get('previous_scores', {}), indent=2)}
Previous feedback summary: {json.dumps(prev, indent=2) if prev else 'None available'}

IMPORTANT: Reference their previous submission specifically. In "what_worked", mention 
improvements since last time. In "needs_improvement", pick up where the last feedback 
left off rather than starting from scratch."""

    # Visual analysis
    if visual_analysis:
        prompt += f"""

## VISUAL ANALYSIS
{json.dumps(visual_analysis, indent=2)}
Note: Visual feedback must respect the sensitivity boundaries in the system prompt."""

    prompt += "\n\nGenerate the feedback JSON now."
    
    return prompt


# ─── Parse Claude's response ────────────────────────────────────────

def _parse_feedback(raw_text: str) -> dict:
    """Parse JSON from Claude's response, handling common formatting issues."""
    text = raw_text.strip()
    
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they're fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    
    try:
        parsed = json.loads(text)
        return {
            "what_worked": parsed.get("what_worked", []),
            "needs_improvement": parsed.get("needs_improvement", []),
            "summary": parsed.get("summary", ""),
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Claude JSON: {e}")
        # Return the raw text as fallback
        return {
            "what_worked": [],
            "needs_improvement": [],
            "summary": text[:500],
            "parse_error": str(e),
        }


# ─── Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Print what the prompt would look like (no API call)
    test_scores = {
        "overall": 6.5, "technical": 7.2, "emotional": 5.7,
        "pitch_accuracy": 0.849, "timing_consistency": 0.373,
        "chord_accuracy": 0.746, "dynamic_control": 0.38,
        "tonal_clarity": 0.85, "mode": "creative",
    }
    test_analysis = {
        "detected_key": "C#", "key_confidence": 0.773,
        "avg_bpm": 220, "duration_seconds": 29.36,
        "onset_count": 108, "avg_rms": 0.0085,
        "dynamic_range": 0.0326, "avg_brightness": 1901.3,
        "pitches_per_second": [
            {"note": "C", "time": 0, "confidence": 1},
            {"note": "G#", "time": 1, "confidence": 1},
            {"note": "C#", "time": 2, "confidence": 1},
        ],
        "onset_timestamps": [0.09, 0.6, 0.79, 1.09, 1.25],
    }
    test_song = {"title": "Hey There Delilah", "artist": "Plain White T's"}
    test_artist = {
        "skill_level": "Intermediate",
        "harshness": "Brutal",
        "style": "Interpretation",
        "genre": "Singer-Songwriter / Folk / Acoustic",
        "environment": "Bedroom Tape",
        "intentional_choices": "Slowed the tempo for emotional effect",
        "influence": "Jeff Buckley",
    }
    
    intensity = INTENSITY_PRESETS[test_artist["harshness"]]
    print("=== SYSTEM PROMPT ===")
    print(_build_system_prompt(intensity, test_artist))
    print("\n=== USER PROMPT ===")
    print(_build_user_prompt(test_scores, test_analysis, test_song, test_artist))
