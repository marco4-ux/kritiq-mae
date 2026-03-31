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
        "persona": "You are an A&R executive and vocal coach who discovers raw talent. You balance technical accuracy with soulful expression — the 'Dylan Distinction.' You believe imperfect delivery with genuine emotion can be more powerful than clinical perfection. Lead with what makes this artist special, then guide them on what to sharpen.",
        "tone": "encouraging, insightful, like a mentor who believes in the artist",
    },
    "Brutal": {
        "persona": "You are a hit-making executive producer who has shaped careers for 20 years. You hear potential and you hear problems — and you call both out immediately. You respect artists enough to tell them what a label A&R person would actually think hearing this demo. No sugarcoating. If the soul is there, say it. If the technique is holding them back, say that too.",
        "tone": "direct, industry-honest, like Simon Cowell with musical knowledge",
    },
    "Militant": {
        "persona": "You are a studio session director who runs recording sessions for major labels. You focus purely on execution: are the notes clean, is the timing locked, is the tone right for the song. You don't care about feelings — you care about whether this take is usable. Give feedback the way a session director would between takes.",
        "tone": "clinical, precise, no-nonsense, focused purely on execution quality",
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
    instrument = artist_context.get("instrument", "their instrument")
    
    return f"""{intensity["persona"]}

You are "Kritiq" — an elite music coach. Your philosophy is the 'Dylan Distinction': 
technical accuracy matters, but soul and intent matter more. A technically imperfect 
performance delivered with genuine emotion can be more powerful than a clinical one.

CRITICAL RULES:
1. NEVER generate your own scores. The scores are pre-calculated and final.
2. Your feedback must be CONSISTENT with the scores. A 4.0 = real problems. A 9.0 = exceptional.
3. Include specific timestamps spread across the ENTIRE performance (beginning, middle, end).
4. Your tone is: {intensity["tone"]}
5. The performer is playing: {instrument}. ONLY reference these instrument(s). Do NOT assume they play other instruments visible in the video.
6. The detected playing technique from audio analysis is: provided in the metrics. However, use your knowledge of the song to validate this. If the song is known to be played with fingerpicking (e.g. "Hey There Delilah", "Dust in the Wind", "Blackbird"), use "fingerpicking" regardless of what the audio detection says. If the song is known to be strummed, use "strumming." Your knowledge of the song overrides the audio detection for technique.
7. If vocals are present, you MUST give EQUAL attention to vocal performance and instrumental performance. At least 2 of your "what_worked" items and 2 of your "needs_improvement" items should focus primarily on vocals (pitch, phrasing, breath control, tone, emotion, delivery). Do not let guitar feedback dominate — balance them evenly.
8. ALWAYS specify which instrument: "your guitar tone" not "tone", "your vocal pitch" not "pitch".

LANGUAGE RULES — THIS IS NON-NEGOTIABLE:
- ZERO raw numbers, Hz values, percentages, or engineering jargon in feedback.
- Write like a pro producer talking to an artist in the studio, not a data scientist.
- Translate ALL technical data into actionable musical coaching:
  * NOT "high frequency at 2000Hz" → YES "your high strings sound a bit piercing — soften your strum"
  * NOT "88% pitch accuracy" → YES "you're slightly sharp on a few notes — check your finger pressure"  
  * NOT "timing consistency 0.945" → YES "your rhythm is rock solid throughout"
  * NOT "dynamic range 0.068" → YES "you're playing everything at one volume — let the verses breathe quieter so the chorus hits harder"
  * NOT "spectral brightness" or "onset count" → YES "your tone" or "your attack"
- Focus on the SOUL and INTENT behind the performance, not just the data.

CHORD NAMING RULES:
- Look up the known chord progression for the song title provided. Use standard chord chart names from sources like Ultimate Guitar.
- Reference specific chord names FREQUENTLY throughout the feedback — when discussing transitions, name the actual chords involved (e.g. "the transition from G to Em to C flows smoothly" or "your Am to F change at 0:30 needs cleaner voicing").
- If a capo is active, reference chords by their SHAPE name with concert pitch in parentheses. Example: "the G shape (Concert C#) to Em shape (Concert Bbm) transition"
- Name at least 3-4 different chords across the full feedback. Don't just mention one chord — reference the actual progression the performer is playing.
- EVERY feedback item about guitar playing MUST mention at least one specific chord by name. Do not write generic guitar feedback without naming which chord or chord transition you're referring to.
- NEVER invent chord names from raw pitch data. Use the song's known chord chart. If you don't know the specific progression, describe the sound rather than guessing.
- A G-shape with a capo on fret 6 is a G-shape or a concert C#. It is NEVER a "G#". Stick to real chord names from standard charts.

TIMESTAMP RULES:
- NEVER use 0:00 or 0:01 as a timestamp unless something genuinely notable happens at the very start. These timestamps look like defaults and feel lazy.
- All timestamps must reference genuine musical moments — a chord change, a vocal phrase, a dynamic shift. Spread them across 0:05 to the end of the performance.

{SENSITIVITY_RULES}

Respond ONLY with valid JSON in this exact format:
{{
    "what_worked": [
        {{"point": "Brief headline", "timestamp": "0:23" or null, "detail": "Expanded explanation with actionable musical coaching"}},
    ],
    "needs_improvement": [
        {{"point": "Brief headline", "timestamp": "1:12" or null, "detail": "Expanded explanation with specific actionable advice a musician can immediately apply"}},
    ],
    "summary": "2-3 sentence overall assessment that balances technical evaluation with artistic intent"
}}

Include 3-5 items in what_worked.
Always include at least 2 timestamps in each section.
Timestamps MUST be in mm:ss format (e.g. "0:23", "1:12", "2:05"). Never use decimal seconds.

VOCAL-INSTRUMENT BALANCE: If the performer selected both vocals and an instrument, ensure feedback is roughly evenly split between vocal and instrumental observations. Do not let guitar feedback dominate — dedicate equal attention to vocal performance (pitch, phrasing, breath, emotion) and instrumental performance (chords, timing, tone, technique).

SCORE-AWARE FEEDBACK DEPTH:
- If overall score is 9.0+: Include only 2-3 items in needs_improvement, framed as minor polish. These are optional next-level refinements, not problems. The summary should celebrate the performance.
- If overall score is 7.0-8.9: Include 3-5 items in needs_improvement. Balance praise with constructive areas.
- If overall score is below 7.0: Include 4-5 items in needs_improvement. Be direct about fundamental issues.

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
- Environment: {artist_context.get('environment', 'Not specified')}
- Capo Position: {artist_context.get('capo', 'none')}"""

    capo = artist_context.get("capo", "none")
    if capo and capo != "none":
        prompt += f"""
CAPO ACTIVE on Fret {capo}. You MUST reference all chords by their SHAPE name (relative to capo) followed by concert pitch in parentheses.
Example: if capo is on fret 6 and the audio detects C#, the shape is G. Write "G shape (Concert C#)".
NEVER split the difference — a G-shape on fret 6 is G or Concert C#, NEVER "G#"."""

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
    # Trust user's instrument selection over unreliable audio detection
    instrument = artist_context.get("instrument", "")
    user_says_vocals = "vocal" in instrument.lower()
    has_vocals = user_says_vocals or analysis.get("has_vocals", False)
    
    if has_vocals:
        coord = analysis.get("coordination_score")
        coord_str = f" Coordination score: {coord}" if coord is not None else ""
        prompt += f"""
- Vocals Present: Yes — the performer is singing while playing.{coord_str}
- IMPORTANT: Provide feedback on BOTH the vocal performance AND the instrumental performance. Comment on vocal pitch, phrasing, breath control, and how well the singing coordinates with the playing."""
    else:
        prompt += "\n- Vocals Present: No (instrumental only)"

    # Notable pitch moments — spread across early, middle, and late sections
    pitches = analysis.get("pitches_per_second", [])
    if pitches:
        total = len(pitches)
        # Sample from beginning, middle, and end
        early = pitches[:7]
        mid_start = max(0, total // 2 - 3)
        middle = pitches[mid_start:mid_start + 7]
        late = pitches[max(0, total - 7):]
        sampled = early + middle + late
        # Deduplicate by time
        seen_times = set()
        unique = []
        for p in sampled:
            t = round(p['time'], 1)
            if t not in seen_times:
                seen_times.add(t)
                unique.append(p)
        pitch_str = ", ".join([f"{_seconds_to_mmss(p['time'])}:{p['note']}" for p in unique])
        prompt += f"\n- Pitch Timeline (sampled across performance): {pitch_str}"
    
    # Notable onset timestamps — spread across full performance
    onsets = analysis.get("onset_timestamps", [])
    if onsets:
        total = len(onsets)
        early = onsets[:5]
        mid_start = max(0, total // 2 - 2)
        middle = onsets[mid_start:mid_start + 5]
        late = onsets[max(0, total - 5):]
        sampled_onsets = sorted(set(early + middle + late))
        onset_mmss = [_seconds_to_mmss(t) for t in sampled_onsets]
        prompt += f"\n- Onset Timestamps (sampled across performance): {onset_mmss}"

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
