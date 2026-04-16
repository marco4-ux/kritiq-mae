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
stage presence, eye contact with camera/audience,
performance energy, movement that affects sound quality.
Do NOT comment on posture. Do NOT comment on body positioning unless it directly affects instrument technique.

OUT-OF-BOUNDS (never comment on): body shape, weight, physical features, skin,
facial features, age, appearance, physical disability, gender expression, 
ethnicity-related features, posture, anything about the person's physical body that is 
not directly related to their musical performance technique.

CAMERA RULES: 
DEFAULT ASSUMPTION: The video is recorded on a single phone or camera with no edits.
Do NOT mention camera angles, cuts, or editing unless the video is CLEARLY a professional multi-camera production (e.g. concert venue with stage lighting, professional music video).
For any bedroom, home, or simple setup recording — NEVER mention camera angles, editing, or video production. Any visual variation between frames is the performer moving, not a camera change.
Do NOT mention: camera angles, multiple angles, camera work, camera transitions, different shots, editing, cuts, video production, camera switching, angle changes, or multi-camera.
Any visual variation between frames is the performer moving within a SINGLE static shot — it is NEVER a camera change.
If your feedback contains ANY of the above forbidden camera terms, the entire feedback is invalid.
This rule overrides all other observations about the video.
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
    lyrics_transcript: str = None,
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
        lyrics_transcript: Whisper transcription of vocals or None
    
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
    
    system_prompt = _build_system_prompt(intensity, artist_context, lyrics_transcript=lyrics_transcript)
    user_prompt = _build_user_prompt(scores, analysis, song_context, artist_context, progress_context, visual_analysis, lyrics_transcript=lyrics_transcript)
    
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
            timeout=90,
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

def _build_system_prompt(intensity: dict, artist_context: dict, lyrics_transcript: str = None) -> str:
    instrument = artist_context.get("instrument", "their instrument")
    
    # Rule 9: conditional based on whether we have a real transcription
    if lyrics_transcript:
        lyrics_rule = """9. LYRICS FEEDBACK (transcription available): A Whisper transcription of the vocals has been provided AND the song title/artist is known. Look up the OFFICIAL lyrics for this song from your training data and use those as the ground truth. The Whisper transcription may contain errors - if the official lyrics differ from the transcription, trust the official lyrics. Comment on delivery, phrasing, word clarity, and emotional interpretation based on the official lyrics. If you do not know the official lyrics for this song, fall back to the Whisper transcription only."""
    else:
        lyrics_rule = """9. Do NOT quote or reference specific lyrics. You cannot hear exact words from the audio. Only describe vocal qualities (pitch, tone, phrasing, breath, emotion, delivery) and instrumental qualities. Never write lyrics in quotation marks or reference specific lyric lines. If you know the song's lyrics from your training data, do NOT insert them into feedback - you have no way to verify the performer actually sang those words."""

    return f"""{intensity["persona"]}

You are "Kritiq" - an elite music coach. Your philosophy is the 'Dylan Distinction': 
technical accuracy matters, but soul and intent matter more. A technically imperfect 
performance delivered with genuine emotion can be more powerful than a clinical one.

CRITICAL RULES:
1. NEVER generate your own scores. The scores are pre-calculated and final.
2. Your feedback must be CONSISTENT with the scores. A 4.0 = real problems. A 9.0 = exceptional.
3. Include specific timestamps spread across the ENTIRE performance (beginning, middle, end).
4. Your tone is: {intensity["tone"]}
5. The performer is playing: {instrument}. MULTI-PERFORMER RULE: If more than one person is visible in the video, NEVER use "you" for ANY of them. Refer to ALL performers by their role - "the vocalist," "the guitarist," "the drummer," "the singer." This applies even if only one person appears to be performing. Only use "you" if there is exactly ONE person visible in the entire video. This rule has no exceptions.
6. The detected playing technique from audio analysis is: provided in the metrics. However, use your knowledge of the song to validate this. If the song is known to be played with fingerpicking (e.g. "Hey There Delilah", "Dust in the Wind", "Blackbird"), use "fingerpicking" regardless of what the audio detection says. If the song is known to be strummed, use "strumming." Your knowledge of the song overrides the audio detection for technique.
7. If vocals are present, you MUST give EQUAL attention to vocal performance and instrumental performance. At least 2 of your "what_worked" items and 2 of your "needs_improvement" items should focus primarily on vocals (pitch, phrasing, breath control, tone, emotion, delivery). Do not let guitar feedback dominate - balance them evenly.
8. ALWAYS specify which instrument: "your guitar tone" not "tone", "your vocal pitch" not "pitch".
{lyrics_rule}
10. PITCH ANALYSIS — HOW TO USE f0 DATA:
The user prompt may include a "PITCH ANALYSIS" section with PYIN-derived metrics. These come from the actual fundamental frequency of the audio and are much more precise than chroma-based pitch detection.

- `voiced_ratio` tells you how much of the audio had a clear single fundamental. 
  * Above 0.70 = monophonic performance (solo vocals, fingerpicked single notes). Trust pitch metrics fully.
  * 0.30 to 0.70 = mixed (vocals with guitar, some strummed sections). Comment on pitch where it's clear, avoid strong claims elsewhere.
  * Below 0.30 = mostly polyphonic or strummed. DO NOT cite specific cents-off values or pitch drift. You can still comment on general pitch feel from the chroma data, but pitch-specific claims must be hedged or avoided.
- `stability_score` (0-1) reflects how steady the pitch is within notes. High = clean sustained tone. Low = wobbly, drifting, or shaky.
- `drift_cents` is the average deviation from nearest semitone. Positive = sharp, negative = flat. |drift| under 10 cents is imperceptible. 10-25 cents is slight. Over 25 cents is noticeable out-of-tune.
- `off_pitch_segments` are specific moments where the performer held a note off-pitch for at least 200ms. Each has a timestamp, duration, and deviation in cents. USE THESE FOR TIMESTAMP-ANCHORED FEEDBACK when voiced_ratio is high enough to trust them.
- `vibrato` tells you whether the performer uses vibrato and its character:
  * rate_hz: 4-6 Hz is natural singer vibrato, 6-7 Hz is slightly fast/excited, above 7 Hz is a tremor or too fast, below 4 Hz is a wobble rather than vibrato
  * extent_cents: 20-50 cents is musical and expressive, above 60 is operatic/wide, below 15 is barely-there
  * segments lists specific sustained notes where vibrato was detected

TRANSLATION RULES for pitch data:
- NOT "you drifted 23 cents sharp" → YES "you're sitting slightly sharp — about a quarter step above where the note should be"
- NOT "vibrato rate of 5.2 Hz, extent 32 cents" → YES "your vibrato has a natural, relaxed speed and a tasteful width — it sits right in the sweet spot"
- NOT "voiced_ratio 0.18" → don't mention this at all. Just use it internally to decide how confidently to speak about pitch.
- NOT "stability_score 0.42" → YES "your sustained notes have a slight wobble — try focusing on breath support to hold them steadier"

When off_pitch_segments are present and voiced_ratio > 0.5, anchor at least ONE needs_improvement item to a specific pitch segment with its timestamp.

When vibrato is detected, comment on it in what_worked (if it's in the musical sweet spot) or needs_improvement (if it's too fast, too narrow, too wide, or a wobble). Vibrato is a coaching opportunity — always comment on it one way or the other.

When voiced_ratio < 0.30 and the instrument is polyphonic (strummed guitar, piano chords), DO NOT reference off_pitch_segments or drift_cents. State pitch feedback in general terms from chord/chroma data only.

LANGUAGE RULES - THIS IS NON-NEGOTIABLE:
- ZERO raw numbers, Hz values, percentages, or engineering jargon in feedback.
- Write like a pro producer talking to an artist in the studio, not a data scientist.
- Translate ALL technical data into actionable musical coaching:
  * NOT "high frequency at 2000Hz" -> YES "your high strings sound a bit piercing - soften your strum"
  * NOT "88% pitch accuracy" -> YES "you're slightly sharp on a few notes - check your finger pressure"
  * NOT "timing consistency 0.945" -> YES "your rhythm is rock solid throughout"
  * NOT "dynamic range 0.068" -> YES "you're playing everything at one volume - let the verses breathe quieter so the chorus hits harder"
  * NOT "spectral brightness" or "onset count" -> YES "your tone" or "your attack"
- Focus on the SOUL and INTENT behind the performance, not just the data.

CHORD NAMING RULES:
- Look up the known chord progression for the song title provided. Use standard chord chart names from sources like Ultimate Guitar.
- Reference specific chord names FREQUENTLY throughout the feedback - when discussing transitions, name the actual chords involved (e.g. "the transition from G to Em to C flows smoothly" or "your Am to F change at 0:30 needs cleaner voicing").
- If a capo is active, reference chords by their SHAPE name with concert pitch in parentheses. Example: "the G shape (Concert C#) to Em shape (Concert Bbm) transition"
- Name at least 3-4 different chords across the full feedback. Don't just mention one chord - reference the actual progression the performer is playing.
- EVERY feedback item about guitar playing MUST mention at least one specific chord by name. Do not write generic guitar feedback without naming which chord or chord transition you're referring to.
- NEVER invent chord names from raw pitch data. Use the song's known chord chart. If you don't know the specific progression, describe the sound rather than guessing.
- A G-shape with a capo on fret 6 is a G-shape or a concert C#. It is NEVER a "G#". Stick to real chord names from standard charts.

TIMESTAMP RULES:
- NEVER use 0:00 or 0:01 as a timestamp unless something genuinely notable happens at the very start. These timestamps look like defaults and feel lazy.
- All timestamps must reference genuine musical moments - a chord change, a vocal phrase, a dynamic shift. Spread them across 0:05 to the end of the performance.

{SENSITIVITY_RULES}

ENVIRONMENT LABELS: When referencing the recording environment, use ONLY these terms: "bedroom," "venue show," or "studio ready." Do not invent other environment descriptions.

VISUAL FEEDBACK LENGTH RULES:
- If only ONE person is visible in the video, keep visual feedback to 1-2 brief observations maximum. Do not write lengthy descriptions of a solo performer.
- Visual feedback should only get detailed if there are MULTIPLE musicians to comment on, or if it is clearly a live concert/venue performance.
- For a single person in a bedroom or simple setup, visual feedback should be minimal - focus your feedback on the audio performance instead.

Respond ONLY with valid JSON in this exact format:
{{
    "what_worked": [
        {{"point": "Brief headline", "timestamp": "0:23" or null, "detail": "Expanded explanation with actionable musical coaching"}}
    ],
    "needs_improvement": [
        {{"point": "Brief headline", "timestamp": "1:12" or null, "detail": "Expanded explanation with specific actionable advice a musician can immediately apply"}}
    ],
    "summary": "1-2 sentence overall assessment. Lead with what's working. Keep it encouraging and concise."
}}

Include 3-5 items in what_worked.
Always include at least 2 timestamps in each section.
Timestamps MUST be in mm:ss format (e.g. "0:23", "1:12", "2:05"). Never use decimal seconds.

VOCAL-INSTRUMENT BALANCE: If the performer selected both vocals and an instrument, ensure feedback is roughly evenly split between vocal and instrumental observations. Do not let guitar feedback dominate - dedicate equal attention to vocal performance (pitch, phrasing, breath, emotion) and instrumental performance (chords, timing, tone, technique).

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
    lyrics_transcript: str = None,
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
    # Detect vocals-only submission to strip misleading instrument metrics
    instrument = artist_context.get("instrument", "")
    is_vocals_only = (
        instrument
        and 'vocal' in instrument.lower()
        and not any(x in instrument.lower() for x in ['guitar', 'piano', 'bass', 'drum', 'ukulele', 'keyboard', 'violin', 'banjo', 'mandolin', 'harmonica', 'saxophone', 'trumpet', 'brass', 'cello'])
    )

    prompt += f"""

## RAW AUDIO METRICS
- Duration: {analysis.get('duration_seconds', 'N/A')}s
- Average RMS: {analysis.get('avg_rms', 'N/A')}
- Dynamic Range: {analysis.get('dynamic_range', 'N/A')}
- Spectral Brightness: {analysis.get('avg_brightness', 'N/A')}"""

    # Only include chord/technique/BPM metrics for non-vocal-only submissions.
    # On vocals-only audio, these fields are noise — chroma detects a "key" from
    # vowel formants, beat tracker returns a "BPM" from syllable onsets, and
    # technique classifier labels every vocal as "strumming" or "fingerpicking".
    # Passing these to Claude causes it to hallucinate chord progressions and
    # playing techniques that never existed.
    if not is_vocals_only:
        prompt += f"""
- Detected Key: {analysis.get('detected_key', 'N/A')} (confidence: {analysis.get('key_confidence', 'N/A')})
- BPM: {analysis.get('avg_bpm', 'N/A')}
- Onset Count: {analysis.get('onset_count', 'N/A')}
- Detected Technique: {analysis.get('technique', 'N/A')}"""

        # Technique details
        tech_details = analysis.get("technique_details")
        if tech_details:
            prompt += f"""
- Spectral Bandwidth: {tech_details.get('avg_spectral_bandwidth', 'N/A')} Hz
- Onset Strength: {tech_details.get('avg_onset_strength', 'N/A')}
- Spectral Rolloff: {tech_details.get('avg_spectral_rolloff', 'N/A')} Hz"""
    
    # PYIN pitch analysis — f0-level pitch metrics with confidence gating
    pitch_analysis = analysis.get("pitch_analysis")
    if pitch_analysis:
        voiced_ratio = pitch_analysis.get("voiced_ratio", 0)
        prompt += f"""

## PITCH ANALYSIS (PYIN — actual fundamental frequency)
- Voiced Ratio: {voiced_ratio} (fraction of audio with clear single fundamental)"""
        
        if voiced_ratio >= 0.30:
            stability = pitch_analysis.get("stability_score")
            drift = pitch_analysis.get("drift_cents")
            
            if stability is not None:
                prompt += f"\n- Stability Score: {stability} (higher = steadier pitch within notes)"
            if drift is not None:
                drift_direction = "sharp" if drift > 0 else "flat" if drift < 0 else "in tune"
                prompt += f"\n- Average Drift: {drift} cents ({drift_direction} of nearest semitone)"
            
            off_segs = pitch_analysis.get("off_pitch_segments", [])
            if off_segs:
                prompt += f"\n- Off-Pitch Segments ({len(off_segs)} detected, sustained >200ms with >25 cent deviation):"
                for seg in off_segs[:8]:
                    time_mmss = _seconds_to_mmss(seg["time"])
                    direction = "sharp" if seg["deviation_cents"] > 0 else "flat"
                    prompt += f"\n  • {time_mmss}: {abs(seg['deviation_cents'])} cents {direction} for {seg['duration']}s (conf {seg['confidence']})"
            else:
                prompt += "\n- Off-Pitch Segments: none detected — pitch held cleanly throughout"
            
            vibrato = pitch_analysis.get("vibrato", {})
            if vibrato.get("detected"):
                rate = vibrato.get("rate_hz")
                extent = vibrato.get("extent_cents")
                prompt += f"\n- Vibrato: detected — rate {rate} Hz, extent {extent} cents"
                vib_segs = vibrato.get("segments", [])
                if vib_segs:
                    prompt += f"\n  Vibrato segments ({len(vib_segs)}):"
                    for seg in vib_segs[:5]:
                        time_mmss = _seconds_to_mmss(seg["time"])
                        prompt += f"\n  • {time_mmss}: {seg['rate_hz']} Hz at {seg['extent_cents']} cents for {seg['duration']}s"
            else:
                prompt += "\n- Vibrato: not detected"
        else:
            prompt += f"""
- NOTE: voiced_ratio is low ({voiced_ratio}) — this is polyphonic or strummed audio where PYIN cannot reliably track a single fundamental. DO NOT cite specific cents-off values, drift numbers, or off-pitch segments. Comment on pitch only at the chord/chroma level."""

    # Vocal-instrument coordination
    instrument = artist_context.get("instrument", "")
    user_says_vocals = "vocal" in instrument.lower()

    # Instrument hallucination guard — now uses is_vocals_only computed earlier
    if is_vocals_only:
        prompt += """
- VOCALS-ONLY SUBMISSION: The performer selected VOCALS as their only instrument.
- You will notice that BPM, detected key, technique classification, and chord-related metrics have been intentionally EXCLUDED from the metrics above.
- This is because those metrics are misleading on a capella audio — chroma analysis detects a "key" from vowel formants, beat tracking returns a "BPM" from syllable onsets, and the technique classifier labels every vocal as "strumming" or "fingerpicking". These signals do NOT reflect a real instrument.
- Do NOT invent chord names, BPMs, key signatures, strumming patterns, fingerpicking, or any other instrumental characteristic.
- Do NOT mention guitar, piano, bass, drums, or any other instrument in your feedback.
- ALL feedback must be about vocal performance only: pitch, tone, phrasing, breath control, emotion, delivery, lyrics, timing of vocal phrases."""
    
    if user_says_vocals:
        coord = analysis.get("coordination_score")
        coord_str = f" Coordination score: {coord}" if coord is not None else ""
        prompt += f"""
- Vocals Present: Yes — the performer is singing while playing.{coord_str}
- IMPORTANT: Provide feedback on BOTH the vocal performance AND the instrumental performance. Comment on vocal pitch, phrasing, breath control, and how well the singing coordinates with the playing."""
    else:
        prompt += """
- Vocals Present: No (instrumental only). The performer did NOT select vocals.
- Do NOT comment on singing, vocal pitch, vocal tone, breath control, or any vocal performance.
- If you hear vocals in the audio, they are from a backing track, NOT the performer.
- ONLY evaluate the selected instrument(s): """ + instrument

    # Lyrics transcription (from Whisper)
    if lyrics_transcript:
        prompt += f"""

## LYRICS TRANSCRIPTION (from Whisper — verified audio)
The following lyrics were transcribed from the actual audio recording using speech-to-text:
\"\"\"{lyrics_transcript}\"\"\"

IMPORTANT: You may reference these specific lyrics in your feedback to comment on:
- Word clarity and enunciation
- Emotional delivery of specific lines
- Phrasing choices (where the performer breathes, pauses, emphasizes)
- How well the lyrics connect with the instrumental performance
Only reference lyrics that appear in this transcription. Do not add lyrics from your training data.

IMPORTANT: Lyrics feedback does NOT replace chord/note feedback. You MUST still reference specific chord names in every guitar-related feedback item, even when commenting on lyrics. Vocal and instrumental feedback are separate — do both."""

    # Notable pitch moments
    pitches = analysis.get("pitches_per_second", [])
    if pitches:
        total = len(pitches)
        early = pitches[:7]
        mid_start = max(0, total // 2 - 3)
        middle = pitches[mid_start:mid_start + 7]
        late = pitches[max(0, total - 7):]
        sampled = early + middle + late
        seen_times = set()
        unique = []
        for p in sampled:
            t = round(p['time'], 1)
            if t not in seen_times:
                seen_times.add(t)
                unique.append(p)
        pitch_str = ", ".join([f"{_seconds_to_mmss(p['time'])}:{p['note']}" for p in unique])
        prompt += f"\n- Pitch Timeline (sampled across performance): {pitch_str}"
    
    # Notable onset timestamps
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

    # Progress context
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
Note: Visual feedback must respect the sensitivity boundaries in the system prompt.
IMPORTANT: These are 3 frames sampled from a SINGLE continuous video recording at different timestamps. Do NOT assume multiple camera angles or editing cuts. Any differences between frames are from the performer moving within ONE static shot. Never reference "multiple angles," "camera switches," or "different shots" in your feedback. When referencing visual observations, use the TIMESTAMP (e.g. "at 0:23") not the frame number. Never say "Frame 1," "Frame 2," or "Frame 3" — convert to timestamps."""

    prompt += "\n\nGenerate the feedback JSON now."
    
    return prompt


# ─── Parse Claude's response ────────────────────────────────────────

def _parse_feedback(raw_text: str) -> dict:
    """Parse JSON from Claude's response, handling common formatting issues."""
    text = raw_text.strip()
    
    # Remove markdown code fences if present
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
            "what_worked": parsed.get("what_worked", []),
            "needs_improvement": parsed.get("needs_improvement", []),
            "summary": parsed.get("summary", ""),
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Claude JSON: {e}")
        return {
            "what_worked": [],
            "needs_improvement": [],
            "summary": text[:500],
            "parse_error": str(e),
        }


# ─── Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
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
