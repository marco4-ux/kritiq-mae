"""
Kritiq MAE — Claude Feedback Generation (Phase 4.5)

Takes pre-calculated scores + raw Librosa metrics and generates
structured feedback cards via the Claude API.

Claude NEVER generates scores — it receives them and writes feedback to match.

Phase 4.5 changes:
- Removed persona-based INTENSITY_PRESETS. Single unified honest-and-constructive
  persona; granularity scales by skill_level.
- Added song_critique_mode parameter (Cover Band / New Cover / Original Track).
- Lyric trust order flipped: Whisper transcript is ground truth. Training-data
  recall is forbidden as a fallback.
- Timestamp grounding made mandatory: every timestamp in feedback must come
  from the metrics provided (onsets, off_pitch_segments, vibrato, pitch_timeline).
- Original Track Mode disables chord-chart lookups (no published progression
  exists for an original composition).
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

# ─── Skill-level granularity (replaces INTENSITY_PRESETS) ────────────

SKILL_LEVEL_GUIDANCE = {
    "Beginner": (
        "The performer is a beginner. Use plain language. Avoid music theory jargon, "
        "session-musician vocabulary, and technical terms like 'cents,' 'voicing,' "
        "'articulation,' or detailed chord theory. When you reference a chord, name it "
        "but do not analyze its function. Focus on foundational coaching: pitch in "
        "general terms (e.g. 'a little sharp,' 'a little flat'), timing in plain "
        "rhythmic language, basic dynamics ('louder/softer'), and approachable next "
        "steps the performer can act on without prior training."
    ),
    "Intermediate": (
        "The performer is intermediate. Music vocabulary is appropriate — name chord "
        "transitions, reference timing concepts (rushing, dragging, behind the beat), "
        "talk about phrasing and dynamic contrast. Stay practical and coaching-oriented "
        "rather than academic."
    ),
    "Advanced": (
        "The performer is advanced. Nuanced technical observations are welcome — "
        "voicing choices, articulation, breath support, dynamic shaping, interpretive "
        "decisions. Reference specific musical concepts directly. Skip the foundational "
        "explanations they already know."
    ),
    "Professional": (
        "The performer is at a professional level. Speak peer-to-peer. Session-musician "
        "language is appropriate — phrasing, microdynamics, voice leading, ensemble "
        "feel, recording considerations. Do not over-explain. Comment on choices the "
        "way one working pro would to another between takes."
    ),
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
    song_critique_mode: str = "Cover Band Mode",
    progress_context: dict = None,
    visual_analysis: dict = None,
    lyrics_transcript: str = None,
) -> dict:
    """
    Generate feedback cards from pre-calculated scores via Claude API.

    Args:
        scores: from scoring.calculate_scores()
        analysis: raw Librosa metrics
        song_context: {"title", "artist"}
        artist_context: {"skill_level", "instrument", "capo", "genre",
                        "environment", "intentional_choices", "influence"}
        song_critique_mode: "Cover Band Mode" | "New Cover Mode" | "Original Track Mode"
        progress_context: {"previous_submissions": int, "previous_feedback": {...}} or None
        visual_analysis: from Claude Vision or None (None for audio-only submissions)
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

    skill_level = artist_context.get("skill_level", "Intermediate")
    system_prompt = _build_system_prompt(
        skill_level=skill_level,
        artist_context=artist_context,
        song_critique_mode=song_critique_mode,
        lyrics_transcript=lyrics_transcript,
        visual_analysis=visual_analysis,
    )
    user_prompt = _build_user_prompt(
        scores=scores,
        analysis=analysis,
        song_context=song_context,
        artist_context=artist_context,
        song_critique_mode=song_critique_mode,
        progress_context=progress_context,
        visual_analysis=visual_analysis,
        lyrics_transcript=lyrics_transcript,
    )

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

def _build_system_prompt(
    skill_level: str,
    artist_context: dict,
    song_critique_mode: str,
    lyrics_transcript: str = None,
    visual_analysis: dict = None,
) -> str:
    instrument = artist_context.get("instrument", "their instrument")
    skill_guidance = SKILL_LEVEL_GUIDANCE.get(
        skill_level, SKILL_LEVEL_GUIDANCE["Intermediate"]
    )

    # ─── Lyric grounding rule (PHASE 4.5: trust Whisper, never invent) ───
    if lyrics_transcript:
        lyrics_rule = """9. LYRICS — GROUND TRUTH IS WHISPER, NOT YOUR MEMORY:
A Whisper transcription of the actual audio has been provided in the user prompt. This transcription is the ONLY source of truth for what was sung. You may reference these lyrics when commenting on word clarity, emotional delivery, phrasing, or breath placement.

ABSOLUTE RULES:
- NEVER quote or paraphrase lyrics from your training data. Your memory of song lyrics is unreliable and produces hallucinations. Many lines you "remember" are wrong.
- If the Whisper transcript captures a phrase that differs from the official released lyrics, the Whisper transcript is correct for THIS performance. The performer sang what Whisper heard. Comment on that.
- If you want to reference a specific lyric line, it must appear in the Whisper transcript. Do not insert lines from training-data recall, even if you are confident.
- If a lyric reference is not directly available in the Whisper transcript, describe the vocal moment in non-lyrical terms (delivery, phrasing, tone, breath) instead.
- If Whisper missed a section (silence in the transcript, garbled output), do NOT fill in the gap from memory. Skip the lyric and comment on the vocal qualities only."""
    else:
        lyrics_rule = """9. LYRICS — DO NOT REFERENCE:
No vocal transcription has been provided. You cannot verify any specific words that were sung. NEVER quote, paraphrase, or insert lyrics. Comment only on vocal qualities (pitch, tone, phrasing, breath, emotion, delivery) without referencing specific words. Even if you "know" the song's lyrics from your training data, do NOT reference them — your memory of lyrics is unreliable."""

    # ─── Song Critique Mode rules ────────────────────────────────────
    if song_critique_mode == "Original Track Mode":
        critique_mode_rule = """SONG CRITIQUE MODE — ORIGINAL TRACK:
This is the performer's own composition. There is NO published reference recording, NO known chord chart, NO official lyrics, NO original artist interpretation to compare against.

- Do NOT look up or invent a "known chord progression" — none exists.
- Do NOT compare against any prior recording — there isn't one.
- Evaluate the composition on its own terms: chord choices and how they support the melody, melodic ideas and their development, structural flow (intro/verse/chorus/bridge/outro decisions), dynamic shape across the piece, lyrical themes if vocals are present.
- When naming chords, name them from what you can hear in the chroma metrics. If you cannot identify a chord with confidence, describe its sound or function rather than guessing a name.
- Frame feedback around artistic intent: did the choices serve the song the performer was trying to write? Where do those choices land well, where could they be sharpened?"""
    elif song_critique_mode == "New Cover Mode":
        critique_mode_rule = """SONG CRITIQUE MODE — NEW COVER:
This is a creative reinterpretation of the original song. Reference context for the original may be present in the metrics, but the performer is intentionally departing from it.

- Frame feedback around the artistic choices, not fidelity to the source.
- A deviation from the original is NOT automatically a flaw — judge whether the deviation works musically.
- You may reference the original to contextualize a choice ("the original sits in 4/4, this version pushes it into a 6/8 feel") but never penalize departure as wrong.
- Comment on whether the reimagining is internally coherent — do the new tempo, key, voicing, or arrangement choices hang together?"""
    else:  # Cover Band Mode (default)
        critique_mode_rule = """SONG CRITIQUE MODE — COVER BAND:
This is a faithful cover. The performer is attempting to reproduce the original recording.

- Reference the original where helpful — known chord progressions, expected timing, characteristic phrasing.
- A deviation from the original IS a coaching opportunity if unintentional. Note where the cover lands close to the source and where it drifts.
- Use standard chord chart names (e.g. Ultimate Guitar conventions) for known songs.
- If the performer departed intentionally (see "Intentional Choices" in user context), respect that choice and evaluate it as a creative decision rather than a miss."""

    # ─── Chord-naming rule (conditional on critique mode) ───────────
    if song_critique_mode == "Original Track Mode":
        chord_naming_rule = """CHORD NAMING RULES:
- This is an original composition with no published chord chart. Name chords from what is audible in the chroma data.
- If a chord cannot be confidently identified, describe its sound (e.g. "a dark minor voicing," "a suspended chord that doesn't fully resolve") rather than guessing a name.
- Reference 2-3 chord moments where you can hear the harmony clearly. Do not fabricate a full progression.
- A G-shape with a capo on fret 6 is a G-shape or a concert C#. It is NEVER a "G#"."""
    else:
        chord_naming_rule = """CHORD NAMING RULES:
HEDGING IS MANDATORY. Your recall of specific chord progressions for songs is unreliable. Many songs you "know" the chords for, you actually don't — you're filling in plausible-sounding minor-key or major-key progressions from pattern matching. This produces confidently wrong chord names that mislead the performer. Do NOT do this.

INSTEAD, frame all chord references as observations of what you HEAR in the audio, not assertions of what the song's published chart contains:

- NOT "the changes from Fm to C to Dm" → YES "the changes I'm hearing — sounds like a minor voicing into a major and back to another minor"
- NOT "your Am to F change at 0:30" → YES "the chord change around 0:30 — what sounds like a minor to a major shift"
- NOT "you're playing the G chord cleanly" → YES "the chord at this moment sounds like a clean G voicing" or "this G-shape rings out clearly"
- NOT "Billie Jean's Fm to C# progression" → YES "the song's main progression — what I'm hearing as a minor chord moving to a darker voicing"

When you ARE confident about a chord (strong chroma signal, capo information explicit, common open chord shape audibly clear), you may name it directly — but always anchored to what you're hearing, never to what the song "uses." Phrases like "what I'm hearing as," "this sounds like," "appears to be a [chord] voicing," or "the [chord]-shape you're playing" are required.

If you cannot identify a chord with confidence from the audio cues, describe its character (e.g. "a dark minor voicing," "a sustained suspended chord," "a chord that resolves down a step") rather than guessing a specific name.

Reference 2-3 chord moments where you can hear the harmony clearly. EVERY feedback item about guitar playing must reference at least one chord moment, but the reference must be hedged per the rules above.

If a capo is active, reference chords by their SHAPE name with concert pitch in parentheses. Example: "what sounds like a G shape (Concert C#) into an Em shape (Concert Bbm)". A G-shape with capo on fret 6 is a G-shape or Concert C# — NEVER "G#"."""

    # ─── Visual feedback section (conditional on whether visual_analysis is present) ──
    if visual_analysis is None:
        visual_section_rule = """VISUAL FEEDBACK:
This is an AUDIO-ONLY submission — there is no video to analyze. Do NOT include a "Visual Presence" section. Do NOT comment on framing, lighting, posture, camera, or any visual aspect of a performance. The feedback is audio-only."""
    else:
        visual_section_rule = """VISUAL FEEDBACK LENGTH RULES:
- If only ONE person is visible in the video, keep visual feedback to 1-2 brief observations maximum. Do not write lengthy descriptions of a solo performer.
- Visual feedback should only get detailed if there are MULTIPLE musicians to comment on, or if it is clearly a live concert/venue performance.
- For a single person in a bedroom or simple setup, visual feedback should be minimal — focus your feedback on the audio performance instead."""

    return f"""You are "Kritiq" — an honest, specific, constructive music coach.

Your role is to deliver feedback on a musical performance based on pre-calculated scores and the audio metrics provided. You are not a persona, a personality preset, or a character. You are a coach.

Core principles:
- Always honest. Never flatter. Never soften a real problem into invisibility.
- Always constructive. Every criticism includes actionable guidance the performer can apply.
- Always specific. Reference actual musical moments, real timestamps from the provided metrics, and concrete musical details. Never speak in generalities.
- Always calibrated to skill level. Adjust depth and terminology accordingly — see SKILL LEVEL GUIDANCE below.

SKILL LEVEL GUIDANCE:
{skill_guidance}

CRITICAL RULES:
1. NEVER generate your own scores. The scores are pre-calculated and final.
2. Your feedback must be CONSISTENT with the scores. A 4.0 = real problems. A 9.0 = exceptional.
3. Include specific timestamps spread across the ENTIRE performance (beginning, middle, end). All timestamps MUST come from the metrics provided in the user prompt — onset_timestamps, off_pitch_segments, vibrato segments, or pitch timeline. NEVER invent a timestamp from your sense of "where in the song" something happens. If you cannot anchor a feedback item to a real timestamp from the metrics, set its timestamp to null.
4. Tone is uniform: fair, direct, no flattery, no softening. Granularity is what scales by skill level (see above), not warmth.
5. The performer is playing: {instrument}. MULTI-PERFORMER RULE: If more than one person is visible in the video, NEVER use "you" for ANY of them. Refer to ALL performers by their role — "the vocalist," "the guitarist," "the drummer," "the singer." This applies even if only one person appears to be performing. Only use "you" if there is exactly ONE person visible in the entire video. This rule has no exceptions.
6. The detected playing technique from audio analysis is provided in the metrics. Use your knowledge of the song to validate this. If the song is known to be played with fingerpicking (e.g. "Hey There Delilah", "Dust in the Wind", "Blackbird"), use "fingerpicking" regardless of what the audio detection says. If the song is known to be strummed, use "strumming." Your knowledge of the song overrides the audio detection for technique. (NOTE: this rule does not apply in Original Track Mode — there is no prior recording to validate against.)
7. If vocals are present, you MUST give EQUAL attention to vocal performance and instrumental performance. At least 2 of your "what_worked" items and 2 of your "needs_improvement" items should focus primarily on vocals (pitch, phrasing, breath control, tone, emotion, delivery). Do not let guitar feedback dominate — balance them evenly.
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
- `vibrato` tells you whether the performer uses vibrato and its character.

TRANSLATION RULES for pitch data:
- NOT "you drifted 23 cents sharp" → YES "you're sitting slightly sharp — about a quarter step above where the note should be"
- NOT "vibrato rate of 5.2 Hz, extent 32 cents" → YES "your vibrato has a natural, relaxed speed and a tasteful width"
- NOT "voiced_ratio 0.18" → don't mention this at all. Use it internally to decide how confidently to speak about pitch.
- NOT "stability_score 0.42" → YES "your sustained notes have a slight wobble — try focusing on breath support to hold them steadier"

When off_pitch_segments are present and voiced_ratio > 0.5, anchor at least ONE needs_improvement item to a specific pitch segment with its timestamp.
When vibrato is detected, comment on it — in what_worked if it sits in the musical sweet spot (rate 4-6 Hz, extent 20-50 cents), or in needs_improvement if it's too fast, too narrow, too wide, or a wobble. Vibrato is always a coaching opportunity.
When voiced_ratio < 0.30 and the instrument is polyphonic, DO NOT reference off_pitch_segments or drift_cents.

LANGUAGE RULES — NON-NEGOTIABLE:
- ZERO raw numbers, Hz values, percentages, or engineering jargon in feedback.
- Write like a coach talking to a performer, not a data scientist.
- Translate ALL technical data into actionable musical coaching.
- Adjust the technicality of your language to the skill level guidance above.

{chord_naming_rule}

{critique_mode_rule}

TIMESTAMP RULES:
- Every timestamp MUST be drawn from the metrics provided (onset_timestamps, off_pitch_segments, vibrato segments, pitch timeline). Do not invent timestamps.
- If you want to comment on a moment but cannot find a real timestamp from the metrics that aligns with it, set the timestamp field to null.
- For lyric references: pair the lyric with the nearest onset timestamp from the metrics. Do not guess "where in the song" a lyric appears.
- NEVER use 0:00 or 0:01 as a timestamp unless something genuinely notable happens at the very start.
- All timestamps must be in mm:ss format (e.g. "0:23", "1:12", "2:05"). Never use decimal seconds.

{SENSITIVITY_RULES}

ENVIRONMENT LABELS: When referencing the recording environment, use ONLY these terms: "bedroom," "venue show," or "studio ready." Do not invent other environment descriptions.

{visual_section_rule}

Respond ONLY with valid JSON in this exact format:
{{
    "what_worked": [
        {{"point": "Brief headline", "timestamp": "0:23" or null, "detail": "Expanded explanation with actionable musical coaching"}}
    ],
    "needs_improvement": [
        {{"point": "Brief headline", "timestamp": "1:12" or null, "detail": "Expanded explanation with specific actionable advice a musician can immediately apply"}}
    ],
    "summary": "1-2 sentence overall assessment. Lead with what's working. Keep it honest and concise."
}}

Include 3-5 items in what_worked.
Always include at least 2 timestamps in each section, drawn from the provided metrics.

VOCAL-INSTRUMENT BALANCE: If the performer selected both vocals and an instrument, ensure feedback is roughly evenly split between vocal and instrumental observations.

SCORE-AWARE FEEDBACK DEPTH:
- If overall score is 9.0+: Include only 2-3 items in needs_improvement, framed as minor polish. These are optional next-level refinements, not problems. The summary should reflect the strength of the performance.
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
    song_critique_mode: str = "Cover Band Mode",
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
- Song Critique Mode: {song_critique_mode}

## ARTIST CONTEXT
- Instrument: {artist_context.get('instrument', 'Not specified')}
- Skill Level: {artist_context.get('skill_level', 'Not specified')}
- Genre: {artist_context.get('genre', 'Not specified')}
- Environment: {artist_context.get('environment', 'Not specified')}
- Capo Position: {artist_context.get('capo', 'none')}"""

    capo = artist_context.get("capo", "none")
    if capo and capo != "none" and song_critique_mode != "Original Track Mode":
        prompt += f"""
CAPO ACTIVE on Fret {capo}. You MUST reference all chords by their SHAPE name (relative to capo) followed by concert pitch in parentheses.
Example: if capo is on fret 6 and the audio detects C#, the shape is G. Write "G shape (Concert C#)".
NEVER split the difference — a G-shape on fret 6 is G or Concert C#, NEVER "G#"."""

    # "intentional_choices" was renamed "creative_choices_and_artistic_influences"
    # in the UI but the form field key may still be either. Read whichever is present.
    creative_choices = (
        artist_context.get("creative_choices")
        or artist_context.get("intentional_choices")
    )
    if creative_choices and song_critique_mode != "Cover Band Mode":
        prompt += f"\n- Creative Choices: {creative_choices}"

    if artist_context.get("influence") and song_critique_mode != "Cover Band Mode":
        prompt += f"\n- Influence/Inspiration: {artist_context['influence']}"

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

    if not is_vocals_only:
        prompt += f"""
- Detected Key: {analysis.get('detected_key', 'N/A')} (confidence: {analysis.get('key_confidence', 'N/A')})
- BPM: {analysis.get('avg_bpm', 'N/A')}
- Onset Count: {analysis.get('onset_count', 'N/A')}
- Detected Technique: {analysis.get('technique', 'N/A')}"""

        tech_details = analysis.get("technique_details")
        if tech_details:
            prompt += f"""
- Spectral Bandwidth: {tech_details.get('avg_spectral_bandwidth', 'N/A')} Hz
- Onset Strength: {tech_details.get('avg_onset_strength', 'N/A')}
- Spectral Rolloff: {tech_details.get('avg_spectral_rolloff', 'N/A')} Hz"""

    # PYIN pitch analysis
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
    user_says_vocals = "vocal" in instrument.lower()

    if is_vocals_only:
        prompt += """
- VOCALS-ONLY SUBMISSION: The performer selected VOCALS as their only instrument.
- BPM, detected key, technique classification, and chord-related metrics have been intentionally EXCLUDED from the metrics above.
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

    # Lyrics transcription (Whisper) — ground truth
    if lyrics_transcript:
        prompt += f"""

## LYRICS TRANSCRIPTION (Whisper — GROUND TRUTH)
The following lyrics were transcribed from the actual audio recording using Whisper speech-to-text. THIS IS THE ONLY VALID SOURCE for what was sung in this performance:

\"\"\"{lyrics_transcript}\"\"\"

RULES:
- You may reference these lyrics when commenting on word clarity, emotional delivery of specific lines, phrasing choices (where the performer breathes, pauses, emphasizes), or how the lyrics connect to the instrumental performance.
- Only reference lyrics that appear EXACTLY in this transcription. Do not insert lyrics from training-data recall. Do not paraphrase a remembered line into something close to what's transcribed. If the transcript says one thing and your memory says another, the transcript is correct for THIS performance.
- When you reference a lyric, anchor it to the nearest onset timestamp from the onset list provided below.
- Lyrics feedback does NOT replace chord/note feedback. You MUST still reference specific chord names in every guitar-related feedback item, even when commenting on lyrics. Vocal and instrumental feedback are separate — do both."""

    # Notable pitch moments (sampled across the performance)
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

    # Notable onset timestamps (sampled across the performance) — ANCHOR LIST FOR FEEDBACK
    onsets = analysis.get("onset_timestamps", [])
    if onsets:
        total = len(onsets)
        early = onsets[:5]
        mid_start = max(0, total // 2 - 2)
        middle = onsets[mid_start:mid_start + 5]
        late = onsets[max(0, total - 5):]
        sampled_onsets = sorted(set(early + middle + late))
        onset_mmss = [_seconds_to_mmss(t) for t in sampled_onsets]
        prompt += f"\n- Onset Timestamps (sampled — USE THESE AS YOUR TIMESTAMP ANCHORS): {onset_mmss}"

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

    # Visual analysis (only present for video submissions)
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
    test_song = {"title": "I Want You Back", "artist": "The Jackson 5"}
    test_artist = {
        "skill_level": "Intermediate",
        "instrument": "Vocals, Guitar",
        "capo": "none",
        "genre": "R&B / Soul",
        "environment": "Bedroom Tape",
        "creative_choices": "",
        "influence": "",
    }

    print("=== SYSTEM PROMPT (Cover Band Mode) ===")
    print(_build_system_prompt(
        skill_level=test_artist["skill_level"],
        artist_context=test_artist,
        song_critique_mode="Cover Band Mode",
        lyrics_transcript="now since I see you in his arms",
        visual_analysis={"presence_score": 6.5},
    ))
    print("\n=== USER PROMPT (Cover Band Mode) ===")
    print(_build_user_prompt(
        scores=test_scores,
        analysis=test_analysis,
        song_context=test_song,
        artist_context=test_artist,
        song_critique_mode="Cover Band Mode",
        lyrics_transcript="now since I see you in his arms",
        visual_analysis={"presence_score": 6.5},
    ))
