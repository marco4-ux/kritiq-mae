"""
Kritiq MAE — Scoring Calibration Module

Converts raw Librosa metrics into deterministic scores (1-10 scale).
Scores are calculated BEFORE Claude sees them. Claude receives pre-calculated
scores and writes feedback to match — never the other way around.

Score ranges (Andy's spec):
    9.0+    World class
    7.0-8.9 Strong
    5.0-6.9 Average
    3.0-4.9 Below average
    Below 3  Fundamental problems

Two modes:
    Strict  — compare user performance against Deezer reference
    Creative — score internal consistency only

Skill level affects Claude's feedback tone ONLY — never the scores.
A 6.5 performance is a 6.5 regardless of skill level selected. (This is now
TRUE in the code; previously skill level silently compressed the numeric
score into a per-tier band, which manufactured fake spread and inflated/
deflated identical audio by up to ~2 points.)

── Scoring rework (de-scaffold + recalibration) ─────────────────────────
WHAT CHANGED and WHY:

1. REMOVED the skill_ranges floor/ceiling system entirely. It was the
   "scaffolding": identical audio scored 5.6 / 6.6 / 7.5 purely by which
   skill tier the user self-declared. That spread was fake (driven by a
   self-reported dial, not the performance) and produced absurd outliers
   (a flat-dynamics track at 9.0, a wavering gospel clip floored up to 7.5).
   Skill level now only reaches Claude's feedback tone, never the number.

2. RECALIBRATED _calibrate_to_10. With the scaffold gone, raw scores float
   to their true curve position — and the OLD curve mapped most concentrated/
   in-time performances into 8-9, which would have pushed the church clip UP.
   The new curve steepens the top so genuinely exceptional metrics are needed
   for 8+, spreads the middle wider, and makes 9+ rare.

3. Sub-metric percentages are now reported RAW (no skill_ceiling scaling).

KNOWN LIMITS (Layer 2 — deferred, the "axes" work):
  - Real spread is capped by input variance: timing_consistency sits at
    ~0.92-0.98 for nearly everything, and dynamic_control maxes at 0.9, so
    most covers still bunch in the upper-middle. Widening this needs NEW
    independent axes (dynamic-envelope variation, harmonic complexity), not
    just a curve. That is the next layer, after this ships and backtests.
  - Because emotional_raw is structurally capped near ~0.92 (dynamic_control
    ceiling), 9+ is currently very hard to reach even for excellent audio.
    If Andy wants 9 to be achievable-but-rare rather than near-impossible,
    that is a sub-metric fix in Layer 2.
"""

import numpy as np
from typing import Optional


def calculate_scores(
    user_analysis: dict,
    reference_analysis: Optional[dict] = None,
    mode: str = "creative",  # "strict" or "creative"
    skill_level: str = "Intermediate",
) -> dict:
    """
    Main entry point. Takes raw Librosa analysis dicts, returns scores.

    skill_level is accepted for backward-compatible call signatures and is
    echoed back for Claude's feedback tone — it does NOT touch the numbers.
    """
    if mode == "strict" and reference_analysis:
        pitch_acc = _pitch_accuracy_strict(user_analysis, reference_analysis)
        chord_acc = _chord_accuracy_strict(user_analysis, reference_analysis)
        timing_con = _timing_consistency(user_analysis)
    else:
        pitch_acc = _pitch_stability_creative(user_analysis)
        chord_acc = _chord_clarity_creative(user_analysis)
        timing_con = _timing_consistency(user_analysis)

    dynamic_ctrl = _dynamic_control(user_analysis)
    tonal_clarity = _tonal_clarity(user_analysis)

    # Duration penalty — short clips show less of the song. Legitimate (not
    # scaffolding); retained. NOTE: this is load-bearing for short clips — a
    # 33s clip is penalized 0.92x vs a 60s clip at 1.0x, which is part of why
    # a short gospel clip now lands below a full-length cover.
    duration = user_analysis.get("duration_seconds", 60)
    if duration < 20:
        duration_penalty = 0.80
    elif duration < 30:
        duration_penalty = 0.85
    elif duration < 45:
        duration_penalty = 0.92
    else:
        duration_penalty = 1.0

    # Technical score: weighted combination of pitch, timing, chords
    technical_raw = (
        pitch_acc * 0.40 +
        timing_con * 0.30 +
        chord_acc * 0.30
    ) * duration_penalty

    # Emotional score: dynamics + tonal quality + timing feel
    emotional_raw = (
        dynamic_ctrl * 0.45 +
        tonal_clarity * 0.30 +
        timing_con * 0.25
    ) * duration_penalty

    # Convert 0-1 raw scores to 1-10 scale with the recalibrated curve.
    # No skill-range compression is applied afterward — the curve output IS
    # the score.
    technical_score = _calibrate_to_10(technical_raw)
    emotional_score = _calibrate_to_10(emotional_raw)

    # Overall: weighted blend
    overall_score = round(technical_score * 0.55 + emotional_score * 0.45, 1)

    return {
        "overall": overall_score,
        "technical": technical_score,
        "emotional": emotional_score,
        # Sub-metrics reported RAW (0-1). No skill scaling.
        "pitch_accuracy": round(pitch_acc, 3),
        "timing_consistency": round(timing_con, 3),
        "chord_accuracy": round(chord_acc, 3),
        "dynamic_control": round(dynamic_ctrl, 3),
        "tonal_clarity": round(tonal_clarity, 3),
        "mode": mode,
        "skill_level": skill_level,  # echoed for feedback tone only
        "score_breakdown": {
            "technical_raw": round(technical_raw, 3),
            "emotional_raw": round(emotional_raw, 3),
            "duration_penalty": duration_penalty,
            "calibration": "descaffold_v1",  # track which curve produced this
            "weights": {
                "technical": {"pitch": 0.40, "timing": 0.30, "chords": 0.30},
                "emotional": {"dynamics": 0.45, "tone": 0.30, "timing_feel": 0.25},
                "overall": {"technical": 0.55, "emotional": 0.45},
            }
        }
    }


# ─── Calibration curve ───────────────────────────────────────────────

def _calibrate_to_10(raw_score: float) -> float:
    """
    Convert 0-1 raw score to 1-10 scale.

    Recalibrated for the de-scaffolded pipeline. With the skill compression
    gone, this curve is the ONLY thing shaping the distribution, so it does
    real work:
      - steepens the top (0.85-1.00) so 8+ requires genuinely strong metrics
        and a concentrated-but-unremarkable performance lands in the 6s, not
        the 8s (this is what pulls the gospel clip down instead of up)
      - spreads the middle wider than the old curve
      - makes 9+ rare

    Linear interpolation between breakpoints.
    """
    raw_score = max(0.0, min(1.0, raw_score))

    breakpoints = [
        (0.00, 1.0),
        (0.40, 2.0),
        (0.55, 3.0),
        (0.65, 3.8),
        (0.75, 4.8),
        (0.80, 5.4),
        (0.85, 6.1),
        (0.89, 6.7),
        (0.92, 7.3),
        (0.94, 7.9),
        (0.96, 8.5),
        (0.98, 9.2),
        (1.00, 9.9),
    ]

    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if raw_score <= x1:
            t = (raw_score - x0) / (x1 - x0) if x1 != x0 else 0
            return round(y0 + t * (y1 - y0), 1)

    return 9.9


# ─── Pitch scoring ───────────────────────────────────────────────────

def _pitch_accuracy_strict(user: dict, ref: dict) -> float:
    """
    Compare user's pitch distribution against reference.
    Uses pitch class distribution similarity with transposition alignment.
    """
    user_pitches = user.get("pitches_per_second", [])
    ref_pitches = ref.get("pitches_per_second", [])

    if not user_pitches or not ref_pitches:
        return 0.5

    user_notes = [p["note"] for p in user_pitches]
    ref_notes = [p["note"] for p in ref_pitches]

    user_dist = _pitch_distribution(user_notes)
    ref_dist = _pitch_distribution(ref_notes)

    direct_score = _cosine_similarity(user_dist, ref_dist)

    user_key = user.get("detected_key", "")
    ref_key = ref.get("detected_key", "")

    best_shifted_score = direct_score
    if user_key != ref_key:
        for shift in range(1, 12):
            shifted_dist = user_dist[shift:] + user_dist[:shift]
            shifted_score = _cosine_similarity(shifted_dist, ref_dist)
            if shifted_score > best_shifted_score:
                best_shifted_score = shifted_score

    distribution_score = best_shifted_score
    key_bonus = 0.05 if user_key == ref_key else 0.0
    return min(1.0, distribution_score + key_bonus)


def _pitch_stability_creative(user: dict) -> float:
    """
    Creative mode: how stable/intentional is the pitch within the performance?
    """
    pitches = user.get("pitches_per_second", [])
    if len(pitches) < 5:
        return 0.5

    note_counts = {}
    for p in pitches:
        note = p["note"]
        note_counts[note] = note_counts.get(note, 0) + 1

    total = sum(note_counts.values())
    if total == 0:
        return 0.5

    sorted_counts = sorted(note_counts.values(), reverse=True)
    top_n = min(7, len(sorted_counts))
    top_concentration = sum(sorted_counts[:top_n]) / total

    timing = user.get("timing_consistency", 0.5)
    brightness = user.get("avg_brightness", 1500)
    tone_bonus = 0.05 if 800 < brightness < 4000 else 0.0

    return min(1.0, top_concentration * 0.6 + timing * 0.35 + tone_bonus)


# ─── Chord scoring ───────────────────────────────────────────────────

def _chord_accuracy_strict(user: dict, ref: dict) -> float:
    """
    Compare user's chord/pitch distribution against reference.
    """
    user_pitches = [p["note"] for p in user.get("pitches_per_second", [])]
    ref_pitches = [p["note"] for p in ref.get("pitches_per_second", [])]

    if not user_pitches or not ref_pitches:
        return 0.5

    user_dist = _pitch_distribution(user_pitches)
    ref_dist = _pitch_distribution(ref_pitches)

    direct_score = _cosine_similarity(user_dist, ref_dist)

    user_key = user.get("detected_key", "")
    ref_key = ref.get("detected_key", "")

    best_score = direct_score
    if user_key != ref_key:
        for shift in range(1, 12):
            shifted_dist = user_dist[shift:] + user_dist[:shift]
            shifted_score = _cosine_similarity(shifted_dist, ref_dist)
            if shifted_score > best_score:
                best_score = shifted_score

    if user_key == ref_key:
        best_score = min(1.0, best_score + 0.05)

    return best_score


def _chord_clarity_creative(user: dict) -> float:
    """
    Creative mode: how clean and defined are the chord voicings?
    """
    pitches = user.get("pitches_per_second", [])
    if len(pitches) < 5:
        return 0.5

    note_counts = {}
    for p in pitches:
        note_counts[p["note"]] = note_counts.get(p["note"], 0) + 1

    total = sum(note_counts.values())
    if total == 0:
        return 0.5

    sorted_counts = sorted(note_counts.values(), reverse=True)
    top_n = min(7, len(sorted_counts))
    concentration = sum(sorted_counts[:top_n]) / total

    timing = user.get("timing_consistency", 0.5)
    brightness = user.get("avg_brightness", 1500)
    tone_score = 0.9 if 800 < brightness < 4000 else 0.5

    return min(1.0, concentration * 0.4 + timing * 0.4 + tone_score * 0.2)


# ─── Timing scoring ──────────────────────────────────────────────────

def _timing_consistency(user: dict) -> float:
    """
    How regular is the timing between onsets?
    Both modes use the same measurement.
    """
    tc = user.get("timing_consistency", None)
    if tc is not None:
        return float(tc)

    onsets = user.get("onset_timestamps", [])
    if len(onsets) < 3:
        return 0.5

    intervals = np.diff(onsets)
    if len(intervals) == 0 or np.mean(intervals) == 0:
        return 0.5

    cv = float(np.std(intervals) / np.mean(intervals))
    return max(0.0, min(1.0, 1.0 - cv))


# ─── Dynamics scoring ────────────────────────────────────────────────

def _dynamic_control(user: dict) -> float:
    """
    Score dynamic range and control.
    """
    avg_rms = user.get("avg_rms", 0)
    dynamic_range = user.get("dynamic_range", 0)

    if avg_rms < 0.005:
        rms_score = 0.2
    elif avg_rms < 0.01:
        rms_score = 0.65
    elif avg_rms < 0.3:
        rms_score = 0.9
    else:
        rms_score = 0.6

    if dynamic_range < 0.005:
        range_score = 0.3
    elif dynamic_range < 0.02:
        range_score = 0.7
    elif dynamic_range < 0.15:
        range_score = 0.9
    elif dynamic_range < 0.4:
        range_score = 0.8
    else:
        range_score = 0.4

    return rms_score * 0.4 + range_score * 0.6


# ─── Tonal clarity ───────────────────────────────────────────────────

def _tonal_clarity(user: dict) -> float:
    """
    Score tonal quality based on spectral brightness.
    """
    brightness = user.get("avg_brightness", 1500)

    if brightness < 500:
        return 0.3
    elif brightness < 800:
        return 0.5
    elif brightness < 1500:
        return 0.7
    elif brightness < 2500:
        return 0.85
    elif brightness < 4000:
        return 0.7
    elif brightness < 5000:
        return 0.5
    else:
        return 0.3


# ─── Utility functions ───────────────────────────────────────────────

def _pitch_distribution(notes: list) -> list:
    """Convert list of note names to 12-element distribution vector."""
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    counts = [0] * 12
    for note in notes:
        try:
            idx = pitch_classes.index(note)
            counts[idx] += 1
        except ValueError:
            pass
    total = sum(counts)
    if total == 0:
        return [1 / 12] * 12
    return [c / total for c in counts]


def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
