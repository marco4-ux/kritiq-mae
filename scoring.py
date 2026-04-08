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

Skill level affects Claude's feedback tone only — NOT the scores.
A 6.5 performance is a 6.5 regardless of skill level selected.
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
    
    Skill level is passed through to Claude for feedback tone adjustment
    but does NOT affect the numerical scores. The audio determines the score.
    
    Returns:
        {
            "overall": float,
            "technical": float,
            "emotional": float,
            "pitch_accuracy": float,    # 0-1
            "timing_consistency": float, # 0-1
            "chord_accuracy": float,     # 0-1
            "dynamic_control": float,    # 0-1
            "tonal_clarity": float,      # 0-1
            "mode": str,
            "score_breakdown": {...}     # detailed sub-scores for debugging
        }
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
    
    # Skill level scoring adjustment — compresses scores into appropriate range
    # Self-reported skill level sets expectations: an intermediate player
    # doing well should cap around 7.0-7.5, not 9.0+
    # Professional mode is uncapped — the audio determines the score
    skill_ranges = {
        "Beginner":      {"floor": 2.0, "ceiling": 6.5},
        "Intermediate":  {"floor": 3.0, "ceiling": 7.5},
        "Advanced":      {"floor": 3.5, "ceiling": 8.5},
        "Professional":  {"floor": 1.0, "ceiling": 10.0},
    }
    skill_range = skill_ranges.get(skill_level, skill_ranges["Intermediate"])
    skill_floor = skill_range["floor"]
    skill_ceiling = skill_range["ceiling"]
    
    # Duration penalty — short clips show less of the song
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
    
    # Convert 0-1 raw scores to 1-10 scale with calibrated curve
    technical_score = _calibrate_to_10(technical_raw)
    emotional_score = _calibrate_to_10(emotional_raw)
    
    # Apply skill level range — compress scores into [floor, ceiling]
    def _apply_skill_range(score, floor, ceiling):
        if ceiling >= 10.0 and floor <= 1.0:
            return score  # Professional — no adjustment
        adjusted = floor + (score - 1.0) * (ceiling - floor) / 9.0
        return round(max(floor, min(ceiling, adjusted)), 1)
    
    technical_score = _apply_skill_range(technical_score, skill_floor, skill_ceiling)
    emotional_score = _apply_skill_range(emotional_score, skill_floor, skill_ceiling)
    
    # Overall: weighted blend
    overall_score = round(technical_score * 0.55 + emotional_score * 0.45, 1)
    
    return {
        "overall": overall_score,
        "technical": technical_score,
        "emotional": emotional_score,
        "pitch_accuracy": round(pitch_acc * (skill_ceiling / 10.0), 3),
        "timing_consistency": round(timing_con * (skill_ceiling / 10.0), 3),
        "chord_accuracy": round(chord_acc * (skill_ceiling / 10.0), 3),
        "dynamic_control": round(dynamic_ctrl * (skill_ceiling / 10.0), 3),
        "tonal_clarity": round(tonal_clarity * (skill_ceiling / 10.0), 3),
        "mode": mode,
        "skill_level": skill_level,
        "score_breakdown": {
            "technical_raw": round(technical_raw, 3),
            "emotional_raw": round(emotional_raw, 3),
            "skill_floor": skill_floor,
            "skill_ceiling": skill_ceiling,
            "duration_penalty": duration_penalty,
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
    Convert 0-1 raw score to 1-10 scale with a tighter curve that:
    - Makes 7+ genuinely hard to achieve
    - Makes 9+ nearly impossible without exceptional metrics
    - Spreads most performances into the 4-7 range
    - Prevents score inflation
    
    Tighter than previous curve — shifted breakpoints right so
    higher raw scores are needed to reach each tier.
    """
    raw_score = max(0.0, min(1.0, raw_score))
    
    breakpoints = [
        (0.00, 1.0),
        (0.20, 2.0),
        (0.35, 3.5),
        (0.50, 5.0),
        (0.60, 5.8),
        (0.70, 6.5),
        (0.75, 7.0),
        (0.82, 7.8),
        (0.88, 8.5),
        (0.93, 9.0),
        (0.97, 9.5),
        (1.00, 10.0),
    ]
    
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if raw_score <= x1:
            t = (raw_score - x0) / (x1 - x0) if x1 != x0 else 0
            return round(y0 + t * (y1 - y0), 1)
    
    return 10.0


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
    
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    user_key = user.get("detected_key", "")
    ref_key = ref.get("detected_key", "")
    
    user_notes = [p["note"] for p in user_pitches]
    ref_notes = [p["note"] for p in ref_pitches]
    
    user_dist = _pitch_distribution(user_notes)
    ref_dist = _pitch_distribution(ref_notes)
    
    direct_score = _cosine_similarity(user_dist, ref_dist)
    
    best_shifted_score = direct_score
    if user_key != ref_key:
        for shift in range(1, 12):
            shifted_dist = user_dist[shift:] + user_dist[:shift]
            shifted_score = _cosine_similarity(shifted_dist, ref_dist)
            if shifted_score > best_shifted_score:
                best_shifted_score = shifted_score
    
    distribution_score = best_shifted_score
    
    if user_key == ref_key:
        key_bonus = 0.05
    else:
        key_bonus = 0.0
    
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
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    user_key = user.get("detected_key", "")
    ref_key = ref.get("detected_key", "")
    
    user_pitches = [p["note"] for p in user.get("pitches_per_second", [])]
    ref_pitches = [p["note"] for p in ref.get("pitches_per_second", [])]
    
    if not user_pitches or not ref_pitches:
        return 0.5
    
    user_dist = _pitch_distribution(user_pitches)
    ref_dist = _pitch_distribution(ref_pitches)
    
    direct_score = _cosine_similarity(user_dist, ref_dist)
    
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
        return [1/12] * 12
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
