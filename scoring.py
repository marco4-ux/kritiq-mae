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
"""

import numpy as np
from typing import Optional


def calculate_scores(
    user_analysis: dict,
    reference_analysis: Optional[dict] = None,
    mode: str = "creative",  # "strict" or "creative"
) -> dict:
    """
    Main entry point. Takes raw Librosa analysis dicts, returns scores.
    
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
    
    # Technical score: weighted combination of pitch, timing, chords
    technical_raw = (
        pitch_acc * 0.40 +
        timing_con * 0.30 +
        chord_acc * 0.30
    )
    
    # Emotional score: dynamics + tonal quality + timing feel
    # Good dynamics and tone = emotional expressiveness
    emotional_raw = (
        dynamic_ctrl * 0.45 +
        tonal_clarity * 0.30 +
        timing_con * 0.25  # timing also affects feel
    )
    
    # Convert 0-1 raw scores to 1-10 scale with calibrated curve
    technical_score = _calibrate_to_10(technical_raw)
    emotional_score = _calibrate_to_10(emotional_raw)
    
    # Overall: weighted blend
    overall_score = round(technical_score * 0.55 + emotional_score * 0.45, 1)
    
    return {
        "overall": overall_score,
        "technical": technical_score,
        "emotional": emotional_score,
        "pitch_accuracy": round(pitch_acc, 3),
        "timing_consistency": round(timing_con, 3),
        "chord_accuracy": round(chord_acc, 3),
        "dynamic_control": round(dynamic_ctrl, 3),
        "tonal_clarity": round(tonal_clarity, 3),
        "mode": mode,
        "score_breakdown": {
            "technical_raw": round(technical_raw, 3),
            "emotional_raw": round(emotional_raw, 3),
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
    Convert 0-1 raw score to 1-10 scale with a curve that:
    - Spreads scores across the full range (no clustering at 6-7)
    - Makes 9+ genuinely hard to achieve
    - Makes below 3 genuinely bad
    
    Uses a sigmoid-shaped mapping:
        raw 0.0 → 1.0
        raw 0.3 → 3.0  (fundamental problems)
        raw 0.5 → 5.0  (average)
        raw 0.7 → 7.0  (strong)
        raw 0.85 → 8.5 (very strong)
        raw 0.95 → 9.5 (world class)
        raw 1.0 → 10.0
    """
    raw_score = max(0.0, min(1.0, raw_score))
    
    # Piecewise linear mapping for precise control over score distribution
    breakpoints = [
        (0.00, 1.0),
        (0.15, 2.0),
        (0.30, 3.5),
        (0.45, 5.0),
        (0.55, 6.0),
        (0.65, 7.0),
        (0.75, 8.0),
        (0.80, 8.5),
        (0.85, 9.0),
        (0.92, 9.5),
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
    Uses pitch class distribution similarity with transposition alignment —
    if the performer plays in a different key, we shift the distribution 
    to match before comparing. This way intentional key changes don't 
    get penalized.
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
    
    # Direct distribution comparison (same key)
    direct_score = _cosine_similarity(user_dist, ref_dist)
    
    # Transposition-aligned comparison: shift user distribution to best match reference
    # This handles intentional key changes — the intervals/patterns should still match
    best_shifted_score = direct_score
    if user_key != ref_key:
        for shift in range(1, 12):
            # Rotate the 12-element distribution list by 'shift' positions
            shifted_dist = user_dist[shift:] + user_dist[:shift]
            shifted_score = _cosine_similarity(shifted_dist, ref_dist)
            if shifted_score > best_shifted_score:
                best_shifted_score = shifted_score
    
    # Use the best score — either direct match or best transposition alignment
    distribution_score = best_shifted_score
    
    # Key match bonus (small — same key is slightly better than transposed)
    if user_key == ref_key:
        key_bonus = 0.05
    else:
        key_bonus = 0.0
    
    return min(1.0, distribution_score + key_bonus)


def _pitch_stability_creative(user: dict) -> float:
    """
    Creative mode: how stable/intentional is the pitch within the performance?
    
    In creative mode we're measuring internal consistency, not reference matching.
    A vocal+guitar performance naturally uses 7-8 pitch classes (chord tones + melody).
    Using top 7 captures all intentional notes without penalizing harmonic richness.
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
    
    # Top 7 concentration — vocal+guitar uses many pitch classes intentionally
    top_n = min(7, len(sorted_counts))
    top_concentration = sum(sorted_counts[:top_n]) / total
    
    # Timing as strong corroboration — consistent timing = intentional pitch choices
    timing = user.get("timing_consistency", 0.5)
    
    # Tonal clarity bonus
    brightness = user.get("avg_brightness", 1500)
    tone_bonus = 0.05 if 800 < brightness < 4000 else 0.0
    
    return min(1.0, top_concentration * 0.6 + timing * 0.35 + tone_bonus)


# ─── Chord scoring ───────────────────────────────────────────────────

def _chord_accuracy_strict(user: dict, ref: dict) -> float:
    """
    Compare user's chord/pitch distribution against reference.
    Uses transposition-aligned comparison so key changes don't tank the score.
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
    
    # Direct comparison
    direct_score = _cosine_similarity(user_dist, ref_dist)
    
    # Transposition-aligned: shift user distribution to find best match
    best_score = direct_score
    if user_key != ref_key:
        for shift in range(1, 12):
            shifted_dist = user_dist[shift:] + user_dist[:shift]
            shifted_score = _cosine_similarity(shifted_dist, ref_dist)
            if shifted_score > best_score:
                best_score = shifted_score
    
    # Small bonus for same key
    if user_key == ref_key:
        best_score = min(1.0, best_score + 0.05)
    
    return best_score


def _chord_clarity_creative(user: dict) -> float:
    """
    Creative mode: how clean and defined are the chord voicings?
    
    In creative mode, we measure internal harmonic clarity.
    Strong timing + clean tone = clean chords. Use corroborating metrics
    heavily since chroma distribution is unreliable with vocals mixed in.
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
    
    # Top 7 concentration — same logic as pitch stability
    top_n = min(7, len(sorted_counts))
    concentration = sum(sorted_counts[:top_n]) / total
    
    # Timing as strong corroboration
    timing = user.get("timing_consistency", 0.5)
    
    # Tonal clarity
    brightness = user.get("avg_brightness", 1500)
    tone_score = 0.9 if 800 < brightness < 4000 else 0.5
    
    return min(1.0, concentration * 0.4 + timing * 0.4 + tone_score * 0.2)


# ─── Timing scoring ──────────────────────────────────────────────────

def _timing_consistency(user: dict) -> float:
    """
    How regular is the timing between onsets?
    Both modes use the same measurement — timing is absolute.
    """
    # Use the pre-calculated value from Librosa analysis if available
    tc = user.get("timing_consistency", None)
    if tc is not None:
        return float(tc)
    
    # Fallback: calculate from onset timestamps
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
    Good dynamics = intentional variation (not flat, not chaotic).
    Controlled, consistent dynamics at a proper level is a skill, not a flaw.
    """
    avg_rms = user.get("avg_rms", 0)
    dynamic_range = user.get("dynamic_range", 0)
    
    # RMS score: penalize only very low (inaudible) or very high (clipping)
    # A controlled, moderate level is good
    if avg_rms < 0.005:
        rms_score = 0.2  # barely audible
    elif avg_rms < 0.01:
        rms_score = 0.65  # quiet but present
    elif avg_rms < 0.3:
        rms_score = 0.9  # good controlled level
    else:
        rms_score = 0.6  # possibly clipping
    
    # Dynamic range score: controlled variation is the goal
    if dynamic_range < 0.005:
        range_score = 0.3  # completely flat / dead signal
    elif dynamic_range < 0.02:
        range_score = 0.7  # very tight — controlled but limited
    elif dynamic_range < 0.15:
        range_score = 0.9  # good expressive range
    elif dynamic_range < 0.4:
        range_score = 0.8  # wide but potentially intentional
    else:
        range_score = 0.4  # chaotic volume swings
    
    return rms_score * 0.4 + range_score * 0.6


# ─── Tonal clarity ───────────────────────────────────────────────────

def _tonal_clarity(user: dict) -> float:
    """
    Score tonal quality based on spectral brightness.
    Not too dull, not too harsh — in the sweet spot.
    """
    brightness = user.get("avg_brightness", 1500)
    
    # Spectral centroid typical ranges (22050 sr):
    # Very dull/muffled: < 800
    # Warm/balanced: 800-2500
    # Bright/present: 2500-4000
    # Harsh/tinny: > 4000
    
    if brightness < 500:
        return 0.3  # very muffled
    elif brightness < 800:
        return 0.5  # dull
    elif brightness < 1500:
        return 0.7  # warm
    elif brightness < 2500:
        return 0.85  # balanced/present (sweet spot)
    elif brightness < 4000:
        return 0.7  # bright but acceptable
    elif brightness < 5000:
        return 0.5  # getting harsh
    else:
        return 0.3  # tinny/harsh


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


# ─── Test / demo ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: score the test data from our /analyze run
    test_user = {
        "detected_key": "C#",
        "key_confidence": 0.768,
        "avg_bpm": 220,
        "timing_consistency": 0.373,
        "avg_rms": 0.0084,
        "dynamic_range": 0.0328,
        "avg_brightness": 1896.5,
        "pitches_per_second": [
            {"note": "B", "time": 0, "confidence": 1},
            {"note": "G#", "time": 1, "confidence": 1},
            {"note": "C#", "time": 2, "confidence": 1},
            {"note": "C#", "time": 3, "confidence": 1},
            {"note": "C#", "time": 4, "confidence": 1},
            {"note": "C#", "time": 5, "confidence": 1},
            {"note": "G#", "time": 7, "confidence": 1},
            {"note": "A#", "time": 8, "confidence": 1},
            {"note": "C#", "time": 9, "confidence": 1},
            {"note": "F", "time": 10, "confidence": 1},
            {"note": "G#", "time": 11, "confidence": 1},
            {"note": "D#", "time": 12, "confidence": 1},
        ],
    }
    
    test_ref = {
        "detected_key": "D",
        "key_confidence": 0.471,
        "pitches_per_second": [
            {"note": "D#", "time": 0, "confidence": 1},
            {"note": "C", "time": 1, "confidence": 1},
            {"note": "B", "time": 2, "confidence": 1},
            {"note": "A", "time": 3, "confidence": 1},
            {"note": "D", "time": 4, "confidence": 1},
            {"note": "E", "time": 5, "confidence": 1},
            {"note": "F#", "time": 6, "confidence": 1},
            {"note": "C#", "time": 7, "confidence": 1},
            {"note": "D", "time": 8, "confidence": 1},
            {"note": "D", "time": 9, "confidence": 1},
            {"note": "D", "time": 10, "confidence": 1},
            {"note": "B", "time": 11, "confidence": 1},
        ],
    }
    
    print("=== Creative Mode (no reference) ===")
    creative = calculate_scores(test_user, mode="creative")
    for k, v in creative.items():
        if k != "score_breakdown":
            print(f"  {k}: {v}")
    
    print("\n=== Strict Mode (vs Deezer reference) ===")
    strict = calculate_scores(test_user, test_ref, mode="strict")
    for k, v in strict.items():
        if k != "score_breakdown":
            print(f"  {k}: {v}")
