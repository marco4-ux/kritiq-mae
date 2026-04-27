from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import json
import time
import requests as http_requests
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
REPLICATE_MODEL = "cjwbw/demucs"
REPLICATE_VERSION = "25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

import jwt as pyjwt

# ─── Phase 4.5: Song Critique Mode mapping ───────────────────────────
# Maps the new form field to:
#   - critique_mode: the canonical mode name passed to feedback.py
#   - reference_weighting: scoring mode, derived from critique mode (no longer
#     a separate user-facing choice)
SONG_CRITIQUE_MODE_MAP = {
    "Cover Band Mode":     {"reference_weighting": "Strict",   "scoring_mode": "strict"},
    "New Cover Mode":      {"reference_weighting": "Creative", "scoring_mode": "creative"},
    "Original Track Mode": {"reference_weighting": "None",     "scoring_mode": "creative"},
}

# Legacy `style` values from Phase 4 and earlier — mapped to new modes for
# backward compatibility while the frontend transitions.
LEGACY_STYLE_TO_CRITIQUE_MODE = {
    "Original Style": "Cover Band Mode",
    "Interpretation": "New Cover Mode",
    "Original Song":  "Original Track Mode",
}

DEFAULT_CRITIQUE_MODE = "Cover Band Mode"


def resolve_song_critique_mode(form) -> str:
    """
    Read the song critique mode from the request form, with backward-compat
    fallback to the legacy `style` field.

    Returns one of: "Cover Band Mode", "New Cover Mode", "Original Track Mode".
    """
    new_value = form.get("song_critique_mode", "").strip()
    if new_value in SONG_CRITIQUE_MODE_MAP:
        return new_value

    # Backward compatibility: accept legacy `style` field
    legacy_style = form.get("style", "").strip()
    if legacy_style in SONG_CRITIQUE_MODE_MAP:
        # Legacy field is already using a new value — accept it
        return legacy_style
    if legacy_style in LEGACY_STYLE_TO_CRITIQUE_MODE:
        mapped = LEGACY_STYLE_TO_CRITIQUE_MODE[legacy_style]
        logger.info(f"Mapped legacy style '{legacy_style}' to critique mode '{mapped}'")
        return mapped

    logger.info(f"No valid song_critique_mode or style found, defaulting to {DEFAULT_CRITIQUE_MODE}")
    return DEFAULT_CRITIQUE_MODE


def parse_audio_only_flag(form) -> bool:
    """Return True if the submission is an audio-only upload (no video)."""
    val = form.get("audio_only", "false").strip().lower()
    return val in ("true", "1", "yes", "on")


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# ─── Health ───────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "MAE server running",
        "replicate_configured": bool(REPLICATE_API_TOKEN),
    })

@app.route("/warmup", methods=["GET", "POST"])
def warmup():
    """
    Pre-warm the Replicate Demucs GPU.
    """
    if not REPLICATE_API_TOKEN:
        return jsonify({"status": "error", "message": "REPLICATE_API_TOKEN not set"}), 500
    
    try:
        import base64
        import struct
        import io
        
        sample_rate = 8000
        num_samples = sample_rate
        wav_buf = io.BytesIO()
        data_size = num_samples * 2
        wav_buf.write(b'RIFF')
        wav_buf.write(struct.pack('<I', 36 + data_size))
        wav_buf.write(b'WAVE')
        wav_buf.write(b'fmt ')
        wav_buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        wav_buf.write(b'data')
        wav_buf.write(struct.pack('<I', data_size))
        wav_buf.write(b'\x00' * data_size)
        
        audio_data = base64.b64encode(wav_buf.getvalue()).decode("utf-8")
        audio_url = f"data:audio/wav;base64,{audio_data}"
        
        headers = {
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "version": REPLICATE_VERSION,
            "input": {
                "audio": audio_url,
                "model_name": "htdemucs",
            }
        }
        
        resp = http_requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        prediction = resp.json()
        
        return jsonify({
            "status": "ok",
            "message": "Warmup request sent - GPU will be ready in ~30-60s",
            "prediction_id": prediction.get("id"),
            "prediction_status": prediction.get("status"),
        })
    
    except Exception as e:
        logger.exception("Warmup failed")
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── Helper: Extract audio from video via FFmpeg ─────────────────────

def extract_audio(input_path, sr=44100, mono=True, max_duration=None):
    """Extract audio from video/audio file, return path to WAV.
    Works on both video and audio inputs — FFmpeg transcodes either to WAV.
    """
    wav_path = input_path + ".wav"
    cmd = ["ffmpeg", "-i", input_path, "-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    if max_duration:
        cmd += ["-t", str(max_duration)]
    cmd += ["-y", wav_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[:500]}")
    return wav_path

# ─── Helper: Upload file to tmpfiles.org for Replicate input ────────

def upload_for_replicate(file_path):
    """Upload a file and return a public URL for Replicate to consume."""
    import base64
    
    file_size = os.path.getsize(file_path)
    
    if file_size < 20 * 1024 * 1024:
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:audio/wav;base64,{data}"
    
    raise RuntimeError(f"File too large for data URI ({file_size} bytes). Need external hosting.")

# ─── Helper: Deezer reference lookup + analysis ──────────────────────

def fetch_deezer_reference(song_title: str, song_artist: str) -> dict:
    """
    Search Deezer for a song, download the 30s preview, run Librosa on it.
    """
    query = f"{song_title} {song_artist}".strip()
    if not query:
        return None
    
    try:
        logger.info(f"Deezer reference lookup: {query}")
        search_resp = http_requests.get(
            "https://api.deezer.com/search",
            params={"q": query},
            timeout=15,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
        
        if not search_data.get("data"):
            logger.info(f"No Deezer results for: {query}")
            return None
        
        track = search_data["data"][0]
        preview_url = track.get("preview")
        if not preview_url:
            logger.info(f"Deezer track found but no preview URL")
            return None
        
        track_info = {
            "title": track.get("title"),
            "artist": track.get("artist", {}).get("name"),
            "album": track.get("album", {}).get("title"),
            "deezer_id": track.get("id"),
            "duration": track.get("duration"),
            "preview_url": preview_url,
        }
        
        dl_resp = http_requests.get(preview_url, timeout=30)
        dl_resp.raise_for_status()
        
        preview_path = tempfile.mktemp(suffix=".mp3")
        with open(preview_path, "wb") as f:
            f.write(dl_resp.content)
        
        wav_path = preview_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-i", preview_path, "-ar", "22050", "-ac", "1", "-y", wav_path],
            capture_output=True, text=True, timeout=30,
        )
        
        ref_analysis = analyze_stem(wav_path, lightweight=True)
        
        if os.path.exists(preview_path):
            os.unlink(preview_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        
        logger.info(f"Deezer reference ready: {track_info['title']} by {track_info['artist']} - key: {ref_analysis.get('detected_key')}")
        
        return {
            "track": track_info,
            "analysis": ref_analysis,
        }
    
    except Exception as e:
        logger.warning(f"Deezer reference lookup failed: {e}")
        return None

# ─── Helper: Supabase operations ─────────────────────────────────────

def _supabase_headers():
    return {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

def get_cached_reference(song_title: str, song_artist: str) -> dict:
    """Look up cached song reference from Supabase."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    
    try:
        resp = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/songs",
            headers=_supabase_headers(),
            params={
                "title": f"eq.{song_title}",
                "artist": f"eq.{song_artist}",
                "select": "id,reference_analysis,deezer_id",
                "limit": "1",
            },
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        
        if rows and rows[0].get("reference_analysis"):
            logger.info(f"Cache HIT: {song_title} by {song_artist}")
            return rows[0]["reference_analysis"]
        
        logger.info(f"Cache MISS: {song_title} by {song_artist}")
        return None
    except Exception as e:
        logger.warning(f"Supabase cache lookup failed: {e}")
        return None

def cache_song_reference(track_info: dict, reference_analysis: dict) -> None:
    """Save song reference to Supabase for future lookups."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    
    try:
        from datetime import datetime, timezone
        
        payload = {
            "title": track_info.get("title", ""),
            "artist": track_info.get("artist", ""),
            "deezer_id": track_info.get("deezer_id"),
            "deezer_preview_url": track_info.get("preview_url"),
            "album": track_info.get("album"),
            "duration_seconds": track_info.get("duration"),
            "reference_analysis": reference_analysis,
            "reference_source": "deezer",
            "reference_analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
        
        resp = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/songs",
            headers={**_supabase_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
            json=payload,
            timeout=10,
        )
        
        if resp.status_code >= 400:
            logger.warning(f"Supabase cache save HTTP {resp.status_code}: {resp.text[:500]}")
        else:
            logger.info(f"Cached reference: {track_info.get('title')} by {track_info.get('artist')}")
    except Exception as e:
        logger.warning(f"Supabase cache save failed: {e}")

def upsert_song(song_title: str, song_artist: str) -> str:
    """Ensure a song exists in the `songs` table and return its UUID."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or not song_title:
        return None
    
    try:
        resp = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/songs",
            headers=_supabase_headers(),
            params={
                "title": f"eq.{song_title}",
                "artist": f"eq.{song_artist}",
                "select": "id",
                "limit": "1",
            },
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            return rows[0].get("id")
        
        payload = {
            "title": song_title,
            "artist": song_artist,
        }
        resp = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/songs",
            headers={**_supabase_headers(), "Prefer": "resolution=merge-duplicates,return=representation"},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            return rows[0].get("id")
        return None
    except Exception as e:
        logger.warning(f"Supabase song upsert failed: {e}")
        return None


def save_submission(data: dict) -> str:
    """Save a submission record to Supabase. Returns an ID or None.
    
    Always writes to `performances` (public ledger).
    If data contains a user_id, also writes to `submissions` (user's tracked library).
    Returns the submissions.id when authenticated, else the performances.id.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    
    user_id = data.pop("user_id", None)
    submitter_name = data.pop("submitter_name", "Anonymous")
    song_title = data.pop("song_title", "")
    song_artist = data.pop("song_artist", "")
    
    # Write to performances (always, anonymous-compatible)
    performance_id = None
    try:
        # Bundle scores + visual_analysis into feedback for the public archive
        # so the modal can render the full breakdown without a join.
        bundled_feedback = dict(data.get("feedback") or {})
        if data.get("scores"):
            bundled_feedback["scores"] = data.get("scores")
        if data.get("visual_analysis"):
            bundled_feedback["visual_analysis"] = data.get("visual_analysis")
        if data.get("performance_analysis"):
            bundled_feedback["analysis"] = data.get("performance_analysis")

        performance_payload = {
            "song_title": song_title,
            "artist_name": song_artist,
            "feedback": bundled_feedback,
            "overall_score": data.get("scores", {}).get("overall") if data.get("scores") else None,
            "processed": True,
            "submitter_name": submitter_name,
        }
        resp = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/performances",
            headers=_supabase_headers(),
            json=performance_payload,
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            performance_id = rows[0].get("id")
    except Exception as e:
        logger.warning(f"Supabase performances save failed: {e}")
    
    # Write to submissions only if authenticated
    if user_id:
        song_id = upsert_song(song_title, song_artist)
        if not song_id:
            logger.warning("Could not resolve song_id for authenticated submission, skipping submissions write")
            return performance_id
        
        try:
            # NOTE: column names retained for back-compat with existing 193 rows.
            #   - DB column `style` holds the song_critique_mode value.
            #   - DB column `intentional_choices` holds the creative_choices value.
            #   - DB column `harshness` is hardcoded to "Unified" post-Phase 4.5.
            #   - DB column `influence` is no longer populated (UI removed).
            submission_payload = {
                "user_id": user_id,
                "song_id": song_id,
                "skill_level": data.get("skill_level"),
                "harshness": data.get("harshness", "Unified"),
                "style": data.get("song_critique_mode") or data.get("style"),
                "genre": data.get("genre"),
                "environment": data.get("environment"),
                "intentional_choices": data.get("creative_choices") or data.get("intentional_choices"),
                "influence": None,
                "reference_weighting": data.get("reference_weighting"),
                "stems": data.get("stems"),
                "performance_analysis": data.get("performance_analysis"),
                "visual_analysis": data.get("visual_analysis"),
                "scores": data.get("scores"),
                "feedback": data.get("feedback"),
                "pipeline_timing": data.get("pipeline_timing"),
                "status": data.get("status", "completed"),
                "completed_at": data.get("completed_at"),
            }
            resp = http_requests.post(
                f"{SUPABASE_URL}/rest/v1/submissions",
                headers=_supabase_headers(),
                json=submission_payload,
                timeout=10,
            )
            resp.raise_for_status()
            rows = resp.json()
            if rows:
                return rows[0].get("id")
        except Exception as e:
            logger.warning(f"Supabase submissions save failed: {e}")
    
    return performance_id

from jwt import PyJWKClient

_jwks_client = None

def _get_jwks_client():
    global _jwks_client
    if _jwks_client is None and SUPABASE_URL:
        _jwks_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True)
    return _jwks_client


def verify_supabase_jwt(auth_header: str) -> tuple:
    """Returns (user_id, email, error). email may be None if not in token."""
    if not auth_header:
        return (None, None, None)

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return (None, None, "Invalid Authorization header format")

    token = parts[1]

    try:
        alg = pyjwt.get_unverified_header(token).get("alg", "").upper()

        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                return (None, None, "SUPABASE_JWT_SECRET not configured")
            payload = pyjwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        elif alg in ("RS256", "ES256"):
            jwks_client = _get_jwks_client()
            if not jwks_client:
                return (None, None, "JWKS client could not be initialized")
            signing_key = jwks_client.get_signing_key_from_jwt(token).key
            payload = pyjwt.decode(token, signing_key, algorithms=[alg], options={"verify_aud": False})
        else:
            return (None, None, f"Unsupported signing algorithm: {alg}")

        user_id = payload.get("sub")
        if not user_id:
            return (None, None, "JWT missing sub claim")
        email = payload.get("email")
        return (user_id, email, None)

    except pyjwt.ExpiredSignatureError:
        return (None, None, "Token expired")
    except pyjwt.InvalidTokenError as e:
        return (None, None, f"Invalid token: {str(e)}")

# ─── Helper: Call Replicate Demucs API ───────────────────────────────

def separate_stems_replicate(audio_url):
    """Send audio to Replicate Demucs, return dict of stem URLs."""
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }
    
    payload = {
        "version": REPLICATE_VERSION,
        "input": {
            "audio": audio_url,
            "model_name": "htdemucs",
        }
    }
    
    logger.info("Sending to Replicate Demucs (sync mode)...")
    resp = http_requests.post(
        "https://api.replicate.com/v1/predictions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    
    if result.get("status") not in ("succeeded", "failed", "canceled"):
        result = _poll_prediction(result["id"])
    
    if result.get("status") == "failed":
        raise RuntimeError(f"Replicate Demucs failed: {result.get('error')}")
    
    output = result.get("output")
    predict_time = result.get("metrics", {}).get("predict_time", 0)
    logger.info(f"Demucs completed in {predict_time:.1f}s")
    
    return output, predict_time

def _poll_prediction(prediction_id, max_wait=300, interval=5):
    """Poll a Replicate prediction until terminal state."""
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}
    url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    
    start = time.time()
    while time.time() - start < max_wait:
        resp = http_requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        status = result.get("status")
        if status in ("succeeded", "failed", "canceled"):
            return result
        logger.info(f"Polling Demucs... status={status}")
        time.sleep(interval)
    
    raise RuntimeError(f"Replicate prediction timed out after {max_wait}s")

# ─── Helper: Download a stem from Replicate URL ─────────────────────

def download_stem(url, output_path):
    """Download a stem audio file from Replicate output URL."""
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}
    resp = http_requests.get(url, headers=headers, timeout=60, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return output_path

# ─── Helper: PYIN pitch analysis ─────────────────────────────────────

def _pyin_analysis(y, sr):
    """
    Run PYIN on the audio signal to extract f0 contour, pitch stability,
    drift, off-pitch segments, vibrato, and voiced ratio.
    """
    import librosa
    import numpy as np
    
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
    except Exception as e:
        logger.warning(f"PYIN failed: {e}")
        return _empty_pitch_analysis()
    
    hop_length = 512
    frames_per_second = sr / hop_length
    
    CONF_THRESHOLD = 0.7
    
    reliable_mask = np.array([
        bool(vf) and vp >= CONF_THRESHOLD and f is not None and not np.isnan(f)
        for vf, vp, f in zip(voiced_flag, voiced_prob, f0)
    ])
    voiced_ratio = float(np.mean(reliable_mask)) if len(reliable_mask) > 0 else 0.0
    
    if voiced_ratio < 0.05:
        return _empty_pitch_analysis(voiced_ratio=voiced_ratio)
    
    contour = []
    step = max(1, int(frames_per_second / 10))
    for i in range(0, len(f0), step):
        hz = f0[i]
        conf = float(voiced_prob[i])
        voiced = bool(voiced_flag[i])
        time_s = i / frames_per_second
        if hz is not None and not np.isnan(hz) and voiced and conf >= CONF_THRESHOLD:
            contour.append({
                "time": round(float(time_s), 2),
                "hz": round(float(hz), 1),
                "confidence": round(conf, 3),
                "voiced": True,
            })
    
    reliable_f0 = np.array([f for f, m in zip(f0, reliable_mask) if m])
    reliable_times = np.array([
        i / frames_per_second for i, m in enumerate(reliable_mask) if m
    ])
    
    stability_score = 0.5
    if len(reliable_f0) > 10:
        cents = 1200 * np.log2(reliable_f0 / 440.0)
        diffs = np.abs(np.diff(cents))
        note_transitions = diffs > 200
        jitter_diffs = diffs[~note_transitions]
        if len(jitter_diffs) > 0:
            mean_jitter = float(np.mean(jitter_diffs))
            stability_score = max(0.0, min(1.0, 1.0 - mean_jitter / 50.0))
    
    drift_cents = 0.0
    if len(reliable_f0) > 0:
        cents = 1200 * np.log2(reliable_f0 / 440.0)
        nearest_semi = np.round(cents / 100.0) * 100.0
        deviations = cents - nearest_semi
        drift_cents = round(float(np.mean(deviations)), 1)
    
    off_pitch_segments = _detect_off_pitch_segments(
        f0, voiced_flag, voiced_prob,
        frames_per_second=frames_per_second,
        deviation_threshold_cents=25,
        min_duration_s=0.2,
        conf_threshold=CONF_THRESHOLD,
    )
    
    vibrato = _detect_vibrato(
        f0, voiced_flag, voiced_prob,
        frames_per_second=frames_per_second,
        conf_threshold=CONF_THRESHOLD,
    )
    
    return {
        "f0_contour": contour,
        "voiced_ratio": round(voiced_ratio, 3),
        "stability_score": round(stability_score, 3),
        "drift_cents": drift_cents,
        "off_pitch_segments": off_pitch_segments,
        "vibrato": vibrato,
    }


def _empty_pitch_analysis(voiced_ratio=0.0):
    """Return an empty pitch_analysis dict when PYIN can't get a clean signal."""
    return {
        "f0_contour": [],
        "voiced_ratio": round(float(voiced_ratio), 3),
        "stability_score": None,
        "drift_cents": None,
        "off_pitch_segments": [],
        "vibrato": {
            "detected": False,
            "rate_hz": None,
            "extent_cents": None,
            "segments": [],
        },
    }


def _detect_off_pitch_segments(f0, voiced_flag, voiced_prob, frames_per_second,
                                deviation_threshold_cents=25, min_duration_s=0.2,
                                conf_threshold=0.7):
    """Find continuous segments where the performer sustained a note off-pitch."""
    import numpy as np
    
    min_frames = int(min_duration_s * frames_per_second)
    segments = []
    current_start = None
    current_devs = []
    current_confs = []
    
    for i, (hz, vf, vp) in enumerate(zip(f0, voiced_flag, voiced_prob)):
        if hz is None or np.isnan(hz) or not vf or vp < conf_threshold:
            if current_start is not None and len(current_devs) >= min_frames:
                _flush_segment(segments, current_start, current_devs, current_confs, frames_per_second)
            current_start = None
            current_devs = []
            current_confs = []
            continue
        
        cents = 1200 * np.log2(hz / 440.0)
        nearest_semi = round(cents / 100.0) * 100.0
        deviation = cents - nearest_semi
        
        if abs(deviation) >= deviation_threshold_cents:
            if current_start is None:
                current_start = i
            current_devs.append(deviation)
            current_confs.append(float(vp))
        else:
            if current_start is not None and len(current_devs) >= min_frames:
                _flush_segment(segments, current_start, current_devs, current_confs, frames_per_second)
            current_start = None
            current_devs = []
            current_confs = []
    
    if current_start is not None and len(current_devs) >= min_frames:
        _flush_segment(segments, current_start, current_devs, current_confs, frames_per_second)
    
    return segments[:15]


def _flush_segment(segments, start_frame, devs, confs, frames_per_second):
    """Append a segment entry from the accumulated deviation data."""
    import numpy as np
    start_time = start_frame / frames_per_second
    duration = len(devs) / frames_per_second
    mean_dev = float(np.mean(devs))
    mean_conf = float(np.mean(confs))
    segments.append({
        "time": round(start_time, 2),
        "duration": round(duration, 2),
        "deviation_cents": round(mean_dev, 1),
        "confidence": round(mean_conf, 3),
    })


def _detect_vibrato(f0, voiced_flag, voiced_prob, frames_per_second,
                     conf_threshold=0.7, min_sustained_s=0.4):
    """Detect vibrato in sustained notes."""
    import numpy as np
    
    min_frames = int(min_sustained_s * frames_per_second)
    sustained_regions = []
    current_start = None
    
    for i, (hz, vf, vp) in enumerate(zip(f0, voiced_flag, voiced_prob)):
        if hz is not None and not np.isnan(hz) and vf and vp >= conf_threshold:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None and (i - current_start) >= min_frames:
                sustained_regions.append((current_start, i))
            current_start = None
    
    if current_start is not None and (len(f0) - current_start) >= min_frames:
        sustained_regions.append((current_start, len(f0)))
    
    if not sustained_regions:
        return {
            "detected": False,
            "rate_hz": None,
            "extent_cents": None,
            "segments": [],
        }
    
    segments = []
    rates = []
    extents = []
    
    for start, end in sustained_regions:
        region_f0 = np.array([f0[i] for i in range(start, end) if f0[i] is not None and not np.isnan(f0[i])])
        if len(region_f0) < min_frames:
            continue
        
        mean_hz = np.mean(region_f0)
        if mean_hz <= 0:
            continue
        cents = 1200 * np.log2(region_f0 / mean_hz)
        
        cents_detrended = cents - np.mean(cents)
        
        extent_cents = float((np.max(cents_detrended) - np.min(cents_detrended)) / 2.0)
        
        if extent_cents < 10:
            continue
        
        rate_hz = _estimate_oscillation_rate(cents_detrended, frames_per_second)
        
        if rate_hz is None or rate_hz < 3.0 or rate_hz > 9.0:
            continue
        
        segment_start_time = start / frames_per_second
        segment_duration = (end - start) / frames_per_second
        
        segments.append({
            "time": round(float(segment_start_time), 2),
            "duration": round(float(segment_duration), 2),
            "rate_hz": round(float(rate_hz), 2),
            "extent_cents": round(extent_cents, 1),
        })
        rates.append(rate_hz)
        extents.append(extent_cents)
    
    if not segments:
        return {
            "detected": False,
            "rate_hz": None,
            "extent_cents": None,
            "segments": [],
        }
    
    return {
        "detected": True,
        "rate_hz": round(float(np.mean(rates)), 2),
        "extent_cents": round(float(np.mean(extents)), 1),
        "segments": segments[:10],
    }


def _estimate_oscillation_rate(signal, frames_per_second):
    """Estimate dominant oscillation rate via autocorrelation."""
    import numpy as np
    
    if len(signal) < 20:
        return None
    
    sig = signal - np.mean(signal)
    if np.std(sig) == 0:
        return None
    
    acf = np.correlate(sig, sig, mode='full')
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0] if acf[0] != 0 else acf
    
    min_lag = int(frames_per_second / 9.0)
    max_lag = int(frames_per_second / 3.0)
    
    if max_lag >= len(acf):
        max_lag = len(acf) - 1
    if min_lag >= max_lag:
        return None
    
    search_region = acf[min_lag:max_lag]
    if len(search_region) == 0:
        return None
    
    peak_lag = int(np.argmax(search_region)) + min_lag
    peak_val = float(acf[peak_lag])
    
    if peak_val < 0.3:
        return None
    
    return frames_per_second / peak_lag


# ─── Helper: Librosa analysis on a stem ──────────────────────────────

def analyze_stem(wav_path, lightweight=False):
    """Run Librosa analysis on an audio file. Returns structured metrics."""
    import librosa
    import numpy as np
    
    y, sr = librosa.load(wav_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    hop_length = 512
    frames_per_second = sr / hop_length
    
    pitches_per_second = []
    for i in range(0, chroma.shape[1], max(1, int(frames_per_second))):
        dominant_idx = int(np.argmax(chroma[:, i]))
        timestamp = i / frames_per_second
        pitches_per_second.append({
            "time": round(float(timestamp), 2),
            "note": pitch_classes[dominant_idx],
            "confidence": round(float(chroma[dominant_idx, i]), 3),
        })
    
    chroma_mean = np.mean(chroma, axis=1)
    detected_key = pitch_classes[int(np.argmax(chroma_mean))]
    key_confidence = round(float(np.max(chroma_mean)), 3)
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    avg_bpm = round(float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0]), 1)
    
    if len(beat_frames) > 1:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        beat_intervals = np.diff(beat_times)
        timing_consistency = round(1.0 - min(float(np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)), 1.0), 3)
    elif len(onset_times) > 1:
        intervals = np.diff(onset_times)
        timing_consistency = round(1.0 - min(float(np.std(intervals) / (np.mean(intervals) + 1e-6)), 1.0), 3)
    else:
        timing_consistency = 0.0
    
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    avg_rms = round(float(np.mean(rms)), 4)
    dynamic_range = round(float(np.max(rms) - np.min(rms)), 4)
    
    if avg_bpm > 150 and avg_rms < 0.02:
        avg_bpm = round(avg_bpm / 2, 1)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    avg_brightness = round(float(np.mean(spectral_centroid)), 1)
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
    avg_bandwidth = float(np.mean(spectral_bandwidth))
    
    if len(onset_frames) > 0:
        onset_strengths = onset_env[onset_frames] if len(onset_frames) <= len(onset_env) else onset_env[:len(onset_frames)]
        avg_onset_strength = float(np.mean(onset_strengths)) if len(onset_strengths) > 0 else 0
    else:
        avg_onset_strength = 0
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
    avg_rolloff = float(np.mean(spectral_rolloff))
    
    bandwidth_threshold = 1800
    strength_threshold = 1.5
    
    if avg_bandwidth > bandwidth_threshold:
        if avg_onset_strength > strength_threshold:
            technique = "strumming (pick)"
        else:
            technique = "strumming (fingers)"
    else:
        if avg_onset_strength > strength_threshold:
            technique = "single notes (pick)"
        else:
            technique = "fingerpicking"
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    vocal_energy = np.mean(mel_db[5:40, :], axis=0)
    instrument_energy = np.mean(mel_db[40:, :], axis=0)
    
    vocal_ratio = float(np.mean(vocal_energy)) / (float(np.mean(instrument_energy)) + 1e-6)
    has_vocals = vocal_ratio > 0.8
    
    if has_vocals:
        if len(vocal_energy) > 1 and len(instrument_energy) > 1:
            min_len = min(len(vocal_energy), len(instrument_energy))
            correlation = float(np.corrcoef(vocal_energy[:min_len], instrument_energy[:min_len])[0, 1])
            coordination_score = round(max(0, min(1, (correlation + 1) / 2)), 3)
        else:
            coordination_score = 0.5
    else:
        coordination_score = None
    
    # --- PYIN pitch analysis ---
    if lightweight:
        pitch_analysis = _empty_pitch_analysis()
    else:
        pitch_analysis = _pyin_analysis(y, sr)
    
    return {
        "duration_seconds": round(duration, 2),
        "detected_key": detected_key,
        "key_confidence": key_confidence,
        "avg_bpm": avg_bpm,
        "timing_consistency": timing_consistency,
        "onset_count": len(onset_times),
        "onset_timestamps": [round(float(t), 2) for t in onset_times[:50]],
        "avg_rms": avg_rms,
        "dynamic_range": dynamic_range,
        "avg_brightness": avg_brightness,
        "pitches_per_second": pitches_per_second[:60],
        "technique": technique,
        "technique_details": {
            "avg_spectral_bandwidth": round(avg_bandwidth, 1),
            "avg_onset_strength": round(avg_onset_strength, 3),
            "avg_spectral_rolloff": round(avg_rolloff, 1),
        },
        "has_vocals": has_vocals,
        "coordination_score": coordination_score,
        "pitch_analysis": pitch_analysis,
    }


# ─── Endpoints ────────────────────────────────────────────────────────

@app.route("/test-ffmpeg", methods=["POST", "OPTIONS"])
def test_ffmpeg():
    if request.method == "OPTIONS":
        return '', 200
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    try:
        wav_path = extract_audio(input_path)
        file_size = os.path.getsize(wav_path)
        return jsonify({"status": "ok", "message": "FFmpeg working", "output_size_bytes": file_size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(input_path): os.unlink(input_path)
        wav = input_path + ".wav"
        if os.path.exists(wav): os.unlink(wav)


@app.route("/test-replicate", methods=["POST", "OPTIONS"])
def test_replicate():
    """Test Replicate Demucs integration end-to-end."""
    if request.method == "OPTIONS":
        return '', 200
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    try:
        wav_path = extract_audio(input_path, sr=44100, mono=False)
        audio_url = upload_for_replicate(wav_path)
        output, predict_time = separate_stems_replicate(audio_url)
        
        return jsonify({
            "status": "ok",
            "message": "Replicate Demucs working",
            "predict_time_seconds": predict_time,
            "output": output,
        })
    except Exception as e:
        logger.exception("test-replicate failed")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(input_path): os.unlink(input_path)
        wav = input_path + ".wav"
        if os.path.exists(wav): os.unlink(wav)


@app.route("/test-pitch", methods=["POST", "OPTIONS"])
def test_pitch():
    """Test Librosa analysis on raw audio (no stem separation)."""
    if request.method == "OPTIONS":
        return '', 200
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    try:
        wav_path = extract_audio(input_path, sr=22050, mono=True)
        metrics = analyze_stem(wav_path)
        return jsonify({"status": "ok", "message": "Pitch analysis complete", **metrics})
    except Exception as e:
        logger.exception("test-pitch failed")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(input_path): os.unlink(input_path)
        wav = input_path + ".wav"
        if os.path.exists(wav): os.unlink(wav)


@app.route("/test-ytdlp", methods=["GET", "POST", "OPTIONS"])
def test_ytdlp():
    """Test if yt-dlp can download audio from YouTube on Railway."""
    if request.method == "OPTIONS":
        return '', 200
    
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("query", "Hey There Delilah Plain White T's")
    else:
        query = request.args.get("query", "Hey There Delilah Plain White T's")
    
    output_path = f"/tmp/ytdlp_test_{int(time.time())}"
    
    try:
        t0 = time.time()
        
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", f"{output_path}.%(ext)s",
            "--no-playlist",
            "--match-filter", "duration < 600",
            "--postprocessor-args", "ffmpeg:-t 30",
            f"ytsearch1:{query}",
        ]
        
        logger.info(f"Running yt-dlp with query: {query}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        t1 = time.time()
        
        wav_path = f"{output_path}.wav"
        if not os.path.exists(wav_path):
            import glob
            matches = glob.glob(f"{output_path}.*")
            return jsonify({
                "status": "error",
                "message": "yt-dlp did not produce expected output",
                "stdout": result.stdout[-1000:] if result.stdout else "",
                "stderr": result.stderr[-1000:] if result.stderr else "",
                "returncode": result.returncode,
                "files_found": matches,
                "elapsed_seconds": round(t1 - t0, 2),
            }), 500
        
        file_size = os.path.getsize(wav_path)
        
        import librosa
        import numpy as np
        
        y, sr = librosa.load(wav_path, sr=22050, duration=30)
        duration = librosa.get_duration(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chroma_mean = np.mean(chroma, axis=1)
        detected_key = pitch_classes[int(np.argmax(chroma_mean))]
        
        return jsonify({
            "status": "ok",
            "message": "yt-dlp download and analysis working",
            "query": query,
            "file_size_bytes": file_size,
            "duration_seconds": round(duration, 2),
            "detected_key": detected_key,
            "elapsed_seconds": round(t1 - t0, 2),
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
        })
    
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "yt-dlp timed out after 120s"}), 500
    except Exception as e:
        logger.exception("test-ytdlp failed")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        import glob
        for f in glob.glob(f"{output_path}*"):
            try: os.unlink(f)
            except: pass


@app.route("/test-deezer", methods=["GET", "POST", "OPTIONS"])
def test_deezer():
    """Test Deezer API: search for a song, download 30s preview, run Librosa."""
    if request.method == "OPTIONS":
        return '', 200
    
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("query", "Hey There Delilah Plain White T's")
    else:
        query = request.args.get("query", "Hey There Delilah Plain White T's")
    
    preview_path = None
    wav_path = None
    
    try:
        t0 = time.time()
        
        logger.info(f"Searching Deezer for: {query}")
        search_resp = http_requests.get(
            "https://api.deezer.com/search",
            params={"q": query},
            timeout=15,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
        
        if not search_data.get("data"):
            return jsonify({"status": "error", "message": "No results found on Deezer", "query": query}), 404
        
        track = search_data["data"][0]
        track_info = {
            "title": track.get("title"),
            "artist": track.get("artist", {}).get("name"),
            "album": track.get("album", {}).get("title"),
            "duration": track.get("duration"),
            "deezer_id": track.get("id"),
            "preview_url": track.get("preview"),
        }
        
        preview_url = track.get("preview")
        if not preview_url:
            return jsonify({
                "status": "error",
                "message": "Track found but no preview URL available",
                "track": track_info,
            }), 404
        
        t1 = time.time()
        
        logger.info(f"Downloading preview: {preview_url}")
        preview_path = f"/tmp/deezer_preview_{int(time.time())}.mp3"
        dl_resp = http_requests.get(preview_url, timeout=30)
        dl_resp.raise_for_status()
        with open(preview_path, "wb") as f:
            f.write(dl_resp.content)
        
        preview_size = os.path.getsize(preview_path)
        t2 = time.time()
        
        wav_path = preview_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-i", preview_path, "-ar", "22050", "-ac", "1", "-y", wav_path],
            capture_output=True, text=True, timeout=30,
        )
        
        metrics = analyze_stem(wav_path)
        t3 = time.time()
        
        return jsonify({
            "status": "ok",
            "message": "Deezer preview download and analysis working",
            "track": track_info,
            "preview_size_bytes": preview_size,
            "timing": {
                "search_seconds": round(t1 - t0, 2),
                "download_seconds": round(t2 - t1, 2),
                "analysis_seconds": round(t3 - t2, 2),
                "total_seconds": round(t3 - t0, 2),
            },
            "analysis": metrics,
        })
    
    except Exception as e:
        logger.exception("test-deezer failed")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if preview_path and os.path.exists(preview_path):
            os.unlink(preview_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

# ─── Song Search (Deezer proxy for frontend autocomplete) ────────

@app.route("/search-songs", methods=["GET", "OPTIONS"])
def search_songs():
    """Proxy Deezer search for frontend autocomplete (avoids CORS)."""
    if request.method == "OPTIONS":
        return '', 200
    
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify({"results": []})
    
    try:
        resp = http_requests.get(
            "https://api.deezer.com/search",
            params={"q": query, "limit": 5},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        seen = set()
        for track in data.get("data", []):
            title = track.get("title", "")
            artist = track.get("artist", {}).get("name", "")
            key = f"{title.lower()}|{artist.lower()}"
            if key not in seen:
                seen.add(key)
                results.append({
                    "title": title,
                    "artist": artist,
                    "id": track.get("id"),
                })
        
        return jsonify({"results": results})
    except Exception as e:
        logger.warning(f"Song search failed: {e}")
        return jsonify({"results": []})

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    """
    Phase 4.5 MAE pipeline:
    1. Resolve song_critique_mode (with backward-compat fallback to legacy `style`)
    2. Derive scoring mode from critique mode
    3. Upload media → FFmpeg extracts audio (works for video or audio-only)
    4. Solo-performance Librosa analysis (Demucs path retained but inactive)
    5. Reference comparison (Cover Band / New Cover only — skipped for Original Track)
    6. Scoring + Whisper (vocals only) + Visual analysis (skipped for audio-only)
    7. Claude feedback (single unified persona, scaled by skill_level)
    8. Save submission

    Auth behavior unchanged from Phase 4:
    - No Authorization header: anonymous, writes to `performances` only
    - Valid JWT: authenticated, writes to both `performances` and `submissions`
    - Invalid JWT: 401 (no silent fallback to anonymous)
    """
    if request.method == "OPTIONS":
        return '', 200
    
    # Verify JWT if provided (optional auth)
    auth_header = request.headers.get("Authorization", "")
    user_id, user_email, auth_error = verify_supabase_jwt(auth_header)
    if auth_header and auth_error:
        return jsonify({"error": f"Authentication failed: {auth_error}"}), 401
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # ─── Phase 4.5: resolve mode + audio-only flag up front ─────────
    song_critique_mode = resolve_song_critique_mode(request.form)
    mode_config = SONG_CRITIQUE_MODE_MAP[song_critique_mode]
    ref_mode = mode_config["scoring_mode"]            # "strict" or "creative" — passed to scoring.py
    reference_weighting_db = mode_config["reference_weighting"]  # "Strict", "Creative", or "None" — written to DB

    audio_only = parse_audio_only_flag(request.form)

    logger.info(
        f"Submission: critique_mode={song_critique_mode}, "
        f"scoring_mode={ref_mode}, audio_only={audio_only}, user_id={user_id}"
    )
    
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".upload", delete=False) as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    temp_files = [input_path]
    
    try:
        t0 = time.time()
        
        logger.info("Step 1: Extracting audio with FFmpeg (max 60s)...")
        wav_path = extract_audio(input_path, sr=44100, mono=False, max_duration=60)
        temp_files.append(wav_path)
        t1 = time.time()
        
        solo_performance = request.form.get("solo_performance", "true").lower() in ("true", "1", "yes")
        
        vocals_url = None
        bass_url = None
        drums_url = None
        other_url = None
        guitar_url = None
        demucs_time = 0
        
        if solo_performance:
            logger.info("Step 2: Solo performance - skipping Demucs, analyzing raw audio...")
            
            analysis_wav = wav_path + ".analysis.wav"
            temp_files.append(analysis_wav)
            subprocess.run(
                ["ffmpeg", "-i", wav_path, "-ar", "22050", "-ac", "1", "-y", analysis_wav],
                capture_output=True, text=True, timeout=30
            )
            t2 = time.time()
        else:
            logger.info("Step 2: Uploading to Replicate Demucs...")
            audio_url = upload_for_replicate(wav_path)
            stems_output, demucs_time = separate_stems_replicate(audio_url)
            t2 = time.time()
            
            logger.info(f"Step 3: Demucs output: {stems_output}")
            
            vocals_url = stems_output.get("vocals") if isinstance(stems_output, dict) else None
            bass_url = stems_output.get("bass") if isinstance(stems_output, dict) else None
            drums_url = stems_output.get("drums") if isinstance(stems_output, dict) else None
            other_url = stems_output.get("other") if isinstance(stems_output, dict) else None
            guitar_url = stems_output.get("guitar") if isinstance(stems_output, dict) else None
            
            instrument_url = other_url
            if not instrument_url:
                return jsonify({
                    "error": "Could not identify instrument stem from Demucs output",
                    "demucs_output": str(stems_output),
                }), 500
            
            logger.info("Step 4: Downloading instrument stem for Librosa analysis...")
            stem_path = tempfile.mktemp(suffix=".wav")
            temp_files.append(stem_path)
            download_stem(instrument_url, stem_path)
            
            analysis_wav = stem_path + ".analysis.wav"
            temp_files.append(analysis_wav)
            subprocess.run(
                ["ffmpeg", "-i", stem_path, "-ar", "22050", "-ac", "1", "-y", analysis_wav],
                capture_output=True, text=True, timeout=30
            )
        
        logger.info("Step 5: Running Librosa analysis...")
        metrics = analyze_stem(analysis_wav)
        gc.collect()
        t3 = time.time()
        
        # ─── Step 6: Reference comparison ───────────────────────────
        # Cover Band Mode + New Cover Mode pull a reference for context.
        # Original Track Mode SKIPS reference fetching entirely — there is
        # no published reference recording for an original composition.
        logger.info(f"Step 6: Resolving reference (critique_mode={song_critique_mode})...")
        from scoring import calculate_scores
        
        song_title = request.form.get("song_title", "")
        song_artist = request.form.get("song_artist", "")
        
        reference_analysis = None
        reference_track = None

        if song_critique_mode == "Original Track Mode":
            # Phase 4.5 follow-up: Original Track Mode now accepts a user-
            # uploaded reference (e.g. their own studio version or demo).
            # The reference is for Claude's context only — scoring stays
            # purely creative because there's no published original to compare
            # against, and judging an original composition against the user's
            # own demo would create incoherent scoring semantics. The reference
            # just gives Claude richer audio context for feedback generation.
            if "reference_file" in request.files:
                logger.info("Original Track Mode — using user-uploaded reference for context only")
                ref_file = request.files["reference_file"]
                ref_path = tempfile.mktemp(suffix=".upload_ref")
                ref_file.save(ref_path)
                temp_files.append(ref_path)

                ref_wav = ref_path + ".wav"
                temp_files.append(ref_wav)
                subprocess.run(
                    ["ffmpeg", "-i", ref_path, "-ar", "22050", "-ac", "1", "-y", ref_wav],
                    capture_output=True, text=True, timeout=30,
                )
                reference_analysis = analyze_stem(ref_wav, lightweight=True)
                gc.collect()
                reference_track = {"title": song_title, "artist": song_artist, "source": "user_upload"}
            else:
                logger.info("Original Track Mode — no reference uploaded, scoring purely on internal consistency")
        elif ref_mode == "strict" and song_title:
            # Cover Band Mode — pull reference for strict comparison
            if "reference_file" in request.files:
                logger.info("Using user-uploaded reference track...")
                ref_file = request.files["reference_file"]
                ref_path = tempfile.mktemp(suffix=".upload_ref")
                ref_file.save(ref_path)
                temp_files.append(ref_path)
                
                ref_wav = ref_path + ".wav"
                temp_files.append(ref_wav)
                subprocess.run(
                    ["ffmpeg", "-i", ref_path, "-ar", "22050", "-ac", "1", "-y", ref_wav],
                    capture_output=True, text=True, timeout=30,
                )
                reference_analysis = analyze_stem(ref_wav, lightweight=True)
                gc.collect()
                reference_track = {"title": song_title, "artist": song_artist, "source": "user_upload"}
            else:
                cached = get_cached_reference(song_title, song_artist)
                if cached:
                    reference_analysis = cached
                    reference_track = {"title": song_title, "artist": song_artist, "source": "cache"}
                else:
                    logger.info("Fetching Deezer reference for strict mode scoring...")
                    deezer_ref = fetch_deezer_reference(song_title, song_artist)
                    if deezer_ref:
                        reference_analysis = deezer_ref["analysis"]
                        reference_track = deezer_ref["track"]
                        cache_song_reference(reference_track, reference_analysis)
                    else:
                        logger.info("No Deezer reference found, falling back to creative mode")
                        ref_mode = "creative"
        elif ref_mode == "creative" and song_title:
            # New Cover Mode — pull reference for context only (scoring stays creative)
            cached = get_cached_reference(song_title, song_artist)
            if cached:
                reference_analysis = cached
                reference_track = {"title": song_title, "artist": song_artist, "source": "cache"}
            else:
                logger.info("Fetching Deezer reference for New Cover context...")
                deezer_ref = fetch_deezer_reference(song_title, song_artist)
                if deezer_ref:
                    reference_analysis = deezer_ref["analysis"]
                    reference_track = deezer_ref["track"]
                    cache_song_reference(reference_track, reference_analysis)

        skill_level = request.form.get("skill_level", "Intermediate")
        scores = calculate_scores(metrics, reference_analysis=reference_analysis, mode=ref_mode, skill_level=skill_level)
        t4 = time.time()
        
        # ─── Step 7: Visual analysis ────────────────────────────────
        # Skipped entirely for audio-only submissions.
        visual_analysis = None
        instrument = request.form.get("instrument", "")

        if audio_only:
            logger.info("Step 7: Audio-only submission — skipping visual analysis")
            t_visual = t4
        else:
            from visual import analyze_video
            logger.info("Step 7: Running visual analysis on video frames...")
            visual_analysis = analyze_video(input_path, instrument=instrument)
            t_visual = time.time()

        # ─── Step 7b: Whisper transcription (vocals only, audio path) ───
        # Whisper runs on the analysis_wav regardless of whether the source
        # was video or audio — same audio data either way.
        lyrics_transcript = None
        if OPENAI_API_KEY and "vocal" in instrument.lower():
            try:
                logger.info("Step 7b: Transcribing vocals with Whisper...")
                import openai
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                with open(analysis_wav, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                    )
                lyrics_transcript = transcript.text
                logger.info(f"Whisper transcript: {lyrics_transcript[:100]}...")
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {e}")
        
        # ─── Step 8: Claude feedback (Phase 4.5: unified persona) ───
        feedback = None
        from feedback import generate_feedback, ANTHROPIC_API_KEY as _ak
        if _ak:
            logger.info("Step 8: Generating Claude feedback...")
            song_context = {
                "title": song_title or "Unknown",
                "artist": song_artist or "Unknown",
            }
            # Phase 4.5 artist_context — `harshness` removed, `style` removed,
            # `intentional_choices` renamed to `creative_choices`,
            # `influence` removed (UI dropped).
            creative_choices = (
                request.form.get("creative_choices")
                or request.form.get("intentional_choices", "")  # legacy fallback
            )
            artist_context = {
                "skill_level": skill_level,
                "instrument": instrument,
                "capo": request.form.get("capo", "none"),
                "genre": request.form.get("genre", ""),
                "environment": request.form.get("environment", "Bedroom Tape"),
                "creative_choices": creative_choices,
            }
            feedback = generate_feedback(
                scores=scores,
                analysis=metrics,
                song_context=song_context,
                artist_context=artist_context,
                song_critique_mode=song_critique_mode,
                visual_analysis=visual_analysis,
                lyrics_transcript=lyrics_transcript,
            )
        t5 = time.time()
        
        # ─── Step 9: Save submission ─────────────────────────────────
        submission_id = None
        try:
            from datetime import datetime, timezone
            submission_data = {
                "user_id": user_id,
                "submitter_name": (user_email.split("@")[0] if user_email else "Anonymous"),
                "song_title": song_title,
                "song_artist": song_artist,
                "skill_level": skill_level,
                "harshness": "Unified",  # Phase 4.5 — persona system removed
                "song_critique_mode": song_critique_mode,
                "genre": request.form.get("genre"),
                "environment": request.form.get("environment", "Bedroom Tape"),
                "creative_choices": creative_choices,
                "reference_weighting": reference_weighting_db,
                "stems": {
                    "vocals": vocals_url, "bass": bass_url,
                    "drums": drums_url, "other": other_url, "guitar": guitar_url,
                },
                "performance_analysis": metrics,
                "scores": scores,
                "feedback": feedback,
                "visual_analysis": visual_analysis,
                "pipeline_timing": {
                    "ffmpeg": round(t1 - t0, 2), "demucs": round(t2 - t1, 2),
                    "librosa": round(t3 - t2, 2), "scoring": round(t4 - t3, 2),
                    "visual": round(t_visual - t4, 2), "feedback": round(t5 - t_visual, 2),
                    "total": round(t5 - t0, 2),
                },
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            submission_id = save_submission(submission_data)
        except Exception as e:
            logger.warning(f"Failed to save submission: {e}")
        
        return jsonify({
            "status": "ok",
            "submission_id": submission_id,
            "song_critique_mode": song_critique_mode,
            "audio_only": audio_only,
            "pipeline_timing": {
                "ffmpeg_seconds": round(t1 - t0, 2),
                "demucs_seconds": round(t2 - t1, 2),
                "demucs_predict_seconds": round(demucs_time, 2),
                "librosa_seconds": round(t3 - t2, 2),
                "scoring_seconds": round(t4 - t3, 2),
                "visual_seconds": round(t_visual - t4, 2),
                "feedback_seconds": round(t5 - t_visual, 2),
                "total_seconds": round(t5 - t0, 2),
            },
            "stems": {
                "vocals": vocals_url,
                "bass": bass_url,
                "drums": drums_url,
                "other": other_url,
                "guitar": guitar_url,
            },
            "analysis": metrics,
            "scores": scores,
            "reference": {
                "track": reference_track,
                "key": reference_analysis.get("detected_key") if reference_analysis else None,
                "source": reference_track.get("source", "deezer") if reference_track else None,
            } if reference_track else None,
            "visual_analysis": visual_analysis,
            "feedback": feedback,
        })
    
    except Exception as e:
        logger.exception("analyze failed")
        return jsonify({"error": str(e)}), 500
    finally:
        for f in temp_files:
            try:
                if os.path.exists(f): os.unlink(f)
            except:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
