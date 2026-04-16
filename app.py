from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import json
import time
import requests as http_requests
import logging

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
    Call this when the user opens the submission form — by the time they
    fill in fields and hit submit, the GPU will be warm and ready.
    Returns immediately after sending the request (doesn't wait for result).
    """
    if not REPLICATE_API_TOKEN:
        return jsonify({"status": "error", "message": "REPLICATE_API_TOKEN not set"}), 500
    
    try:
        import base64
        import struct
        import io
        
        # Generate a 1-second silent WAV in memory (smallest valid audio)
        sample_rate = 8000
        num_samples = sample_rate  # 1 second
        wav_buf = io.BytesIO()
        # WAV header
        data_size = num_samples * 2  # 16-bit mono
        wav_buf.write(b'RIFF')
        wav_buf.write(struct.pack('<I', 36 + data_size))
        wav_buf.write(b'WAVE')
        wav_buf.write(b'fmt ')
        wav_buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        wav_buf.write(b'data')
        wav_buf.write(struct.pack('<I', data_size))
        wav_buf.write(b'\x00' * data_size)  # silence
        
        audio_data = base64.b64encode(wav_buf.getvalue()).decode("utf-8")
        audio_url = f"data:audio/wav;base64,{audio_data}"
        
        # Send async prediction (don't wait for result — just wake up the GPU)
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
    """Extract audio from video/audio file, return path to WAV."""
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
    """Upload a file and return a public URL for Replicate to consume.
    Uses a simple file hosting approach via data URI for small files,
    or a temporary upload service."""
    import base64
    
    file_size = os.path.getsize(file_path)
    
    # For files under 10MB, use data URI (Replicate accepts these)
    if file_size < 20 * 1024 * 1024:
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:audio/wav;base64,{data}"
    
    raise RuntimeError(f"File too large for data URI ({file_size} bytes). Need external hosting.")

# ─── Helper: Deezer reference lookup + analysis ──────────────────────

def fetch_deezer_reference(song_title: str, song_artist: str) -> dict:
    """
    Search Deezer for a song, download the 30s preview, run Librosa on it.
    Returns {"track": {...}, "analysis": {...}} or None if not found.
    Results should be cached in Supabase (songs table) after first lookup.
    """
    query = f"{song_title} {song_artist}".strip()
    if not query:
        return None
    
    try:
        # Step 1: Search Deezer
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
        
        # Step 2: Download 30s preview
        dl_resp = http_requests.get(preview_url, timeout=30)
        dl_resp.raise_for_status()
        
        preview_path = tempfile.mktemp(suffix=".mp3")
        with open(preview_path, "wb") as f:
            f.write(dl_resp.content)
        
        # Step 3: Convert to WAV and analyze
        wav_path = preview_path + ".wav"
        subprocess.run(
            ["ffmpeg", "-i", preview_path, "-ar", "22050", "-ac", "1", "-y", wav_path],
            capture_output=True, text=True, timeout=30,
        )
        
        ref_analysis = analyze_stem(wav_path)
        
        # Cleanup
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
    """Look up cached song reference from Supabase. Returns analysis dict or None."""
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
            "reference_analysis": reference_analysis,  # PostgREST accepts dicts for JSONB
            "reference_source": "deezer",
            "reference_analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Upsert - insert or update on conflict (title + artist unique constraint)
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
    """
    Ensure a song exists in the `songs` table and return its UUID.
    If the song already exists (by title + artist), returns its existing id.
    If not, inserts a minimal row and returns the new id.
    
    Returns the song UUID as a string, or None on failure.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or not song_title:
        return None
    
    try:
        # Check if song already exists
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
        
        # Insert minimal row
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
    song_title = data.pop("song_title", "")
    song_artist = data.pop("song_artist", "")
    
    # Write to performances (always, anonymous-compatible)
    performance_id = None
    try:
        performance_payload = {
            "song_title": song_title,
            "artist_name": song_artist,
            "feedback": data.get("feedback"),
            "overall_score": data.get("scores", {}).get("overall") if data.get("scores") else None,
            "processed": True,
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
            submission_payload = {
                "user_id": user_id,
                "song_id": song_id,
                "skill_level": data.get("skill_level"),
                "harshness": data.get("harshness"),
                "style": data.get("style"),
                "genre": data.get("genre"),
                "environment": data.get("environment"),
                "intentional_choices": data.get("intentional_choices"),
                "influence": data.get("influence"),
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
    if not auth_header:
        return (None, None)

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return (None, "Invalid Authorization header format")

    token = parts[1]

    try:
        alg = pyjwt.get_unverified_header(token).get("alg", "").upper()

        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                return (None, "SUPABASE_JWT_SECRET not configured")
            payload = pyjwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        elif alg in ("RS256", "ES256"):
            jwks_client = _get_jwks_client()
            if not jwks_client:
                return (None, "JWKS client could not be initialized")
            signing_key = jwks_client.get_signing_key_from_jwt(token).key
            payload = pyjwt.decode(token, signing_key, algorithms=[alg], options={"verify_aud": False})
        else:
            return (None, f"Unsupported signing algorithm: {alg}")

        user_id = payload.get("sub")
        if not user_id:
            return (None, "JWT missing sub claim")
        return (user_id, None)

    except pyjwt.ExpiredSignatureError:
        return (None, "Token expired")
    except pyjwt.InvalidTokenError as e:
        return (None, f"Invalid token: {str(e)}")

# ─── Helper: Call Replicate Demucs API ───────────────────────────────

def separate_stems_replicate(audio_url):
    """Send audio to Replicate Demucs, return dict of stem URLs."""
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # sync mode - wait up to 60s
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
    
    # If sync completed, status will be "succeeded"
    # If it took >60s, we need to poll
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
    
    Returns a dict with these keys:
        f0_contour: list of {time, hz, confidence, voiced} — downsampled to ~10Hz
        voiced_ratio: 0..1 — fraction of audio with a clear single fundamental
        stability_score: 0..1 — inverse of frame-to-frame f0 jitter (higher = more stable)
        drift_cents: float — mean deviation from local tonic in cents (signed)
        off_pitch_segments: list of {time, duration, deviation_cents, confidence}
        vibrato: {
            detected: bool,
            rate_hz: float or null,
            extent_cents: float or null,
            segments: [{time, duration, rate_hz, extent_cents}]
        }
    
    On strummed/polyphonic audio, voiced_ratio will be low (typically <0.3)
    and consumers should downweight pitch-specific claims accordingly.
    """
    import librosa
    import numpy as np
    
    # PYIN with librosa defaults (C2 65Hz to C7 2093Hz)
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
    
    # Confidence threshold for treating a frame as usable pitch data
    CONF_THRESHOLD = 0.7
    
    # Voiced ratio: fraction of frames with confident fundamental
    reliable_mask = np.array([
        bool(vf) and vp >= CONF_THRESHOLD and f is not None and not np.isnan(f)
        for vf, vp, f in zip(voiced_flag, voiced_prob, f0)
    ])
    voiced_ratio = float(np.mean(reliable_mask)) if len(reliable_mask) > 0 else 0.0
    
    # If essentially no reliable pitch data, return empty analysis early
    if voiced_ratio < 0.05:
        return _empty_pitch_analysis(voiced_ratio=voiced_ratio)
    
    # Downsample contour to ~10Hz for Claude's consumption (not every frame)
    # Full resolution stays available via numpy calculations below
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
    
    # Only compute derived metrics on reliable frames
    reliable_f0 = np.array([f for f, m in zip(f0, reliable_mask) if m])
    reliable_times = np.array([
        i / frames_per_second for i, m in enumerate(reliable_mask) if m
    ])
    
    # Stability: inverse of normalized frame-to-frame jitter in cents
    # Convert to cents first (log scale is how we hear pitch)
    stability_score = 0.5
    if len(reliable_f0) > 10:
        cents = 1200 * np.log2(reliable_f0 / 440.0)
        # Jitter = std of frame-to-frame differences
        diffs = np.abs(np.diff(cents))
        # Discard huge jumps (>200 cents) as note transitions, not instability
        note_transitions = diffs > 200
        jitter_diffs = diffs[~note_transitions]
        if len(jitter_diffs) > 0:
            mean_jitter = float(np.mean(jitter_diffs))
            # 0 cents jitter = perfect (1.0), 50+ cents = unstable (0.0)
            stability_score = max(0.0, min(1.0, 1.0 - mean_jitter / 50.0))
    
    # Drift: mean deviation from nearest semitone in cents (signed)
    drift_cents = 0.0
    if len(reliable_f0) > 0:
        cents = 1200 * np.log2(reliable_f0 / 440.0)
        nearest_semi = np.round(cents / 100.0) * 100.0
        deviations = cents - nearest_semi
        drift_cents = round(float(np.mean(deviations)), 1)
    
    # Off-pitch segments: sustained regions where deviation from nearest semitone
    # exceeds 25 cents for at least 200ms
    off_pitch_segments = _detect_off_pitch_segments(
        f0, voiced_flag, voiced_prob,
        frames_per_second=frames_per_second,
        deviation_threshold_cents=25,
        min_duration_s=0.2,
        conf_threshold=CONF_THRESHOLD,
    )
    
    # Vibrato: detect sustained notes and analyze f0 oscillation within
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
    """
    Find continuous segments where the performer sustained a note off-pitch.
    Returns list of {time, duration, deviation_cents, confidence}.
    """
    import numpy as np
    
    min_frames = int(min_duration_s * frames_per_second)
    segments = []
    current_start = None
    current_devs = []
    current_confs = []
    
    for i, (hz, vf, vp) in enumerate(zip(f0, voiced_flag, voiced_prob)):
        if hz is None or np.isnan(hz) or not vf or vp < conf_threshold:
            # End of segment if we had one building
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
    
    # Flush any trailing segment
    if current_start is not None and len(current_devs) >= min_frames:
        _flush_segment(segments, current_start, current_devs, current_confs, frames_per_second)
    
    # Cap at 15 segments to keep prompt size reasonable
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
    """
    Detect vibrato in sustained notes.
    Returns {detected, rate_hz, extent_cents, segments}.
    
    Algorithm:
    1. Find sustained voiced regions of at least min_sustained_s
    2. For each region, check if f0 oscillates periodically around a mean
    3. Measure rate (Hz) and extent (cents) via autocorrelation on the detrended signal
    """
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
        
        # Convert to cents relative to region mean
        mean_hz = np.mean(region_f0)
        if mean_hz <= 0:
            continue
        cents = 1200 * np.log2(region_f0 / mean_hz)
        
        # Detrend to isolate oscillation
        cents_detrended = cents - np.mean(cents)
        
        # Extent: half the peak-to-peak range (amplitude of oscillation)
        extent_cents = float((np.max(cents_detrended) - np.min(cents_detrended)) / 2.0)
        
        # Reject if barely any oscillation (<10 cents extent is not vibrato)
        if extent_cents < 10:
            continue
        
        # Rate: autocorrelation peak detection
        rate_hz = _estimate_oscillation_rate(cents_detrended, frames_per_second)
        
        # Vibrato is typically 3-9 Hz. Outside that range it's not musical vibrato.
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
        "segments": segments[:10],  # cap for prompt size
    }


def _estimate_oscillation_rate(signal, frames_per_second):
    """
    Estimate dominant oscillation rate via autocorrelation.
    Returns rate in Hz or None if no clear peak in the 3-9 Hz band.
    """
    import numpy as np
    
    if len(signal) < 20:
        return None
    
    # Autocorrelate
    sig = signal - np.mean(signal)
    if np.std(sig) == 0:
        return None
    
    acf = np.correlate(sig, sig, mode='full')
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0] if acf[0] != 0 else acf
    
    # Look for peaks in lag range corresponding to 3-9 Hz
    min_lag = int(frames_per_second / 9.0)  # fastest: 9 Hz
    max_lag = int(frames_per_second / 3.0)  # slowest: 3 Hz
    
    if max_lag >= len(acf):
        max_lag = len(acf) - 1
    if min_lag >= max_lag:
        return None
    
    search_region = acf[min_lag:max_lag]
    if len(search_region) == 0:
        return None
    
    peak_lag = int(np.argmax(search_region)) + min_lag
    peak_val = float(acf[peak_lag])
    
    # Require a reasonably strong periodic component
    if peak_val < 0.3:
        return None
    
    return frames_per_second / peak_lag


# ─── Helper: Librosa analysis on a stem ──────────────────────────────

def analyze_stem(wav_path):
    """Run Librosa analysis on an audio file. Returns structured metrics."""
    import librosa
    import numpy as np
    
    y, sr = librosa.load(wav_path, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # --- Chroma / chord detection ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    hop_length = 512
    frames_per_second = sr / hop_length
    
    # Per-second dominant pitch
    pitches_per_second = []
    for i in range(0, chroma.shape[1], max(1, int(frames_per_second))):
        dominant_idx = int(np.argmax(chroma[:, i]))
        timestamp = i / frames_per_second
        pitches_per_second.append({
            "time": round(float(timestamp), 2),
            "note": pitch_classes[dominant_idx],
            "confidence": round(float(chroma[dominant_idx, i]), 3),
        })
    
    # --- Overall key detection ---
    chroma_mean = np.mean(chroma, axis=1)
    detected_key = pitch_classes[int(np.argmax(chroma_mean))]
    key_confidence = round(float(np.max(chroma_mean)), 3)
    
    # --- Onset / timing analysis ---
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    
    # BPM: use librosa's beat tracker, not onset intervals
    # beat_track is designed for actual tempo detection and handles
    # arpeggiated/fingerpicked playing correctly
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
    avg_bpm = round(float(tempo) if not hasattr(tempo, '__len__') else float(tempo[0]), 1)
    
    # Timing regularity: std of inter-beat intervals (not inter-onset)
    if len(beat_frames) > 1:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)
        beat_intervals = np.diff(beat_times)
        timing_consistency = round(1.0 - min(float(np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-6)), 1.0), 3)
    elif len(onset_times) > 1:
        intervals = np.diff(onset_times)
        timing_consistency = round(1.0 - min(float(np.std(intervals) / (np.mean(intervals) + 1e-6)), 1.0), 3)
    else:
        timing_consistency = 0.0
    
    # --- RMS energy (dynamics) ---
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    avg_rms = round(float(np.mean(rms)), 4)
    dynamic_range = round(float(np.max(rms) - np.min(rms)), 4)
    
    # Perceptual tempo correction: fingerpicked ballads often register at
    # double BPM because Librosa counts every pluck as a beat.
    # If BPM > 150 and the track is low-energy, halve it.
    if avg_bpm > 150 and avg_rms < 0.02:
        avg_bpm = round(avg_bpm / 2, 1)
    
    # --- Spectral features (tone quality) ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    avg_brightness = round(float(np.mean(spectral_centroid)), 1)
    
    # --- Technique identification ---
    # Detect playing style from onset and spectral characteristics
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    
    # Strumming vs single note: strumming has wider spectral spread per onset
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
    avg_bandwidth = float(np.mean(spectral_bandwidth))
    
    # Onset sharpness: picks have sharper attacks than fingers
    if len(onset_frames) > 0:
        onset_strengths = onset_env[onset_frames] if len(onset_frames) <= len(onset_env) else onset_env[:len(onset_frames)]
        avg_onset_strength = float(np.mean(onset_strengths)) if len(onset_strengths) > 0 else 0
    else:
        avg_onset_strength = 0
    
    # Spectral rolloff: fingerpicking tends to have lower rolloff (warmer tone)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
    avg_rolloff = float(np.mean(spectral_rolloff))
    
    # Classify technique
    # High bandwidth + high onset strength = strumming with pick
    # High bandwidth + low onset strength = strumming with fingers
    # Low bandwidth + high onset strength = single notes with pick
    # Low bandwidth + low onset strength = fingerpicking
    bandwidth_threshold = 1800  # Hz
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
    
    # --- Vocal presence detection ---
    # Use spectral features to detect vocal-range energy (80Hz-1100Hz fundamental)
    # and vocal formant presence
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Vocal range bins (approximate: 80-1100Hz maps to mel bins ~5-40 at sr=22050)
    vocal_energy = np.mean(mel_db[5:40, :], axis=0)
    instrument_energy = np.mean(mel_db[40:, :], axis=0)
    
    # Detect segments where vocal energy is prominent
    vocal_ratio = float(np.mean(vocal_energy)) / (float(np.mean(instrument_energy)) + 1e-6)
    has_vocals = vocal_ratio > 0.8  # vocals detected if ratio is high enough
    
    # Vocal-instrument coordination: compare onset timing patterns
    # in vocal vs instrument frequency ranges
    if has_vocals:
        # Measure correlation between vocal and instrument energy over time
        if len(vocal_energy) > 1 and len(instrument_energy) > 1:
            min_len = min(len(vocal_energy), len(instrument_energy))
            correlation = float(np.corrcoef(vocal_energy[:min_len], instrument_energy[:min_len])[0, 1])
            coordination_score = round(max(0, min(1, (correlation + 1) / 2)), 3)  # normalize -1..1 to 0..1
        else:
            coordination_score = 0.5
    else:
        coordination_score = None  # no vocals detected
    
    # --- PYIN pitch analysis ---
    # Runs on the full mix. voiced_ratio will naturally be low for strummed
    # polyphonic audio, high for vocals and monophonic playing.
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
        # 1. Extract audio
        wav_path = extract_audio(input_path, sr=44100, mono=False)
        
        # 2. Upload for Replicate
        audio_url = upload_for_replicate(wav_path)
        
        # 3. Call Replicate Demucs
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
    
    # Accept song query via POST JSON or GET query param
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("query", "Hey There Delilah Plain White T's")
    else:
        query = request.args.get("query", "Hey There Delilah Plain White T's")
    
    output_path = f"/tmp/ytdlp_test_{int(time.time())}"
    
    try:
        t0 = time.time()
        
        # Step 1: Search and download audio only, max 30 seconds
        cmd = [
            "yt-dlp",
            "-x",                          # extract audio only
            "--audio-format", "wav",        # convert to wav
            "--audio-quality", "0",         # best quality
            "-o", f"{output_path}.%(ext)s", # output template
            "--no-playlist",                # single video only
            "--match-filter", "duration < 600",  # skip videos > 10 min
            "--postprocessor-args", "ffmpeg:-t 30",  # only keep first 30 seconds
            f"ytsearch1:{query}",           # search YouTube, take first result
        ]
        
        logger.info(f"Running yt-dlp with query: {query}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        t1 = time.time()
        
        # Check for output file
        wav_path = f"{output_path}.wav"
        if not os.path.exists(wav_path):
            # yt-dlp might have used a different extension
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
        
        # Step 2: Quick Librosa check on the downloaded audio
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
        
        # Step 1: Search Deezer
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
        
        # Step 2: Download the 30s preview MP3
        logger.info(f"Downloading preview: {preview_url}")
        preview_path = f"/tmp/deezer_preview_{int(time.time())}.mp3"
        dl_resp = http_requests.get(preview_url, timeout=30)
        dl_resp.raise_for_status()
        with open(preview_path, "wb") as f:
            f.write(dl_resp.content)
        
        preview_size = os.path.getsize(preview_path)
        t2 = time.time()
        
        # Step 3: Convert to WAV and run Librosa
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
    Full MAE pipeline:
    1. Upload video/audio → FFmpeg extracts WAV
    2. WAV → Replicate Demucs → separated stems (vocals, no_vocals/other)
    3. Instrument stem → Librosa analysis → structured metrics
    4. Return JSON with all metrics + stem URLs
    
    Auth behavior:
    - No Authorization header: anonymous path, writes to `performances` only
    - Valid JWT: authenticated path, writes to both `performances` and `submissions`
    - Invalid JWT: 401 (no silent fallback to anonymous)
    """
    if request.method == "OPTIONS":
        return '', 200
    
    # Verify JWT if provided (optional auth)
    auth_header = request.headers.get("Authorization", "")
    user_id, auth_error = verify_supabase_jwt(auth_header)
    if auth_header and auth_error:
        return jsonify({"error": f"Authentication failed: {auth_error}"}), 401
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".upload", delete=False) as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    temp_files = [input_path]
    
    try:
        t0 = time.time()
        
        # Step 1: Extract audio (capped at 60s - enough for full scoring, keeps Demucs fast)
        logger.info("Step 1: Extracting audio with FFmpeg (max 60s)...")
        wav_path = extract_audio(input_path, sr=44100, mono=False, max_duration=60)
        temp_files.append(wav_path)
        t1 = time.time()
        
        # Check if solo performance (skip Demucs) or full mix (use Demucs)
        solo_performance = request.form.get("solo_performance", "true").lower() in ("true", "1", "yes")
        
        vocals_url = None
        bass_url = None
        drums_url = None
        other_url = None
        guitar_url = None
        demucs_time = 0
        
        if solo_performance:
            # Solo performance: skip Demucs, analyze raw audio directly
            logger.info("Step 2: Solo performance - skipping Demucs, analyzing raw audio...")
            
            # Convert to mono 22050 for Librosa
            analysis_wav = wav_path + ".analysis.wav"
            temp_files.append(analysis_wav)
            subprocess.run(
                ["ffmpeg", "-i", wav_path, "-ar", "22050", "-ac", "1", "-y", analysis_wav],
                capture_output=True, text=True, timeout=30
            )
            t2 = time.time()
        else:
            # Full mix: use Demucs for stem separation
            logger.info("Step 2: Uploading to Replicate Demucs...")
            audio_url = upload_for_replicate(wav_path)
            stems_output, demucs_time = separate_stems_replicate(audio_url)
            t2 = time.time()
            
            # Parse Demucs output
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
            
            # Download instrument stem and convert for Librosa
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
        t3 = time.time()
        
        # Step 6: Calculate deterministic scores from metrics
        logger.info("Step 6: Calculating scores...")
        from scoring import calculate_scores
        
        # Get reference mode from request
        ref_mode = request.form.get("reference_weighting", "creative").lower()
        song_title = request.form.get("song_title", "")
        song_artist = request.form.get("song_artist", "")
        
        # If strict mode and song info provided, look up reference
        reference_analysis = None
        reference_track = None
        if ref_mode == "strict" and song_title:
            # Step 6a: Check for user-uploaded reference track
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
                reference_analysis = analyze_stem(ref_wav)
                reference_track = {"title": song_title, "artist": song_artist, "source": "user_upload"}
            else:
                # Step 6b: Check Supabase cache
                cached = get_cached_reference(song_title, song_artist)
                if cached:
                    reference_analysis = cached
                    reference_track = {"title": song_title, "artist": song_artist, "source": "cache"}
                else:
                    # Step 6c: Cache miss - fetch from Deezer, analyze, cache
                    logger.info("Fetching Deezer reference for strict mode scoring...")
                    deezer_ref = fetch_deezer_reference(song_title, song_artist)
                    if deezer_ref:
                        reference_analysis = deezer_ref["analysis"]
                        reference_track = deezer_ref["track"]
                        cache_song_reference(reference_track, reference_analysis)
                    else:
                        logger.info("No Deezer reference found, falling back to creative mode")
                        ref_mode = "creative"
        
        skill_level = request.form.get("skill_level", "Intermediate")
        scores = calculate_scores(metrics, reference_analysis=reference_analysis, mode=ref_mode, skill_level=skill_level)
        t4 = time.time()
        
        # Step 7: Visual analysis via Claude Vision (runs on original video, not audio)
        visual_analysis = None
        from visual import analyze_video
        instrument = request.form.get("instrument", "")
        logger.info("Step 7: Running visual analysis on video frames...")
        visual_analysis = analyze_video(input_path, instrument=instrument)
        t_visual = time.time()

        # Step 7b: Whisper transcription (if vocals selected and OpenAI key available)
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
        
        # Step 8: Generate Claude feedback (optional - skip if no API key)
        feedback = None
        from feedback import generate_feedback, ANTHROPIC_API_KEY as _ak
        if _ak:
            logger.info("Step 8: Generating Claude feedback...")
            song_context = {
                "title": request.form.get("song_title", "Unknown"),
                "artist": request.form.get("song_artist", "Unknown"),
            }
            artist_context = {
                "skill_level": request.form.get("skill_level", "Intermediate"),
                "harshness": request.form.get("harshness", "Supportive Producer"),
                "instrument": instrument,
                "capo": request.form.get("capo", "none"),
                "style": request.form.get("style", "Original Style"),
                "genre": request.form.get("genre", ""),
                "environment": request.form.get("environment", "Bedroom Tape"),
                "intentional_choices": request.form.get("intentional_choices", ""),
                "influence": request.form.get("influence", ""),
            }
            feedback = generate_feedback(
                scores=scores,
                analysis=metrics,
                song_context=song_context,
                artist_context=artist_context,
                visual_analysis=visual_analysis,
                lyrics_transcript=lyrics_transcript,
            )
        t5 = time.time()
        
        # Step 9: Save submission to Supabase (non-blocking, don't fail pipeline if this fails)
        submission_id = None
        try:
            from datetime import datetime, timezone
            submission_data = {
                "user_id": user_id,  # None for anonymous; populated for authenticated
                "song_title": song_title,
                "song_artist": song_artist,
                "skill_level": request.form.get("skill_level", "Intermediate"),
                "harshness": request.form.get("harshness", "Supportive Producer"),
                "style": request.form.get("style", "Original Style"),
                "genre": request.form.get("genre"),
                "environment": request.form.get("environment", "Bedroom Tape"),
                "intentional_choices": request.form.get("intentional_choices"),
                "influence": request.form.get("influence"),
                "reference_weighting": ref_mode.capitalize(),
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
