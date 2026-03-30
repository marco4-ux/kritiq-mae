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
    if file_size < 10 * 1024 * 1024:
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:audio/wav;base64,{data}"
    
    raise RuntimeError(f"File too large for data URI ({file_size} bytes). Need external hosting.")

# ─── Helper: Call Replicate Demucs API ───────────────────────────────

def separate_stems_replicate(audio_url):
    """Send audio to Replicate Demucs, return dict of stem URLs."""
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN not set")
    
    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # sync mode — wait up to 60s
    }
    
    payload = {
        "input": {
            "audio": audio_url,
            "model_name": "htdemucs",
        }
    }
    
    logger.info("Sending to Replicate Demucs (sync mode)...")
    resp = http_requests.post(
        f"https://api.replicate.com/v1/models/{REPLICATE_MODEL}/predictions",
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
    
    # Timing regularity: std of inter-onset intervals (lower = more consistent)
    if len(onset_times) > 1:
        intervals = np.diff(onset_times)
        timing_consistency = round(1.0 - min(float(np.std(intervals) / (np.mean(intervals) + 1e-6)), 1.0), 3)
        avg_bpm = round(60.0 / float(np.mean(intervals)), 1) if np.mean(intervals) > 0 else 0
    else:
        timing_consistency = 0.0
        avg_bpm = 0
    
    # --- RMS energy (dynamics) ---
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    avg_rms = round(float(np.mean(rms)), 4)
    dynamic_range = round(float(np.max(rms) - np.min(rms)), 4)
    
    # --- Spectral features (tone quality) ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    avg_brightness = round(float(np.mean(spectral_centroid)), 1)
    
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
        "pitches_per_second": pitches_per_second[:60],  # cap at 60s
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


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    """
    Full MAE pipeline:
    1. Upload video/audio → FFmpeg extracts WAV
    2. WAV → Replicate Demucs → separated stems (vocals, no_vocals/other)
    3. Instrument stem → Librosa analysis → structured metrics
    4. Return JSON with all metrics + stem URLs
    """
    if request.method == "OPTIONS":
        return '', 200
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".upload", delete=False) as tmp:
        file.save(tmp.name)
        input_path = tmp.name
    
    temp_files = [input_path]
    
    try:
        t0 = time.time()
        
        # Step 1: Extract audio
        logger.info("Step 1: Extracting audio with FFmpeg...")
        wav_path = extract_audio(input_path, sr=44100, mono=False)
        temp_files.append(wav_path)
        t1 = time.time()
        
        # Step 2: Send to Replicate Demucs for stem separation
        logger.info("Step 2: Uploading to Replicate Demucs...")
        audio_url = upload_for_replicate(wav_path)
        stems_output, demucs_time = separate_stems_replicate(audio_url)
        t2 = time.time()
        
        # Step 3: Download the "no_vocals" (instrument) stem for analysis
        # Demucs output is typically a dict with stem name → URL
        # or a list. We need to find the instrument stem.
        logger.info(f"Step 3: Demucs output type: {type(stems_output)}, content: {stems_output}")
        
        instrument_url = None
        vocals_url = None
        
        if isinstance(stems_output, dict):
            # Dict with stem names as keys
            instrument_url = stems_output.get("no_vocals") or stems_output.get("other") or stems_output.get("accompaniment")
            vocals_url = stems_output.get("vocals")
        elif isinstance(stems_output, list):
            # List of URLs — typically [bass, drums, other, vocals] or similar
            # We'll need to identify which is which by filename
            for url in stems_output:
                if isinstance(url, str):
                    lower = url.lower()
                    if "no_vocals" in lower or "other" in lower or "accompaniment" in lower:
                        instrument_url = url
                    elif "vocal" in lower:
                        vocals_url = url
            # Fallback: if we can't identify, use first non-vocal
            if not instrument_url and stems_output:
                instrument_url = stems_output[0] if not vocals_url or stems_output[0] != vocals_url else stems_output[-1]
        elif isinstance(stems_output, str):
            # Single URL — might be a zip or combined output
            instrument_url = stems_output
        
        if not instrument_url:
            return jsonify({
                "error": "Could not identify instrument stem from Demucs output",
                "demucs_output": str(stems_output),
            }), 500
        
        # Step 4: Download instrument stem and analyze with Librosa
        logger.info("Step 4: Downloading instrument stem for Librosa analysis...")
        stem_path = tempfile.mktemp(suffix=".wav")
        temp_files.append(stem_path)
        download_stem(instrument_url, stem_path)
        
        # Convert to mono 22050 for Librosa
        stem_wav = stem_path + ".analysis.wav"
        temp_files.append(stem_wav)
        subprocess.run(
            ["ffmpeg", "-i", stem_path, "-ar", "22050", "-ac", "1", "-y", stem_wav],
            capture_output=True, text=True, timeout=30
        )
        
        logger.info("Step 5: Running Librosa analysis on instrument stem...")
        metrics = analyze_stem(stem_wav)
        t3 = time.time()
        
        return jsonify({
            "status": "ok",
            "pipeline_timing": {
                "ffmpeg_seconds": round(t1 - t0, 2),
                "demucs_seconds": round(t2 - t1, 2),
                "demucs_predict_seconds": round(demucs_time, 2),
                "librosa_seconds": round(t3 - t2, 2),
                "total_seconds": round(t3 - t0, 2),
            },
            "stems": {
                "instrument_url": instrument_url,
                "vocals_url": vocals_url,
            },
            "analysis": metrics,
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
