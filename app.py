from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import json

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route("/health", methods=["GET"])
def health():
    """Basic health check"""
    return jsonify({"status": "ok", "message": "MAE server running"})


@app.route("/test-ffmpeg", methods=["POST"])
def test_ffmpeg():
    """Test that FFmpeg is installed and can process audio"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name
    
    output_path = input_path.replace(".mp4", ".wav")
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", "-y", output_path],
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode != 0:
            return jsonify({"error": "FFmpeg failed", "stderr": result.stderr}), 500
        
        file_size = os.path.getsize(output_path)
        
        return jsonify({
            "status": "ok",
            "message": "FFmpeg working",
            "output_size_bytes": file_size
        })
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


@app.route("/test-demucs", methods=["POST", "OPTIONS"])
def test_demucs():
    """Test that Demucs can separate audio stems"""
    if request.method == "OPTIONS":
        return '', 200
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name
    
    wav_path = input_path.replace(".mp4", ".wav")
    
    try:
        subprocess.run(
           ["ffmpeg", "-i", input_path, "-ar", "44100", "-ac", "1", "-t", "30", "-y", wav_path],
            capture_output=True, text=True, timeout=60
        )
        os.unlink(input_path)
        
        result = subprocess.run(
            ["python", "-m", "demucs", "--two-stems", "vocals",
             "-n", "htdemucs", "--segment", "10",
             "-o", "/tmp/demucs_out", "--mp3", wav_path],
            capture_output=True, text=True, timeout=300
        )
        
        if result.returncode != 0:
            return jsonify({"error": "Demucs failed", "stderr": result.stderr}), 500
        
        stems_found = []
        for root, dirs, files in os.walk("/tmp/demucs_out"):
            for f in files:
                filepath = os.path.join(root, f)
                stems_found.append({
                    "name": f,
                    "size_bytes": os.path.getsize(filepath)
                })
        
        return jsonify({
            "status": "ok",
            "message": "Demucs separation complete",
            "stems": stems_found
        })
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        subprocess.run(["rm", "-rf", "/tmp/demucs_out"], capture_output=True)

@app.route("/test-pitch", methods=["POST", "OPTIONS"])
def test_pitch():
    """Test pitch detection with librosa"""
    if request.method == "OPTIONS":
        return '', 200
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name
    
    wav_path = input_path.replace(".mp4", ".wav")
    
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-ar", "22050", "-ac", "1", "-y", wav_path],
            capture_output=True, text=True, timeout=60
        )
        os.unlink(input_path)
        
        import librosa
        import numpy as np
        
        y, sr = librosa.load(wav_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        dominant_pitches = []
        
        hop_length = 512
        frames_per_second = sr / hop_length
        for i in range(0, chroma.shape[1], max(1, int(frames_per_second))):
            dominant_idx = np.argmax(chroma[:, i])
            timestamp = i / frames_per_second
            dominant_pitches.append({
                "time": round(float(timestamp), 2),
                "note": pitch_classes[dominant_idx],
                "confidence": round(float(chroma[dominant_idx, i]), 3)
            })
        
        return jsonify({
            "status": "ok",
            "message": "Pitch detection complete",
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "detected_pitches": dominant_pitches[:30]
        })
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
