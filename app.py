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


@app.route("/test-demucs", methods=["POST"])
def test_demucs():
    """Test that Demucs can separate audio stems"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name
    
    output_dir = tempfile.mkdtemp()
    
    try:
        # Run Demucs separation - htdemucs is the v4 model
        result = subprocess.run(
            ["python", "-m", "demucs", "--two-stems", "vocals", 
             "-o", output_dir, "--mp3", input_path],
            capture_output=True, text=True, timeout=300
        )
        
        if result.returncode != 0:
            return jsonify({"error": "Demucs failed", "stderr": result.stderr}), 500
        
        # Find output files
        stems_found = []
        for root, dirs, files in os.walk(output_dir):
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
        os.unlink(input_path)
        # Clean up output dir
        subprocess.run(["rm", "-rf", output_dir], capture_output=True)


@app.route("/test-pitch", methods=["POST"])
def test_pitch():
    """Test pitch detection with librosa + crepe"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        file.save(tmp_in.name)
        input_path = tmp_in.name
    
    try:
        import librosa
        import numpy as np
        
        # Load audio
        y, sr = librosa.load(input_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Chroma features (pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Get the most prominent pitch class per frame
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        dominant_pitches = []
        
        # Sample every ~1 second
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
            "detected_pitches": dominant_pitches[:20]  # First 20 seconds
        })
    finally:
        os.unlink(input_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
