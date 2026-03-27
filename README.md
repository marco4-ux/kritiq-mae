# Kritiq MAE (Music Analysis Engine)

Server-side audio processing for Kritiq app.

## Endpoints

- `GET /health` — Health check
- `POST /test-ffmpeg` — Upload video/audio, test FFmpeg extraction
- `POST /test-demucs` — Upload WAV, test vocal/instrument separation
- `POST /test-pitch` — Upload WAV, test pitch/note detection

## Deploy to Railway

1. Push this repo to GitHub
2. Connect GitHub repo to Railway project
3. Railway auto-detects the Dockerfile and deploys
4. Server runs on the assigned PORT

## Local Development

```bash
pip install -r requirements.txt
python app.py
```
