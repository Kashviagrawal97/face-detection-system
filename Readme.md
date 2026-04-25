# Face Detection — Haar Cascade + FastAPI

Face detection system using OpenCV Haar Cascade with a REST API for registering, verifying, and deleting users.

## Project Structure

├── face_detection.py    # Haar Cascade model + face extraction + live webcam
├── api.py               # FastAPI server (3 endpoints)
└── requirements.txt

## Run
```bash
# API server
uvicorn api:app --host localhost --port 8000

# Live webcam (local only)
python face_detection.py
```

Swagger UI available at `http://localhost:8000/docs`

## Tech Stack
Python, OpenCV, FastAPI, NumPy