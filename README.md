# Face Recognition Kiosk — Attendance & Access Control

A real-time, edge-based face recognition system with multi-layer anti-spoofing. Designed for kiosk deployment to automate attendance logging and access control.

---

## Overview

The system processes a live camera feed through a sequential pipeline:

1. **Face Detection** — MediaPipe BlazeFace locates faces and extracts 468 3D landmarks.
2. **Image Quality Assessment** — Laplacian variance, brightness, and contrast checks discard blurry or poorly lit frames before any AI inference.
3. **Geometric Alignment** — A 5-point similarity transform (eyes, nose, mouth corners) warps each face to a canonical 160×160 crop.
4. **Head Pose Estimation** — SolvePnP derives Euler angles (Yaw/Pitch/Roll). Frames where the user is not facing the camera are skipped.
5. **Anti-Spoofing** — MiniFASNetV2 classifies the input as `real` or `spoof` (print attack, screen replay). Spoofs are rejected immediately.
6. **Face Embedding** — FaceNet (InceptionResnetV1, VGGFace2 pretrained) extracts a 512-dimensional L2-normalized embedding.
7. **Temporal Voting** — Per-track results are buffered over N frames. Identity is confirmed only when vote count, average similarity, and score variance all meet configured thresholds.
8. **Attendance Logging** — On confirmation, a timestamped record (name, score, face crop) is written to CSV. A per-person cooldown prevents duplicate entries.

---

## Architecture

```
┌──────────────────────────────────────┐
│  Kiosk Frontend (Next.js)            │
│  Admin Dashboard (Next.js)           │
└─────────────┬────────────────────────┘
              │  WebSocket / REST
┌─────────────▼────────────────────────┐
│  FastAPI Backend  (server.py)        │
│  ├─ /ws           — frame processing │
│  ├─ /api/register — enrollment       │
│  ├─ /api/users    — user management  │
│  ├─ /api/stats    — dashboard data   │
│  └─ /api/door     — manual control   │
└─────────────┬────────────────────────┘
              │
┌─────────────▼────────────────────────┐
│  Core Pipeline                       │
│  FaceDetector → Liveness → Embedder  │
│  → Recognizer → AttendanceLogger     │
└──────────────────────────────────────┘
              │
┌─────────────▼────────────────────────┐
│  Data Layer                          │
│  data/embeddings.json  (vector DB)   │
│  data/attendance.csv   (log)         │
│  data/faces/           (enroll imgs) │
└──────────────────────────────────────┘
```

---

## Project Structure

```
.
├── core/
│   ├── detector.py         # Face detection, 5-point alignment, head pose
│   ├── embedder.py         # FaceNet wrapper, quality gate, cosine matching
│   ├── liveness.py         # MiniFASNet model definition and inference
│   ├── liveness_tracker.py # Per-track liveness voting
│   └── recognizer.py       # Temporal voting logic, IoU tracker
├── config/
│   ├── settings.py         # Camera, paths, quality thresholds
│   └── thresholds.py       # Accept/reject scores, voting parameters
├── frontend/               # Next.js app (Kiosk UI + Admin Dashboard)
├── models/                 # Model weights (not tracked, see Setup)
├── data/                   # Runtime data (not tracked)
├── server.py               # FastAPI server (WebSocket + REST API)
├── app.py                  # Standalone desktop app (Tkinter)
├── db.py                   # Embedding database (JSON-backed)
├── attendance.py           # CSV attendance logger
└── .env.local              # Secrets (not tracked)
```

---

## Requirements

- Python 3.9+
- Node.js 18+
- CUDA-capable GPU (optional; CPU is supported)

**Python dependencies:**
```
torch torchvision
opencv-python
mediapipe
facenet-pytorch
fastapi uvicorn
httpx
pillow
numpy
```

**Node dependencies:**
```bash
cd frontend && npm install
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/hoangvu1806/FaceAttendance.git
cd FaceAttendance
pip install -r requirements.txt
```

### 2. Download model weights

Place the following file in the `models/` directory:

```
models/2.7_80x80_MiniFASNetV2.pth
```

Source: [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

### 3. Configure environment

Copy `.env.local.example` to `.env.local` and fill in values:

```env
ADMIN_PASSWORD=your_password
TELEGRAM_BOT_TOKEN=        # optional
TELEGRAM_CHAT_ID=          # optional
```

### 4. Create data directories

```bash
mkdir -p data/faces data/attendance_images
```

---

## Running

### Backend (FastAPI)

```bash
python server.py
# Listens on http://0.0.0.0:8000
```

### Frontend (Next.js)

```bash
cd frontend
npm run dev
# Available at http://localhost:3000
```

### Standalone Desktop App (no server required)

```bash
python app.py
```

---

## Configuration

All tuneable parameters are centralized in `config/`:

| File | Key | Default | Description |
| :--- | :--- | :--- | :--- |
| `thresholds.py` | `T_REJECT` | `0.70` | Cosine similarity below this → Unknown |
| `thresholds.py` | `T_ACCEPT` | `0.82` | Cosine similarity above this → Candidate |
| `thresholds.py` | `VOTING_FRAMES` | `8` | Frame buffer size for temporal voting |
| `thresholds.py` | `VOTING_THRESHOLD` | `6` | Minimum votes required for confirmation |
| `thresholds.py` | `MIN_CONSISTENCY` | `0.85` | Vote ratio required |
| `thresholds.py` | `ATTENDANCE_COOLDOWN` | `30` | Seconds before re-logging same person |
| `settings.py` | `MIN_SHARPNESS` | `100.0` | Laplacian variance threshold |
| `settings.py` | `DETECTION_CONFIDENCE` | `0.7` | MediaPipe detection confidence |

---

## API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `WS` | `/ws` | Stream JPEG frames; receive recognition results |
| `POST` | `/api/login` | Admin authentication |
| `GET` | `/api/users` | List enrolled persons |
| `POST` | `/api/register` | Enroll a new person (base64 images) |
| `DELETE` | `/api/users/{name}` | Remove a person |
| `PUT` | `/api/users/{name}` | Rename a person |
| `GET` | `/api/users/{name}/images` | Fetch enrollment images |
| `GET` | `/api/stats` | Today's attendance stats + kiosk status |
| `GET` | `/api/full_stats` | Full attendance log |
| `POST` | `/api/door/open` | Manually trigger door unlock |

---

## Key Design Decisions

**Temporal voting over single-frame decisions.** A single frame can be ambiguous due to motion blur or partial occlusion. The system accumulates results over a sliding window and applies majority voting with variance constraints to ensure stable identification before logging.

**Quality gate before AI inference.** Running FaceNet and MiniFASNet on every frame is expensive and produces noisy results on low-quality inputs. By filtering with Laplacian sharpness and luminance statistics first, throughput and accuracy both improve.

**Edge computation.** All models run locally on the kiosk device. No biometric data leaves the local network.

---

## License

MIT