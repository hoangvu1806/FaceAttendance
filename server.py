
import cv2
import time
import asyncio
import uvicorn
import numpy as np
import base64
import os
import csv
import httpx
from datetime import datetime
from typing import List, Optional, Dict
from collections import deque, Counter

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from db import EmbeddingDB
from attendance import AttendanceLogger
from config import settings, thresholds

# Reuse core modules
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
from core.liveness import load_detector


# ============================================================================
# CONFIG & AUTH
# ============================================================================
def load_env_config():
    config = {
        "ADMIN_PASSWORD": "admin123",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": ""
    }
    env_path = '.env.local'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    config[key.strip()] = val.strip().strip('"').strip("'")
    return config

CONF = load_env_config()

# ============================================================================
# DATA MODELS
# ============================================================================
class LoginRequest(BaseModel):
    password: str

class RegisterRequest(BaseModel):
    name: str
    images: List[str]

class ManualAttendanceRequest(BaseModel):
    name: str

class RenameRequest(BaseModel):
    new_name: str

class DoorControlRequest(BaseModel):
    action: str # "open"
    duration: int = 5

# ============================================================================
# WEBSOCKET MANAGER & KIOSK TRACKING
# ============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, kiosk_id: str = "Unknown"):
        await websocket.accept()
        self.active_connections[websocket] = {
            "kiosk_id": kiosk_id,
            "status": "online",
            "connected_at": time.time(),
            "last_heartbeat": time.time()
        }

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    async def update_heartbeat(self, websocket: WebSocket, camera_status: str):
        if websocket in self.active_connections:
            self.active_connections[websocket]["last_heartbeat"] = time.time()
            self.active_connections[websocket]["camera_status"] = camera_status

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections.keys()):
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)
    
    def get_kiosk_stats(self):
        stats = []
        now = time.time()
        for ws, info in self.active_connections.items():
            stats.append({
                "id": info.get("kiosk_id", "Unknown"),
                "status": "online" if (now - info["last_heartbeat"]) < 10 else "unstable",
                "camera": info.get("camera_status", "ok"),
                "uptime": int(now - info["connected_at"])
            })
        return stats

manager = ConnectionManager()

# ============================================================================
# TELEGRAM BOT
# ============================================================================
async def send_telegram_alert(message: str):
    token = CONF.get("TELEGRAM_BOT_TOKEN")
    chat_id = CONF.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram Config Missing: Token or Chat ID not set.")
        return
    
    print(f"Sending Telegram Alert: {message}")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": f"KIOSK ALERT \n\n{message}"}
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                 print(f"Telegram Failed {resp.status_code}: {resp.text}")
            else:
                 print("Telegram Sent")
    except Exception as e:
        print(f"Telegram Error: {e}")

# ============================================================================
# SYSTEM STATE & CORE LOGIC
# ============================================================================
class SystemState:
    def __init__(self):
        self.detector = None
        self.embedder = None
        self.liveness_detector = None
        self.db = EmbeddingDB()
        self.attendance = AttendanceLogger()
        self.running = True
        
        # Access Params
        self.t_low = thresholds.T_REJECT
        self.t_high = thresholds.T_ACCEPT
        self.voting_frames = thresholds.VOTING_FRAMES
        self.voting_threshold = thresholds.VOTING_THRESHOLD
        self.attendance_cooldown = thresholds.ATTENDANCE_COOLDOWN
        self.confirmed_cache_time = settings.CONFIRMED_CACHE_TIME
        
        # State
        self.frame_buffer = {}
        self.last_attendance = {}
        
        # Face Tracking State
        self.face_tracks = {}
        self.next_track_id = 0
        self.max_track_age = 30 # frames
        self.iou_threshold = 0.3
        
        # Door State
        self.door_open_until = 0

    def load_models(self):
        print("Loading models...")
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.liveness_detector = load_detector(r'models/2.7_80x80_MiniFASNetV2.pth', 'cpu')
        print("Models loaded.")
    
    # ---------------------------------------------------------
    # TRACKING & VOTING LOGIC (RESTORED)
    # ---------------------------------------------------------
    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if xi2 <= xi1 or yi2 <= yi1: return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (w1 * h1) + (w2 * h2) - inter
        return inter / union if union > 0 else 0.0

    def update_face_tracks(self, current_faces):
        current_time = time.time()
        # Age tracks
        for tid in list(self.face_tracks.keys()):
            self.face_tracks[tid]['age'] += 1
            if self.face_tracks[tid]['age'] > self.max_track_age:
                del self.face_tracks[tid]
                if tid in self.frame_buffer: del self.frame_buffer[tid]
        
        matched_tracks = set()
        face_assignments = {}
        
        for idx, (box, _) in enumerate(current_faces):
            best_tid, best_iou = None, 0
            for tid, tdata in self.face_tracks.items():
                if tid in matched_tracks: continue
                iou = self.calculate_iou(box, tdata['last_box'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                self.face_tracks[best_tid].update({'last_box': box, 'age': 0, 'last_seen': current_time})
                matched_tracks.add(best_tid)
                face_assignments[idx] = best_tid
            else:
                new_tid = self.next_track_id
                self.next_track_id += 1
                self.face_tracks[new_tid] = {
                    'last_box': box, 'age': 0, 'last_seen': current_time,
                    'confirmed_name': None, 'confirmed_score': 0.0, 'confirmed_time': None,
                    'attendance_status': None
                }
                face_assignments[idx] = new_tid
        return face_assignments

    def add_vote(self, track_id, name, score):
        if track_id not in self.frame_buffer:
            self.frame_buffer[track_id] = deque(maxlen=self.voting_frames)
        self.frame_buffer[track_id].append({
            "name": name, 
            "score": score,
            "time": time.time()
        })

    def get_voting_result(self, track_id):
        """Get voting result from frame buffer with strict consistency check - Matches app.py logic"""
        if track_id not in self.frame_buffer:
            return "pending", None, 0.0, 0
        
        buffer = self.frame_buffer[track_id]
        if len(buffer) < self.voting_frames:
            return "pending", None, 0.0, len(buffer)
        
        names = [vote["name"] for vote in buffer]
        scores = [vote["score"] for vote in buffer]
        
        name_counts = Counter(names)
        most_common_name, vote_count = name_counts.most_common(1)[0]
        
        # Get scores for the most common name
        name_scores = [s for n, s in zip(names, scores) if n == most_common_name]
        avg_score = np.mean(name_scores)
        min_score = np.min(name_scores)
        score_std = np.std(name_scores)
        
        # Balanced acceptance criteria
        consistency_ratio = vote_count / len(buffer)
        
        # STRICT Match with app.py logic
        if (vote_count >= self.voting_threshold and 
            avg_score >= self.t_high and 
            min_score >= (self.t_high - 0.08) and  # Allow some variance
            score_std < thresholds.MIN_SCORE_VARIANCE and  # Use config value
            consistency_ratio >= thresholds.MIN_CONSISTENCY):  # Use config value
            return "accepted", most_common_name, avg_score, vote_count
        elif len(buffer) >= self.voting_frames:
            return "rejected", most_common_name, avg_score, vote_count
        else:
            return "pending", most_common_name, avg_score, vote_count

    def match(self, embedding):
        """Match embedding using margin-based decision - Matches app.py logic"""
        db = self.db.get_all()
        name, decision, s1, s2, margin = self.embedder.match_with_margin(embedding, db)
        
        # Convert to match app.py return style (name, score) but keep decision for internal logic if needed
        # app.py: returns name, score. If unknown, name is None.
        if decision == "accept":
            return name, s1
        elif decision == "uncertain":
            return name, s1  # Let voting decide
        else:  # unknown
            return None, s1

    # ---------------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------------
    def process_image_data(self, frame_bytes, websocket: WebSocket):
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return None
        
        current_time = time.time()
        asyncio.create_task(manager.update_heartbeat(websocket, "active"))

        # 1. Detection
        faces = self.detector.detect_and_crop(frame) # Returns [(box, face_img), ...]
        landmarks_list = self.detector.get_landmarks(frame)
        
        # 2. Update Tracks
        face_assignments = self.update_face_tracks(faces)
        
        results_data = []
        for idx, (box, face_img) in enumerate(faces):
            x, y, w, h = box
            track_id = face_assignments.get(idx, -1)
            
            # 3. Liveness Check
            liveness_score, liveness_label = self.liveness_detector.predict(frame, box)
            if liveness_label == 'fake':
                results_data.append({
                    "track_id": track_id,
                    "box": [int(x), int(y), int(w), int(h)],
                    "status": "fake", 
                    "message": "FAKE FACE DETECTED!", 
                    "color": "#ff0000", 
                    "liveness": float(liveness_score)
                })
                continue
            
            # 4. Recognition Logic - Strictly following app.py recognize_frame
            # Initialize default result
            result_meta = {
                "name": None, 
                "message": "Analyzing...", 
                "color": "#00a5ff", 
                "status": "pending"
            }
            
            # OPTIMIZATION: Check if this track is already confirmed
            # This logic mimics recognizer_frame optimization in app.py
            cached = False
            if track_id != -1 and track_id in self.face_tracks:
                tdata = self.face_tracks[track_id]
                
                # If already confirmed and not too old, return cached result
                if (tdata['confirmed_name'] is not None and 
                    tdata['confirmed_time'] is not None and
                    (current_time - tdata['confirmed_time']) < self.confirmed_cache_time):
                    
                    result_meta = {
                        "name": tdata['confirmed_name'],
                        "message": f"{tdata['confirmed_name']} - CONFIRMED",
                        "color": "#00ff00", 
                        "status": "confirmed"
                    }
                    cached = True
            
            if not cached:
                is_frontal = False
                if idx < len(landmarks_list):
                    is_frontal = self.detector.is_frontal_face(landmarks_list[idx])
                
                # Extract embedding (skip quality check for speed as per app.py)
                emb = self.embedder.extract(face_img, check_quality=True)
                
                if not emb:
                    result_meta = {"name": None, "message": "Error", "color": "#808080", "status": "error"}
                else:
                    # Match
                    name, score = self.match(emb)
                    
                    if name is None:
                        result_meta = {
                            "name": None, 
                            "message": f"Unknown ({score:.2f})", 
                            "color": "#ff0000", 
                            "status": "unknown"
                        }
                    else:
                        # Voting Logic
                        if track_id != -1:
                            self.add_vote(track_id, name, score)
                            decision, voted_name, avg_score, vote_count = self.get_voting_result(track_id)
                            
                            if decision == "accepted":
                                # Cache confirmed result
                                self.face_tracks[track_id].update({
                                    'confirmed_name': voted_name,
                                    'confirmed_score': avg_score,
                                    'confirmed_time': current_time
                                })
                                result_meta = {
                                    "name": voted_name,
                                    "message": f"{voted_name} - VERIFIED ({vote_count}/{self.voting_frames})",
                                    "color": "#00ff00",
                                    "status": "accepted"
                                }
                            elif decision == "rejected":
                                result_meta = {
                                    "name": voted_name,
                                    "message": f"{voted_name}? - UNCERTAIN ({vote_count}/{self.voting_frames})",
                                    "color": "#ffa500",  # Orange for rejected/uncertain
                                    "status": "rejected"
                                }
                            else:
                                # Pending / Single Frame Logic
                                if score < self.t_low:
                                    status_msg = f"Unknown ({score:.2f})"
                                    color = "#ff0000"
                                elif score > self.t_high:
                                    if is_frontal:
                                        status_msg = f"{name} ({score:.2f}) [{vote_count}/{self.voting_frames}]"
                                        color = "#00ff00"
                                    else:
                                        status_msg = f"{name} - Look straight ({score:.2f})"
                                        color = "#ffa500" # Orange
                                else:
                                    status_msg = f"{name}? Look straight ({score:.2f})"
                                    color = "#ffa500" # Orange
                                
                                result_meta = {
                                    "name": name,
                                    "message": status_msg,
                                    "color": color,
                                    "status": "pending"
                                }
                        else:
                            # Handling if no track_id (should not happen often)
                            result_meta = {"name": name, "message": f"{name} ({score:.2f})", "color": "#00ff00", "status": "pending"}

            # 5. Log Attendance - Strictly following app.py process_frame
            status = result_meta.get("status")
            name = result_meta.get("name")
            new_log = None
            
            if status in ["confirmed", "accepted"] and name:
                # Check attendance cooldown
                if name not in self.last_attendance or (current_time - self.last_attendance[name]) >= self.attendance_cooldown:
                    # Logic: Log attendance ONCE per track if possible, or just based on cooldown
                    # app.py also checks track['attendance_status'] != 'new'
                    track_data = self.face_tracks.get(track_id, {})
                    
                    if track_data.get('attendance_status') != 'new':
                        timestamp = self.attendance.log(name, 1.0, face_img)
                        self.last_attendance[name] = current_time
                        track_data['attendance_status'] = 'new'
                        track_data['attendance_time'] = current_time
                        
                        log_data = {"time": timestamp.split(' ')[1], "name": name, "score": "0.99"} # Simplified score
                        new_log = log_data
                        
                        result_meta['message'] = "ATTENDANCE RECORDED"
                        result_meta['color'] = "#00ff00"
                        
                        # Trigger Door (Server specific action)
                        self.door_open_until = time.time() + 5
                        
                        # Broadcast
                        asyncio.create_task(manager.broadcast({
                            "type": "log", 
                            "data": log_data, 
                            "stats": self.get_stats_dict(),
                            "door": "open"
                        }))
                else:
                    # Already attended recently
                    elapsed = current_time - self.last_attendance[name]
                    remaining = int(self.attendance_cooldown - elapsed)
                    
                    # Store status in track
                    if track_id != -1:
                        track_data = self.face_tracks.get(track_id, {})
                        track_data['attendance_status'] = 'already'
                        track_data['attendance_time'] = current_time
                    
                    result_meta['message'] = f"ALREADY ATTENDED ({remaining}s)"
                    result_meta['color'] = "#808080" # Gray

            results_data.append({
                "track_id": track_id,
                "box": [int(x), int(y), int(w), int(h)],
                "name": result_meta.get("name"),
                "message": result_meta["message"],
                "color": result_meta["color"],
                "liveness": float(liveness_score),
                "log": new_log
            })

        return results_data
    
    def get_stats_dict(self):
        counts = self.attendance.get_today_count()
        return {"total_today": sum(counts.values()), "unique_today": len(counts)}
    
    def get_hourly_stats(self):
        if not os.path.exists(self.attendance.log_file): return {}
        today_str = datetime.now().strftime('%Y-%m-%d')
        hours = {h: 0 for h in range(24)}
        with open(self.attendance.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) > 0 and row[0].startswith(today_str):
                    try: hours[int(row[0].split(' ')[1].split(':')[0])] += 1
                    except: pass
        return hours

state = SystemState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_loop
    main_loop = asyncio.get_running_loop()
    state.load_models()
    asyncio.create_task(send_telegram_alert("System Started"))
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, kiosk_id: str = "Main_Kiosk"):
    await manager.connect(websocket, kiosk_id)
    try:
        while True:
            data = await websocket.receive_bytes()
            door_status = "open" if time.time() < state.door_open_until else "locked"
            results = state.process_image_data(data, websocket)
            await websocket.send_json({
                "type": "result", "faces": results if results else [], "door": door_status
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        pass

@app.get("/api/attendance")
async def get_attendance(): return JSONResponse(content=[])

@app.get("/api/stats")
async def get_stats():
    base = state.get_stats_dict()
    info = {
        **base,
        "kiosks": manager.get_kiosk_stats(),
        "hourly": state.get_hourly_stats(),
        "door_status": "open" if time.time() < state.door_open_until else "locked"
    }
    return JSONResponse(content=info)

# ADMIN
@app.post("/api/login")
async def login(req: LoginRequest):
    if req.password == CONF["ADMIN_PASSWORD"]: return {"status": "success", "token": "admin-token"}
    raise HTTPException(401, "Invalid password")

@app.get("/api/users")
async def get_users(): return state.db.list_persons()

@app.delete("/api/users/{name}")
async def delete_user(name: str):
    if state.db.delete_person(name):
        state.db.save()
        return {"status": "success"}
    raise HTTPException(404, "User not found")

@app.put("/api/users/{name}")
async def rename_user(name: str, req: RenameRequest):
    if name not in state.db.data: raise HTTPException(404)
    data = state.db.data[name]
    state.db.data[req.new_name] = data
    del state.db.data[name]
    state.db.save()
    try: os.rename(os.path.join(state.db.img_dir, name), os.path.join(state.db.img_dir, req.new_name))
    except: pass
    return {"status": "success"}

@app.post("/api/register")
async def register(req: RegisterRequest):
    if not req.images: raise HTTPException(400)
    
    final_faces = []
    
    for s in req.images:
        # Decode base64 to image
        s = s.split(',')[1] if ',' in s else s
        n = np.frombuffer(base64.b64decode(s), np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        
        if img is None: continue
        
        # Detect and crop face (Matches app.py logic)
        # app.py uses default output_size=160
        faces = state.detector.detect_and_crop(img, output_size=160)
        
        if faces:
            # Take the first face found (usually assuming 1 person during enrollment)
            # stored as (bbox, aligned_face)
            _, aligned_face = faces[0]
            final_faces.append(aligned_face)
    
    if not final_faces: 
        raise HTTPException(400, "No faces detected in the samples")
    
    # Extract embeddings from the ALIGNED/CROPPED faces
    embs = state.embedder.extract_multiple(final_faces)
    
    # Check if we got valid embeddings
    valid_embs = [e for e in embs if e is not None]
    valid_faces = [f for f, e in zip(final_faces, embs) if e is not None]
    
    if not valid_embs: 
        raise HTTPException(400, "Could not extract features from faces")
    
    state.db.add_person(req.name, valid_embs, valid_faces)
    return {"status": "success", "count": len(valid_embs)}

@app.get("/api/users/{name}/images")
async def get_images(name: str):
    imgs = state.db.get_person_images(name)
    encoded = []
    for i in imgs:
        encoded.append("data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', i)[1]).decode())
    return {"images": encoded}

@app.post("/api/door/open")
async def open_door(req: DoorControlRequest):
    state.door_open_until = time.time() + req.duration
    asyncio.create_task(send_telegram_alert(f"Door Opened Manually via Admin for {req.duration}s"))
    return {"status": "success", "locked_at": state.door_open_until}

@app.get("/api/full_stats")
async def full_stats():
    if not os.path.exists(state.attendance.log_file): return []
    with open(state.attendance.log_file, 'r', encoding='utf-8') as f:
        return list(csv.reader(f))[1:]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
