import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
from db import EmbeddingDB
from attendance import AttendanceLogger
from collections import deque, Counter
from config import settings, thresholds


class FaceAccessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Access System - Advanced")
        self.root.geometry("1200x700")
        
        self.detector = None
        self.embedder = None
        self.liveness_detector = None
        self.db = EmbeddingDB()
        self.attendance = AttendanceLogger()
        self.models_ready = False
        
        self.cap = None
        self.running = False
        
        # Load from config
        self.t_low = thresholds.T_REJECT
        self.t_high = thresholds.T_ACCEPT
        self.voting_frames = thresholds.VOTING_FRAMES
        self.voting_threshold = thresholds.VOTING_THRESHOLD
        
        self.enroll_mode = False
        self.enroll_name = ""
        self.enroll_samples = []
        self.enroll_faces = []
        self.enroll_target = 15
        
        self.frame_buffer = {}
        self.last_attendance = {}  # Track last attendance time
        self.attendance_cooldown = thresholds.ATTENDANCE_COOLDOWN
        
        # Face tracking
        self.face_tracks = {}
        self.next_track_id = 0
        self.max_track_age = 30  # frames
        self.iou_threshold = 0.3
        
        # Performance optimization (from config)
        self.frame_count = 0
        self.skip_detection_frames = settings.SKIP_DETECTION_FRAMES
        self.last_faces = []  # Cache last detection
        self.last_landmarks = []
        self.fps_target = settings.FPS_TARGET
        self.confirmed_cache_time = settings.CONFIRMED_CACHE_TIME
        
        # Display options
        self.show_landmarks = tk.BooleanVar(value=False)
        self.show_5points = tk.BooleanVar(value=False)
        self.show_aligned = tk.BooleanVar(value=False)
        self.aligned_faces_cache = []
        
        self.setup_ui()
    
    def setup_ui(self):
        left = ttk.Frame(self.root, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left.pack_propagate(False)
        
        ctrl = ttk.LabelFrame(left, text="Control", padding=10)
        ctrl.pack(fill=tk.X, pady=5)
        
        ttk.Label(ctrl, text="Camera:").pack(anchor=tk.W)
        self.cam_entry = ttk.Entry(ctrl)
        self.cam_entry.insert(0, settings.CAMERA_DEFAULT)
        self.cam_entry.pack(fill=tk.X, pady=2)
        
        
        self.btn_start = ttk.Button(ctrl, text="Start Camera", command=self.toggle_camera)
        self.btn_start.pack(fill=tk.X, pady=10)
        
        ttk.Button(ctrl, text="Enroll New", command=self.start_enroll).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Manage Users", command=self.manage_users).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="View Attendance", command=self.view_log).pack(fill=tk.X, pady=2)
        
        # Display options
        display_frame = ttk.LabelFrame(left, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(display_frame, text="Show 468 Landmarks", 
                       variable=self.show_landmarks).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(display_frame, text="Show 5 Key Points", 
                       variable=self.show_5points).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(display_frame, text="Show Aligned Faces", 
                       variable=self.show_aligned).pack(anchor=tk.W, pady=2)
        
        users = ttk.LabelFrame(left, text="Registered Users", padding=5)
        users.pack(fill=tk.BOTH, expand=True, pady=5)
        self.user_list = tk.Listbox(users, height=8)
        self.user_list.pack(fill=tk.BOTH, expand=True)
        self.refresh_users()
        
        log_frame = ttk.LabelFrame(left, text="Recent Attendance", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_frame, height=10, width=30, yscrollcommand=log_scroll.set, 
                                font=("Courier", 8), state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_text.yview)
        
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(right, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.status = ttk.Label(right, text="Ready", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, pady=5)
        
        self.update_log_display()
    

    
    def refresh_users(self):
        self.user_list.delete(0, tk.END)
        self.db.data = self.db.load()
        persons = self.db.list_persons()
        for name in persons:
            self.user_list.insert(tk.END, name)
    
    def draw_landmarks(self, frame, landmarks):
        """Draw 468 landmarks"""
        for point in landmarks:
            cv2.circle(frame, point, 1, (0, 255, 255), -1)
    
    def draw_5_points(self, frame, points):
        """Draw 5 key points"""
        colors = [
            (255, 0, 0),    # Left eye - Blue
            (255, 0, 0),    # Right eye - Blue
            (0, 255, 0),    # Nose - Green
            (0, 0, 255),    # Left mouth - Red
            (0, 0, 255)     # Right mouth - Red
        ]
        labels = ["L_Eye", "R_Eye", "Nose", "L_Mouth", "R_Mouth"]
        
        for i, (point, color, label) in enumerate(zip(points, colors, labels)):
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
            cv2.putText(frame, label, (x+8, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def load_models(self):
        if self.models_ready:
            return True
        self.status.config(text="Loading models...")
        self.root.update()
        try:
            from core.detector import FaceDetector
            from core.embedder import FaceEmbedder
            from core.liveness import load_detector
            
            self.detector = FaceDetector()
            self.embedder = FaceEmbedder()
            
            # Load liveness detector
            self.liveness_detector = load_detector(
                model_path=r'models\2.7_80x80_MiniFASNetV2.pth',
                device='cpu'
            )
            
            self.models_ready = True
            self.status.config(text="Ready (with Liveness)")
            return True
        except Exception as e:
            self.status.config(text=f"Error: {e}")
            return False
    
    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        if not self.load_models():
            return
        
        src = self.cam_entry.get()
        try:
            src = int(src)
        except:
            pass
        
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
        
        # OPTIMIZATION: Lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
        
        self.running = True
        self.btn_start.config(text="Stop")
        self.status.config(text="Running")
        self.process_frame()
    
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.config(text="Start")
        self.status.config(text="Stopped")
        self.canvas.delete("all")
        
        self.frame_buffer.clear()
        self.face_tracks.clear()
        self.next_track_id = 0
        self.last_attendance.clear()
        
        if self.enroll_mode:
            self.enroll_mode = False
            self.enroll_samples = []
            self.enroll_faces = []
    
    def process_frame(self):
        if not self.running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.process_frame)
            return
        
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        self.frame_count += 1
        
        # OPTIMIZATION: Skip detection on some frames, use cached results
        if self.frame_count % (self.skip_detection_frames + 1) == 0 or not self.last_faces:
            faces = self.detector.detect_and_crop(frame)
            landmarks_list = self.detector.get_landmarks(frame)
            self.last_faces = faces
            self.last_landmarks = landmarks_list
        else:
            # Use cached detection
            faces = self.last_faces
            landmarks_list = self.last_landmarks
        
        # Update face tracking
        face_assignments = self.update_face_tracks(faces)
        
        granted_names = []
        self.aligned_faces_cache = []  # Clear cache for this frame
        
        for idx, (box, face_img) in enumerate(faces):
            x, y, w, h = box
            
            # Get landmarks for this face
            landmarks = None
            if idx < len(landmarks_list):
                landmarks = landmarks_list[idx]
            
            # Draw landmarks if enabled
            if landmarks and self.show_landmarks.get():
                self.draw_landmarks(frame, landmarks)
            
            # Draw 5 key points if enabled
            if landmarks and self.show_5points.get():
                points_5 = self.detector.get_5_points(landmarks)
                if points_5 is not None:
                    self.draw_5_points(frame, points_5)
            
            # Get aligned face if enabled
            if landmarks and self.show_aligned.get():
                aligned = self.detector.align_face(frame, landmarks, output_size=160)
                if aligned is not None:
                    self.aligned_faces_cache.append((idx, aligned))
            
            if self.enroll_mode:
                color = (255, 165, 0)
                label = f"Enrolling: {len(self.enroll_samples)}/{self.enroll_target}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                continue
            
            # LIVENESS CHECK - Kiểm tra trước khi nhận diện
            liveness_score, liveness_label = self.liveness_detector.predict(frame, box)
            
            # Nếu là fake face, hiển thị cảnh báo và bỏ qua nhận diện
            if liveness_label == 'fake':
                color = (0, 0, 255)  # Red
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, "FAKE FACE DETECTED!", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Liveness: {liveness_score:.3f}", (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw warning icon
                cv2.putText(frame, "⚠", (x + w - 30, y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                continue
            
            is_frontal = False
            if idx < len(landmarks_list):
                landmarks = landmarks_list[idx]
                is_frontal = self.detector.is_frontal_face(landmarks)
            
            # Get actual track ID from assignments
            track_id = face_assignments.get(idx, None)
            result = self.recognize_frame(face_img, is_frontal, track_id=track_id, liveness_score=liveness_score)
            
            color = result["color"]
            message = result["message"]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, message, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display liveness score (green checkmark for real face)
            liveness_text = f"✓ Live: {liveness_score:.3f}"
            cv2.putText(frame, liveness_text, (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            if result["voting_info"]:
                voting = result["voting_info"]
                vote_text = f"{voting['vote_count']}/{voting['total_frames']}"
                cv2.putText(frame, vote_text, (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                if voting["decision"] == "accepted":
                    name = result["name"]
                    
                    # Check attendance cooldown
                    if name not in self.last_attendance or (current_time - self.last_attendance[name]) >= self.attendance_cooldown:
                        # Record attendance ONCE
                        if track_id is not None:
                            track_data = self.face_tracks.get(track_id, {})
                            if track_data.get('attendance_status') != 'new':
                                # First time recording for this track
                                self.attendance.log(name, result["score"], face_img)
                                self.last_attendance[name] = current_time
                                self.update_log_display()
                                
                                # Mark track as attended
                                track_data['attendance_status'] = 'new'
                                track_data['attendance_time'] = current_time
                        
                        granted_names.append((name, "new", track_id))
                        
                        # Green border for new attendance
                        cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 5)
                        cv2.putText(frame, "ATTENDANCE RECORDED", (x, y + h + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # Already attended recently
                        elapsed = current_time - self.last_attendance[name]
                        remaining = int(self.attendance_cooldown - elapsed)
                        
                        # Mark track status
                        if track_id is not None:
                            track_data = self.face_tracks.get(track_id, {})
                            track_data['attendance_status'] = 'already'
                            track_data['attendance_time'] = current_time
                        
                        granted_names.append((name, "already", remaining, track_id))
                        
                        # Gray border for already attended
                        cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), (150, 150, 150), 3)
                        cv2.putText(frame, f"Already attended ({remaining}s)", (x, y + h + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)
        
        # Display large notification at bottom right (persistent while face in frame)
        h, w = frame.shape[:2]
        notification_index = 0
        
        for track_id, track_data in self.face_tracks.items():
            attendance_status = track_data.get('attendance_status')
            if not attendance_status:
                continue
            
            # Check if track is still active (seen recently)
            if (current_time - track_data.get('last_seen', 0)) > 1.0:
                continue
            
            name = track_data.get('confirmed_name')
            if not name:
                continue
            
            box_w, box_h = 400, 110
            box_x = w - box_w - 20
            # Position at bottom edge with small margin
            box_y = h - box_h - 10 - (notification_index * (box_h + 10))
            
            if attendance_status == 'new':
                # OPTIMIZED: Direct rectangle drawing without overlay copy
                # Green background with alpha
                sub_img = frame[box_y:box_y+box_h, box_x:box_x+box_w]
                green_rect = np.full_like(sub_img, (0, 200, 0), dtype=np.uint8)
                cv2.addWeighted(green_rect, 0.7, sub_img, 0.3, 0, sub_img)
                
                # Border
                cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 3)
                
                # Text
                cv2.putText(frame, name, (box_x + 20, box_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(frame, "ATTENDANCE RECORDED", (box_x + 20, box_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                notification_index += 1
                
            elif attendance_status == 'already':
                # Calculate remaining time
                if name in self.last_attendance:
                    elapsed = current_time - self.last_attendance[name]
                    remaining = int(self.attendance_cooldown - elapsed)
                    
                    # OPTIMIZED: Direct rectangle drawing
                    sub_img = frame[box_y:box_y+box_h, box_x:box_x+box_w]
                    gray_rect = np.full_like(sub_img, (100, 100, 100), dtype=np.uint8)
                    cv2.addWeighted(gray_rect, 0.7, sub_img, 0.3, 0, sub_img)
                    
                    # Border
                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (150, 150, 150), 3)
                    
                    # Text
                    cv2.putText(frame, name, (box_x + 20, box_y + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(frame, f"ALREADY ATTENDED ({remaining}s)", (box_x + 20, box_y + 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    notification_index += 1
        
        self.cleanup_old_tracks(max_age=3.0)
        
        if self.enroll_mode:
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        self.show_frame(frame)
        self.root.after(self.fps_target, self.process_frame)
    
    def add_vote(self, track_id, name, score):
        """Add a vote for temporal voting"""
        if track_id not in self.frame_buffer:
            self.frame_buffer[track_id] = deque(maxlen=self.voting_frames)
        
        self.frame_buffer[track_id].append({
            "name": name,
            "score": score,
            "time": time.time()
        })
    
    def get_voting_result(self, track_id):
        """Get voting result from frame buffer with strict consistency check"""
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
        
        # Accept if:
        # 1. Enough votes (e.g., 4/5 = 80%)
        # 2. Good average score
        # 3. Most scores above threshold (allow 1-2 outliers)
        # 4. Reasonable variance
        # 5. Good consistency ratio
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
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_face_tracks(self, current_faces):
        """Update face tracks with current detections"""
        current_time = time.time()
        
        # Age all existing tracks
        for track_id in list(self.face_tracks.keys()):
            self.face_tracks[track_id]['age'] += 1
            if self.face_tracks[track_id]['age'] > self.max_track_age:
                del self.face_tracks[track_id]
                if track_id in self.frame_buffer:
                    del self.frame_buffer[track_id]
        
        # Match current faces with existing tracks
        matched_tracks = set()
        face_assignments = {}
        
        for face_idx, (box, face_img) in enumerate(current_faces):
            best_track_id = None
            best_iou = 0
            
            for track_id, track_data in self.face_tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self.calculate_iou(box, track_data['last_box'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.face_tracks[best_track_id].update({
                    'last_box': box,
                    'age': 0,
                    'last_seen': current_time
                })
                matched_tracks.add(best_track_id)
                face_assignments[face_idx] = best_track_id
            else:
                # Create new track
                new_track_id = self.next_track_id
                self.next_track_id += 1
                
                self.face_tracks[new_track_id] = {
                    'last_box': box,
                    'age': 0,
                    'last_seen': current_time,
                    'confirmed_name': None,
                    'confirmed_score': 0.0,
                    'confirmed_time': None,
                    'skip_recognition': False,
                    'attendance_status': None,  # 'new' or 'already'
                    'attendance_time': None
                }
                face_assignments[face_idx] = new_track_id
        
        return face_assignments
    
    def cleanup_old_tracks(self, max_age=5.0):
        """Remove old tracks that haven't been updated"""
        current_time = time.time()
        to_remove = []
        
        for track_id, buffer in self.frame_buffer.items():
            if buffer and (current_time - buffer[-1]["time"]) > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.frame_buffer[track_id]
        
        # Also cleanup face tracks
        to_remove_tracks = []
        for track_id, track_data in self.face_tracks.items():
            if (current_time - track_data['last_seen']) > max_age:
                to_remove_tracks.append(track_id)
        
        for track_id in to_remove_tracks:
            del self.face_tracks[track_id]
    
    def recognize_frame(self, face_img, is_frontal=True, track_id=None, liveness_score=1.0):
        """Recognize face from a single frame with temporal voting and tracking"""
        
        # OPTIMIZATION: Check if this track is already confirmed and skip expensive recognition
        if track_id is not None and track_id in self.face_tracks:
            track_data = self.face_tracks[track_id]
            
            # If already confirmed and not too old, return cached result WITHOUT extraction
            if (track_data['confirmed_name'] is not None and 
                track_data['confirmed_time'] is not None and
                (time.time() - track_data['confirmed_time']) < self.confirmed_cache_time):
                
                # Return cached result - NO embedding extraction needed!
                return {
                    "name": track_data['confirmed_name'],
                    "score": track_data['confirmed_score'],
                    "status": "confirmed",
                    "message": f"{track_data['confirmed_name']} - CONFIRMED",
                    "color": (0, 255, 0),
                    "voting_info": {
                        "decision": "accepted",  # Changed to "accepted" for attendance logic
                        "voted_name": track_data['confirmed_name'],
                        "avg_score": track_data['confirmed_score'],
                        "vote_count": self.voting_frames,
                        "total_frames": self.voting_frames
                    },
                    "liveness_score": liveness_score
                }
        
        # Normal recognition process (only for new/unconfirmed tracks)
        emb = self.embedder.extract(face_img, check_quality=True)  # Skip quality check for speed
        if not emb:
            return {
                "name": None,
                "score": 0.0,
                "status": "error",
                "message": "Error",
                "color": (128, 128, 128),
                "voting_info": None
            }
        
        name, score = self.match(emb)
        
        if name is None:
            return {
                "name": None,
                "score": score,
                "status": "unknown",
                "message": f"Unknown ({score:.2f})",
                "color": (0, 0, 255),
                "voting_info": None
            }
        
        if track_id is not None:
            self.add_vote(track_id, name, score)
            decision, voted_name, avg_score, vote_count = self.get_voting_result(track_id)
            
            voting_info = {
                "decision": decision,
                "voted_name": voted_name,
                "avg_score": avg_score,
                "vote_count": vote_count,
                "total_frames": len(self.frame_buffer.get(track_id, []))
            }
            
            if decision == "accepted":
                # Cache the confirmed result in track
                if track_id in self.face_tracks:
                    self.face_tracks[track_id].update({
                        'confirmed_name': voted_name,
                        'confirmed_score': avg_score,
                        'confirmed_time': time.time()
                    })
                
                return {
                    "name": voted_name,
                    "score": avg_score,
                    "status": "accepted",
                    "message": f"{voted_name} - VERIFIED ({vote_count}/{self.voting_frames})",
                    "color": (0, 255, 0),
                    "voting_info": voting_info
                }
            elif decision == "rejected":
                return {
                    "name": voted_name,
                    "score": avg_score,
                    "status": "rejected",
                    "message": f"{voted_name}? - UNCERTAIN ({vote_count}/{self.voting_frames})",
                    "color": (0, 100, 255),
                    "voting_info": voting_info
                }
            else:
                if score < self.t_low:
                    status_msg = f"Unknown ({score:.2f})"
                    color = (0, 0, 255)
                elif score > self.t_high:
                    if is_frontal:
                        status_msg = f"{name} ({score:.2f}) [{vote_count}/{self.voting_frames}]"
                        color = (0, 255, 0)
                    else:
                        status_msg = f"{name} - Look straight ({score:.2f})"
                        color = (0, 165, 255)
                else:
                    status_msg = f"{name}? Look straight ({score:.2f})"
                    color = (0, 165, 255)
                
                return {
                    "name": name,
                    "score": score,
                    "status": "pending",
                    "message": status_msg,
                    "color": color,
                    "voting_info": voting_info
                }
        else:
            if score < self.t_low:
                status_msg = f"Unknown ({score:.2f})"
                color = (0, 0, 255)
            elif score > self.t_high:
                status_msg = f"{name} ({score:.2f})"
                color = (0, 255, 0)
            else:
                status_msg = f"{name}? ({score:.2f})"
                color = (0, 165, 255)
            
            return {
                "name": name,
                "score": score,
                "status": "single",
                "message": status_msg,
                "color": color,
                "voting_info": None
            }
    
    def update_log_display(self):
        recent = self.attendance.get_recent(15)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        
        for row in reversed(recent):
            if len(row) >= 3:
                time_str = row[0].split()[1] if ' ' in row[0] else row[0]
                name = row[1]
                score = row[2]
                self.log_text.insert(tk.END, f"{time_str} {name}\n")
        
        self.log_text.config(state=tk.DISABLED)
    

    
    def view_log(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Attendance Log")
        dialog.geometry("700x500")
        dialog.transient(self.root)
        
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Attendance History", font=("Arial", 12, "bold")).pack(pady=5)
        
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree = ttk.Treeview(tree_frame, columns=("Time", "Name", "Score", "Image"), 
                           show="headings", yscrollcommand=scrollbar.set)
        tree.heading("Time", text="Timestamp")
        tree.heading("Name", text="Name")
        tree.heading("Score", text="Score")
        tree.heading("Image", text="Image")
        
        tree.column("Time", width=150)
        tree.column("Name", width=150)
        tree.column("Score", width=80)
        tree.column("Image", width=250)
        
        tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=tree.yview)
        
        recent = self.attendance.get_recent(100)
        for row in reversed(recent):
            if len(row) >= 4:
                tree.insert("", 0, values=row)
        
        counts = self.attendance.get_today_count()
        if counts:
            stats_text = "Today: " + ", ".join([f"{name}({count})" for name, count in counts.items()])
            ttk.Label(frame, text=stats_text).pack(pady=5)
        
        ttk.Button(frame, text="Close", command=dialog.destroy).pack(pady=5)
    
    def show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 1 and ch > 1:
            iw, ih = img.size
            scale = min(cw/iw, ch/ih)
            img = img.resize((int(iw*scale), int(ih*scale)), Image.Resampling.BILINEAR)
        
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.photo, anchor=tk.CENTER)
        
        # Display aligned faces if enabled
        if self.show_aligned.get() and self.aligned_faces_cache:
            self.display_aligned_faces()
    
    def display_aligned_faces(self):
        """Display aligned faces in bottom right corner"""
        if not self.aligned_faces_cache:
            return
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        # Display parameters
        face_size = 120
        padding = 10
        start_x = cw - face_size - padding
        start_y = ch - padding
        
        # Display up to 3 faces vertically
        for i, (idx, aligned_face) in enumerate(self.aligned_faces_cache[:3]):
            y_pos = start_y - (i + 1) * (face_size + padding)
            
            # Convert to PhotoImage
            face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(face_rgb)
            face_img = face_img.resize((face_size, face_size), Image.Resampling.BILINEAR)
            face_photo = ImageTk.PhotoImage(face_img)
            
            # Draw on canvas
            self.canvas.create_image(start_x, y_pos, image=face_photo, anchor=tk.NW)
            
            # Draw border
            self.canvas.create_rectangle(
                start_x, y_pos, 
                start_x + face_size, y_pos + face_size,
                outline='#00ff00', width=2
            )
            
            # Draw label
            self.canvas.create_text(
                start_x + face_size // 2, y_pos + face_size + 5,
                text=f"Face {idx + 1}",
                fill='white', font=('Arial', 10, 'bold')
            )
            
            # Keep reference to prevent garbage collection
            if not hasattr(self, 'aligned_photos'):
                self.aligned_photos = []
            self.aligned_photos.append(face_photo)
    
    def match(self, embedding):
        """Match embedding using margin-based decision"""
        db = self.db.get_all()
        name, decision, s1, s2, margin = self.embedder.match_with_margin(embedding, db)
        
        # Convert decision to old format for compatibility
        if decision == "accept":
            return name, s1
        elif decision == "uncertain":
            return name, s1  # Let voting decide
        else:  # unknown
            return None, s1
    
    def start_enroll(self):
        if not self.load_models():
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Enroll")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Name:").pack(pady=10)
        name_entry = ttk.Entry(dialog, width=30)
        name_entry.pack()
        name_entry.focus()
        
        def on_camera():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Enter name")
                return
            self.enroll_name = name
            self.enroll_samples = []
            self.enroll_faces = []
            self.enroll_mode = True
            dialog.destroy()
            
            self.root.bind("<space>", self.capture_sample)
            self.root.bind("<Escape>", lambda e: self.cancel_enroll())
            
            if not self.running:
                self.start_camera()
            self.status.config(text=f"Enrolling: {name}")
        
        def on_files():
            name = name_entry.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Enter name")
                return
            files = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.png *.jpeg")])
            if files:
                dialog.destroy()
                self.enroll_from_files(name, files)
        
        btns = ttk.Frame(dialog)
        btns.pack(pady=20)
        ttk.Button(btns, text="Camera", command=on_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="Files", command=on_files).pack(side=tk.LEFT, padx=5)
    
    def capture_sample(self, event=None):
        if not self.enroll_mode or not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        faces = self.detector.detect_and_crop(frame)
        
        if faces:
            _, face_img = faces[0]
            emb = self.embedder.extract(face_img, check_quality=True)
            if emb:
                self.enroll_samples.append(emb)
                self.enroll_faces.append(face_img.copy())
                self.status.config(text=f"Captured {len(self.enroll_samples)}/{self.enroll_target}")
                
                if len(self.enroll_samples) >= self.enroll_target:
                    self.finish_enroll()
    
    def cancel_enroll(self):
        self.root.unbind("<space>")
        self.root.unbind("<Escape>")
        self.enroll_mode = False
        self.enroll_samples = []
        self.enroll_faces = []
        self.status.config(text="Cancelled")
    
    def finish_enroll(self):
        self.root.unbind("<space>")
        self.root.unbind("<Escape>")
        self.enroll_mode = False
        
        if self.enroll_samples:
            self.db.add_person(self.enroll_name, self.enroll_samples, self.enroll_faces)
            self.refresh_users()
            messagebox.showinfo("Success", f"Enrolled {self.enroll_name} with {len(self.enroll_samples)} samples")
        
        self.enroll_samples = []
        self.enroll_faces = []
        self.status.config(text="Running")
    
    def enroll_from_files(self, name, files):
        samples = []
        face_imgs = []
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            faces = self.detector.detect_and_crop(img)
            for _, face_img in faces:
                emb = self.embedder.extract(face_img, check_quality=True)
                if emb:
                    samples.append(emb)
                    face_imgs.append(face_img.copy())
        
        if samples:
            self.db.add_person(name, samples, face_imgs)
            self.refresh_users()
            messagebox.showinfo("Success", f"Enrolled {name} with {len(samples)} samples")
        else:
            messagebox.showerror("Error", "No faces found")
    
    def manage_users(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Manage Users")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        
        left_frame = ttk.Frame(dialog)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(left_frame, text="Users:").pack(anchor=tk.W)
        listbox = tk.Listbox(left_frame)
        listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        for name in self.db.list_persons():
            listbox.insert(tk.END, name)
        
        right_frame = ttk.Frame(dialog)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(right_frame, text="Face Samples:").pack(anchor=tk.W)
        img_canvas = tk.Canvas(right_frame, bg="gray")
        img_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        def show_faces(event=None):
            sel = listbox.curselection()
            if not sel:
                return
            name = listbox.get(sel[0])
            images = self.db.get_person_images(name)
            
            img_canvas.delete("all")
            if not images:
                img_canvas.create_text(150, 100, text="No images saved", fill="white")
                return
            
            cols = 3
            size = 80
            for i, img in enumerate(images[:9]):
                row = i // cols
                col = i % cols
                x = col * (size + 10) + 10
                y = row * (size + 10) + 10
                
                img_resized = cv2.resize(img, (size, size))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(img_rgb))
                img_canvas.create_image(x, y, image=photo, anchor=tk.NW)
                img_canvas.image_refs = getattr(img_canvas, 'image_refs', [])
                img_canvas.image_refs.append(photo)
        
        listbox.bind('<<ListboxSelect>>', show_faces)
        
        def delete():
            sel = listbox.curselection()
            if sel:
                name = listbox.get(sel[0])
                if messagebox.askyesno("Confirm", f"Delete {name}?"):
                    self.db.remove_person(name)
                    listbox.delete(sel[0])
                    self.refresh_users()
                    img_canvas.delete("all")
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Delete", command=delete).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=dialog.destroy).pack(side=tk.LEFT, padx=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAccessApp(root)
    root.mainloop()
