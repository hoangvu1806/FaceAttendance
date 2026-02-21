import sys
from pathlib import Path

# Add parent directory to path so we can import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from core.detector import FaceDetector


class AlignmentTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Alignment Test - Attendance System")
        self.root.geometry("1400x800")
        
        # Variables
        self.camera = None
        self.is_running = False
        self.detector = None
        self.current_frame = None
        
        # Settings
        self.show_landmarks = tk.BooleanVar(value=True)
        self.show_5points = tk.BooleanVar(value=True)
        self.show_bbox = tk.BooleanVar(value=True)
        self.show_pose = tk.BooleanVar(value=True)
        self.show_aligned = tk.BooleanVar(value=True)
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        self.setup_ui()
        self.load_detector()
        
    def setup_ui(self):
        """Thiết lập giao diện"""
        # Top frame - Controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Camera selection
        ttk.Label(control_frame, text="Camera:").pack(side=tk.LEFT, padx=5)
        self.camera_var = tk.IntVar(value=0)
        camera_combo = ttk.Combobox(control_frame, textvariable=self.camera_var,
                                    values=[0, 1, 2, 3],
                                    state="readonly", width=8)
        camera_combo.pack(side=tk.LEFT, padx=5)
        
        # Refresh camera button
        refresh_btn = ttk.Button(control_frame, text="🔄", width=3,
                                command=self.detect_cameras)
        refresh_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Buttons
        self.start_btn = ttk.Button(control_frame, text="▶ Start", 
                                    command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", 
                                   command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Display options
        ttk.Checkbutton(control_frame, text="BBox", 
                       variable=self.show_bbox).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(control_frame, text="468 Landmarks", 
                       variable=self.show_landmarks).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(control_frame, text="5 Points", 
                       variable=self.show_5points).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(control_frame, text="Pose", 
                       variable=self.show_pose).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(control_frame, text="Aligned", 
                       variable=self.show_aligned).pack(side=tk.LEFT, padx=5)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Left: Original video
        left_frame = ttk.LabelFrame(main_frame, text="Original Video", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.canvas_original = tk.Canvas(left_frame, width=640, height=480, bg='black')
        self.canvas_original.pack()
        
        # Right: Info and aligned faces
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(right_frame, text="Detection Info", padding="10")
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # FPS
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X)
        
        ttk.Label(info_grid, text="FPS:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.fps_label = ttk.Label(info_grid, text="0.0", font=('Arial', 10, 'bold'))
        self.fps_label.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_grid, text="Faces:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.face_count_label = ttk.Label(info_grid, text="0", font=('Arial', 10))
        self.face_count_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_grid, text="Frontal:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.frontal_label = ttk.Label(info_grid, text="-", font=('Arial', 10))
        self.frontal_label.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Pose info
        ttk.Label(info_grid, text="Yaw:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.yaw_label = ttk.Label(info_grid, text="-")
        self.yaw_label.grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_grid, text="Pitch:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.pitch_label = ttk.Label(info_grid, text="-")
        self.pitch_label.grid(row=4, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_grid, text="Roll:").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.roll_label = ttk.Label(info_grid, text="-")
        self.roll_label.grid(row=5, column=1, sticky=tk.W, pady=2)
        
        # Aligned faces panel
        aligned_frame = ttk.LabelFrame(right_frame, text="Aligned Faces (160x160)", padding="10")
        aligned_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        self.canvas_aligned = tk.Canvas(aligned_frame, width=500, height=500, bg='gray20')
        self.canvas_aligned.pack(fill=tk.BOTH, expand=True)
        
        # Bottom status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_bar = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
        
    def detect_cameras(self):
        """Phát hiện cameras có sẵn"""
        self.status_bar.config(text="Detecting cameras...")
        available_cameras = []
        
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            # Update combobox
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Combobox):
                            if child.cget('textvariable') == str(self.camera_var):
                                child['values'] = available_cameras
                                break
            self.status_bar.config(text=f"Found cameras: {available_cameras}")
        else:
            self.status_bar.config(text="No cameras found!")
    
    def load_detector(self):
        """Load face detector"""
        try:
            self.status_bar.config(text="Loading face detector...")
            self.detector = FaceDetector(min_detection_confidence=0.7)
            self.status_bar.config(text="Face detector loaded - Ready")
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")
    
    def start_camera(self):
        """Bắt đầu camera"""
        try:
            camera_index = self.camera_var.get()
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {camera_index}")
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.status_bar.config(text=f"Camera {camera_index} started")
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.update_frame, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")
    
    def stop_camera(self):
        """Dừng camera"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="Camera stopped")
    
    def draw_landmarks(self, frame, landmarks):
        """Vẽ 468 landmarks"""
        for point in landmarks:
            cv2.circle(frame, point, 1, (0, 255, 0), -1)
    
    def draw_5_points(self, frame, points):
        """Vẽ 5 điểm chính"""
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
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)
            cv2.putText(frame, label, (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def update_frame(self):
        """Update frame từ camera"""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Detect faces
            faces = self.detector.detect(frame)
            landmarks_list = self.detector.get_landmarks(frame)
            
            # Update face count
            self.face_count_label.config(text=str(len(faces)))
            
            aligned_faces = []
            
            # Process each face
            for i, bbox in enumerate(faces):
                x, y, w, h = bbox
                
                # Match landmarks
                landmarks = self.detector.match_landmarks_to_bbox(bbox, landmarks_list)
                
                if landmarks:
                    # Draw bbox
                    if self.show_bbox.get():
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Face {i+1}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw 468 landmarks
                    if self.show_landmarks.get():
                        self.draw_landmarks(frame, landmarks)
                    
                    # Get and draw 5 points
                    points_5 = self.detector.get_5_points(landmarks)
                    if points_5 is not None and self.show_5points.get():
                        self.draw_5_points(frame, points_5)
                    
                    # Get pose
                    pose = self.detector.get_head_pose(landmarks)
                    if pose and self.show_pose.get():
                        # Draw pose info
                        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
                        pose_text = f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}"
                        cv2.putText(frame, pose_text, (x, y+h+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # Update labels (only for first face)
                        if i == 0:
                            self.yaw_label.config(text=f"{yaw:.1f}°")
                            self.pitch_label.config(text=f"{pitch:.1f}°")
                            self.roll_label.config(text=f"{roll:.1f}°")
                            
                            # Check if frontal
                            is_frontal = self.detector.is_frontal_face(landmarks)
                            self.frontal_label.config(
                                text="✓ Yes" if is_frontal else "✗ No",
                                foreground='green' if is_frontal else 'red'
                            )
                    
                    # Align face
                    if self.show_aligned.get():
                        aligned = self.detector.align_face(frame, landmarks, output_size=160)
                        if aligned is not None:
                            aligned_faces.append(aligned)
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.fps_label.config(text=f"{self.fps:.1f}")
                self.frame_count = 0
                self.last_time = current_time
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display original frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            photo = ImageTk.PhotoImage(image=img)
            
            self.canvas_original.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_original.image = photo
            
            # Display aligned faces
            if aligned_faces:
                self.display_aligned_faces(aligned_faces)
            
            time.sleep(0.01)
    
    def display_aligned_faces(self, aligned_faces):
        """Hiển thị các aligned faces"""
        self.canvas_aligned.delete("all")
        
        # Calculate grid layout
        n_faces = len(aligned_faces)
        cols = min(3, n_faces)
        rows = (n_faces + cols - 1) // cols
        
        padding = 10
        face_size = 160
        
        for i, face in enumerate(aligned_faces):
            row = i // cols
            col = i % cols
            
            x = padding + col * (face_size + padding)
            y = padding + row * (face_size + padding)
            
            # Convert to PhotoImage
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(face_rgb)
            photo = ImageTk.PhotoImage(image=img)
            
            # Draw on canvas
            self.canvas_aligned.create_image(x, y, anchor=tk.NW, image=photo)
            self.canvas_aligned.create_text(x + face_size//2, y + face_size + 5,
                                           text=f"Face {i+1}",
                                           fill="white", font=('Arial', 10, 'bold'))
            
            # Keep reference
            if not hasattr(self, 'aligned_photos'):
                self.aligned_photos = []
            self.aligned_photos.append(photo)
    
    def on_closing(self):
        """Xử lý khi đóng app"""
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AlignmentTestApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
