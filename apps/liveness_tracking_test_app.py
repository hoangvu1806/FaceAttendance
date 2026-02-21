import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from core.detector import FaceDetector
from core.liveness import load_detector_with_tracking
from config.liveness import (
    LIVENESS_MODEL_PATH,
    LIVENESS_VOTING_METHOD,
    LIVENESS_MAX_HISTORY,
    LIVENESS_MIN_SAMPLES,
    LIVENESS_TRACK_TIMEOUT
)


class LivenessTrackingTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Liveness Detection with Tracking & Voting")
        self.root.geometry("1600x900")
        
        # Variables
        self.camera = None
        self.is_running = False
        self.face_detector = None
        self.liveness_detector = None
        
        # Settings
        self.show_bbox = tk.BooleanVar(value=True)
        self.show_tracking_id = tk.BooleanVar(value=True)
        self.show_vote_info = tk.BooleanVar(value=True)
        self.voting_method = tk.StringVar(value=LIVENESS_VOTING_METHOD)
        
        # Stats
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Track colors (for visualization)
        self.track_colors = {}
        
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        """Setup UI"""
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
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Buttons
        self.start_btn = ttk.Button(control_frame, text="▶ Start", 
                                    command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", 
                                   command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(control_frame, text="🔄 Reset Tracking",
                                    command=self.reset_tracking)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Voting method
        ttk.Label(control_frame, text="Voting:").pack(side=tk.LEFT, padx=5)
        voting_combo = ttk.Combobox(control_frame, textvariable=self.voting_method,
                                    values=['majority', 'weighted', 'confidence_threshold'],
                                    state="readonly", width=18)
        voting_combo.pack(side=tk.LEFT, padx=5)
        voting_combo.bind('<<ComboboxSelected>>', self.on_voting_method_changed)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Display options
        ttk.Checkbutton(control_frame, text="BBox", 
                       variable=self.show_bbox).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(control_frame, text="Track ID", 
                       variable=self.show_tracking_id).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(control_frame, text="Vote Info", 
                       variable=self.show_vote_info).pack(side=tk.LEFT, padx=5)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Left: Video
        left_frame = ttk.LabelFrame(main_frame, text="Live Video", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.canvas_video = tk.Canvas(left_frame, width=800, height=600, bg='black')
        self.canvas_video.pack()
        
        # Right: Info
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # System info
        sys_frame = ttk.LabelFrame(right_frame, text="System Info", padding="10")
        sys_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        info_grid = ttk.Frame(sys_frame)
        info_grid.pack(fill=tk.X)
        
        ttk.Label(info_grid, text="FPS:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.fps_label = ttk.Label(info_grid, text="0.0", font=('Arial', 10, 'bold'))
        self.fps_label.grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_grid, text="Active Tracks:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.tracks_label = ttk.Label(info_grid, text="0", font=('Arial', 10))
        self.tracks_label.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Tracks info
        tracks_frame = ttk.LabelFrame(right_frame, text="Active Tracks", padding="10")
        tracks_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # Create scrollable text widget
        scroll = ttk.Scrollbar(tracks_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tracks_text = tk.Text(tracks_frame, height=20, width=50,
                                   yscrollcommand=scroll.set,
                                   font=('Courier', 9))
        self.tracks_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.tracks_text.yview)
        
        # Configure text tags for colors
        self.tracks_text.tag_config('real', foreground='green', font=('Courier', 9, 'bold'))
        self.tracks_text.tag_config('fake', foreground='red', font=('Courier', 9, 'bold'))
        self.tracks_text.tag_config('header', foreground='blue', font=('Courier', 9, 'bold'))
        
        # Bottom status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_bar = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
        
    def load_models(self):
        """Load models"""
        try:
            self.status_bar.config(text="Loading models...")
            
            # Load face detector
            self.face_detector = FaceDetector(min_detection_confidence=0.7)
            
            # Load liveness detector with tracking
            self.liveness_detector = load_detector_with_tracking(
                model_path=LIVENESS_MODEL_PATH,
                device='cpu',  # Use CPU for compatibility
                max_history=LIVENESS_MAX_HISTORY,
                voting_method=self.voting_method.get(),
                min_samples_for_voting=LIVENESS_MIN_SAMPLES
            )
            
            self.status_bar.config(text="Models loaded - Ready")
        except Exception as e:
            self.status_bar.config(text=f"Error loading models: {str(e)}")
    
    def on_voting_method_changed(self, event=None):
        """Handle voting method change"""
        if self.liveness_detector:
            self.liveness_detector.tracker.voting_method = self.voting_method.get()
            self.status_bar.config(text=f"Voting method changed to: {self.voting_method.get()}")
    
    def reset_tracking(self):
        """Reset all tracks"""
        if self.liveness_detector:
            self.liveness_detector.reset_tracking()
            self.track_colors.clear()
            self.status_bar.config(text="Tracking reset")
    
    def start_camera(self):
        """Start camera"""
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
        """Stop camera"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="Camera stopped")
    
    def get_track_color(self, track_id):
        """Get consistent color for a track"""
        if track_id not in self.track_colors:
            # Generate random color
            np.random.seed(track_id)
            color = tuple(np.random.randint(100, 255, 3).tolist())
            self.track_colors[track_id] = color
        return self.track_colors[track_id]
    
    def update_frame(self):
        """Update frame from camera"""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.face_detector.detect(frame)
            
            # Predict liveness with tracking
            if faces:
                results = self.liveness_detector.predict_with_tracking(frame, faces)
                
                # Update tracks count
                self.tracks_label.config(text=str(len(results)))
                
                # Update tracks info text
                self.update_tracks_info(results)
                
                # Draw results
                for track_id, bbox, is_real, confidence, vote_info in results:
                    x, y, w, h = bbox
                    
                    # Get track color
                    color = self.get_track_color(track_id)
                    
                    # Draw bbox
                    if self.show_bbox.get():
                        # Thicker border for real faces
                        thickness = 3 if is_real else 2
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Draw track ID
                    if self.show_tracking_id.get():
                        label = f"ID:{track_id}"
                        cv2.putText(frame, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw liveness result
                    result_text = "REAL" if is_real else "FAKE"
                    result_color = (0, 255, 0) if is_real else (0, 0, 255)
                    cv2.putText(frame, result_text, (x, y+h+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
                    
                    # Draw confidence
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(frame, conf_text, (x, y+h+50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
                    
                    # Draw vote info
                    if self.show_vote_info.get():
                        stability = vote_info.get('stability', 0)
                        age = vote_info.get('age', 0)
                        info_text = f"S:{stability:.2f} A:{age}"
                        cv2.putText(frame, info_text, (x, y+h+75),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                self.tracks_label.config(text="0")
                self.tracks_text.delete(1.0, tk.END)
            
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
            
            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((800, 600))
            photo = ImageTk.PhotoImage(image=img)
            
            self.canvas_video.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_video.image = photo
            
            time.sleep(0.01)
    
    def update_tracks_info(self, results):
        """Update tracks information display"""
        self.tracks_text.delete(1.0, tk.END)
        
        for track_id, bbox, is_real, confidence, vote_info in results:
            # Header
            self.tracks_text.insert(tk.END, f"Track {track_id}\n", 'header')
            
            # Result
            result_text = "  Result: "
            self.tracks_text.insert(tk.END, result_text)
            self.tracks_text.insert(tk.END, f"{'REAL' if is_real else 'FAKE'}\n",
                                   'real' if is_real else 'fake')
            
            # Confidence
            self.tracks_text.insert(tk.END, f"  Confidence: {confidence:.3f}\n")
            
            # Stability
            stability = vote_info.get('stability', 0)
            self.tracks_text.insert(tk.END, f"  Stability: {stability:.3f}\n")
            
            # Age
            age = vote_info.get('age', 0)
            self.tracks_text.insert(tk.END, f"  Age: {age} frames\n")
            
            # Vote info
            method = vote_info.get('method', 'unknown')
            self.tracks_text.insert(tk.END, f"  Method: {method}\n")
            
            if 'real_count' in vote_info:
                real_count = vote_info['real_count']
                fake_count = vote_info['fake_count']
                self.tracks_text.insert(tk.END, 
                    f"  Votes: Real={real_count}, Fake={fake_count}\n")
            
            if 'real_ratio' in vote_info:
                ratio = vote_info['real_ratio']
                self.tracks_text.insert(tk.END, f"  Real Ratio: {ratio:.2%}\n")
            
            self.tracks_text.insert(tk.END, "\n")
    
    def on_closing(self):
        """Handle window close"""
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = LivenessTrackingTestApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
