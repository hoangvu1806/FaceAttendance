import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from core.liveness import load_detector


class LivenessCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Liveness Detector - Camera Test")
        self.root.geometry("1000x700")

        self.camera = None
        self.is_running = False
        self.detector = None
        self.face_cascade = None
        self.current_frame = None

        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        self.setup_ui()
        self.load_models()
        
    def setup_ui(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

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
        
        # Separator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Model selection
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="MiniFASNetV2")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                   values=["MiniFASNetV2", "MiniFASNetV1SE"],
                                   state="readonly", width=15)
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Threshold
        ttk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_spin = ttk.Spinbox(control_frame, from_=0.0, to=1.0, 
                                     increment=0.1, textvariable=self.threshold_var,
                                     width=10)
        threshold_spin.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Buttons
        self.start_btn = ttk.Button(control_frame, text="▶ Start", 
                                    command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", 
                                   command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Main frame - Video display
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Video canvas
        self.canvas = tk.Canvas(main_frame, width=640, height=480, bg='black')
        self.canvas.pack(side=tk.LEFT, padx=5)
        
        # Info panel
        info_frame = ttk.LabelFrame(main_frame, text="Detection Info", padding="10")
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # FPS
        ttk.Label(info_frame, text="FPS:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.fps_label = ttk.Label(info_frame, text="0.0", font=('Arial', 12, 'bold'))
        self.fps_label.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Status
        ttk.Label(info_frame, text="Status:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.status_label = ttk.Label(info_frame, text="Not started", 
                                      font=('Arial', 12, 'bold'))
        self.status_label.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Result
        ttk.Label(info_frame, text="Result:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.result_label = ttk.Label(info_frame, text="-", 
                                      font=('Arial', 16, 'bold'))
        self.result_label.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Score
        ttk.Label(info_frame, text="Score:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.score_label = ttk.Label(info_frame, text="-", 
                                     font=('Arial', 14))
        self.score_label.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Face count
        ttk.Label(info_frame, text="Faces:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.face_count_label = ttk.Label(info_frame, text="0", 
                                          font=('Arial', 12))
        self.face_count_label.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Log
        ttk.Label(info_frame, text="Log:").grid(row=5, column=0, sticky=tk.NW, pady=5)
        log_scroll = ttk.Scrollbar(info_frame)
        log_scroll.grid(row=6, column=2, sticky=tk.NS)
        
        self.log_text = tk.Text(info_frame, height=15, width=30, 
                               yscrollcommand=log_scroll.set)
        self.log_text.grid(row=6, column=0, columnspan=2, pady=5)
        log_scroll.config(command=self.log_text.yview)
        
        # Bottom frame - Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_bar = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
        
    def detect_cameras(self):
        """Phát hiện các camera có sẵn"""
        self.log("Detecting available cameras...")
        available_cameras = []
        
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if available_cameras:
            self.log(f"Found cameras: {available_cameras}")
            # Update combobox values
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Combobox) and child.cget('textvariable') == str(self.camera_var):
                            child['values'] = available_cameras
                            break
        else:
            self.log("No cameras found!")
            messagebox.showwarning("Warning", "No cameras detected!")
    
    def load_models(self):
        """Load face detector và liveness detector"""
        try:
            self.log("Loading models...")
            
            # Load Haar Cascade cho face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade")
            
            # Load liveness detector
            model_path = 'Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'
            self.detector = load_detector(model_path=model_path, device='cpu')
            
            self.log("Models loaded successfully!")
            self.status_bar.config(text="Models loaded - Ready to start")
            
            # Auto-detect cameras
            self.detect_cameras()
            
        except Exception as e:
            self.log(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models:\n{str(e)}")
    
    def log(self, message):
        """Thêm message vào log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        
    def start_camera(self):
        """Bắt đầu camera"""
        try:
            camera_index = self.camera_var.get()
            self.log(f"Opening camera {camera_index}...")
            
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {camera_index}")
            
            # Get camera info
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Running")
            
            self.log(f"Camera {camera_index} started: {width}x{height} @ {fps}fps")
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.update_frame, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            self.log(f"Error starting camera: {str(e)}")
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")
    
    def stop_camera(self):
        """Dừng camera"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped")
        
        self.log("Camera stopped")
        
    def update_frame(self):
        """Update frame từ camera"""
        while self.is_running:
            ret, frame = self.camera.read()
            
            if not ret:
                self.log("Failed to read frame")
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )
            
            # Update face count
            self.face_count_label.config(text=str(len(faces)))
            
            # Process faces
            for (x, y, w, h) in faces:
                # Predict liveness
                bbox = [x, y, w, h]
                try:
                    score, label = self.detector.predict(frame, bbox)
                    
                    # Determine color based on result
                    threshold = self.threshold_var.get()
                    if label == 'real' and score >= threshold:
                        color = (0, 255, 0)  # Green for real
                        text = f"REAL: {score:.2f}"
                        self.result_label.config(text="REAL FACE", foreground='green')
                    else:
                        color = (0, 0, 255)  # Red for fake
                        text = f"FAKE: {score:.2f}"
                        self.result_label.config(text="FAKE FACE", foreground='red')
                    
                    self.score_label.config(text=f"{score:.4f}")
                    
                    # Draw rectangle and text
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                except Exception as e:
                    self.log(f"Prediction error: {str(e)}")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
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
            
            # Convert to PhotoImage and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            photo = ImageTk.PhotoImage(image=img)
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            
            time.sleep(0.01)
    
    def on_closing(self):
        """Xử lý khi đóng app"""
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = LivenessCameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
