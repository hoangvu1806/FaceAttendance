"""
Global settings for Face Attendance System
"""

# ============================================================================
# CAMERA SETTINGS
# ============================================================================
CAMERA_DEFAULT = "1"
CAMERA_FPS = 40
FPS_TARGET = 25  # Target milliseconds per frame

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================
SKIP_DETECTION_FRAMES = 1  # Skip N frames between detections (1 = every 2nd frame)
CONFIRMED_CACHE_TIME = 10.0  # Seconds to cache confirmed recognition

# ============================================================================
# IMAGE QUALITY THRESHOLDS
# ============================================================================
MIN_SHARPNESS = 100.0  # Laplacian variance (blur detection)
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 220.0
MIN_CONTRAST = 20.0

# ============================================================================
# FACE DETECTION
# ============================================================================
DETECTION_CONFIDENCE = 0.7  # MediaPipe detection confidence
MAX_FACES = 3  # Maximum faces to detect simultaneously
OUTPUT_SIZE = 160  # Face crop size (160 for FaceNet)

# ============================================================================
# TRACKING
# ============================================================================
IOU_THRESHOLD = 0.3  # IoU threshold for face tracking
MAX_TRACK_AGE = 30  # Maximum frames before track expires
TRACK_CLEANUP_AGE = 3.0  # Seconds before cleaning old tracks

# ============================================================================
# ENROLLMENT
# ============================================================================
ENROLL_TARGET_SAMPLES = 5  # Number of samples during enrollment

# ============================================================================
# UI SETTINGS
# ============================================================================
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

# ============================================================================
# MODEL PATHS
# ============================================================================
LIVENESS_MODEL_PATH = 'Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth'

# ============================================================================
# DATABASE PATHS
# ============================================================================
EMBEDDING_DB_PATH = 'data/embeddings.json'
ATTENDANCE_LOG_FILE = 'data/attendance.csv'
ATTENDANCE_IMAGE_DIR = 'data/attendance_images'
FACE_IMAGES_PATH = 'data/faces'

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
