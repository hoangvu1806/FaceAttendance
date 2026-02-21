"""
Configuration package for Face Attendance System
"""

from .settings import *
from .thresholds import *

__all__ = [
    # Camera & Performance
    'CAMERA_DEFAULT',
    'CAMERA_FPS',
    'FPS_TARGET',
    'SKIP_DETECTION_FRAMES',
    'CONFIRMED_CACHE_TIME',
    
    # Quality Control
    'MIN_SHARPNESS',
    'MIN_BRIGHTNESS',
    'MAX_BRIGHTNESS',
    'MIN_CONTRAST',
    
    # Face Detection
    'DETECTION_CONFIDENCE',
    'MAX_FACES',
    'OUTPUT_SIZE',
    
    # Recognition Thresholds
    'T_REJECT',
    'T_ACCEPT',
    'MIN_MARGIN',
    
    # Voting
    'VOTING_FRAMES',
    'VOTING_THRESHOLD',
    'MIN_SCORE_VARIANCE',
    'MIN_CONSISTENCY',
    
    # Attendance
    'ATTENDANCE_COOLDOWN',
    'ATTENDANCE_LOG_FILE',
    'ATTENDANCE_IMAGE_DIR',
    
    # Head Pose
    'YAW_THRESHOLD',
    'PITCH_THRESHOLD',
    'ROLL_THRESHOLD',
    'MAX_YAW',
    'MAX_PITCH',
    'MAX_ROLL',
    
    # Tracking
    'IOU_THRESHOLD',
    'MAX_TRACK_AGE',
    'TRACK_CLEANUP_AGE',
    
    # Enrollment
    'ENROLL_TARGET_SAMPLES',
    
    # UI
    'WINDOW_WIDTH',
    'WINDOW_HEIGHT',
    
    # Paths
    'LIVENESS_MODEL_PATH',
    'EMBEDDING_DB_PATH',
    'FACE_IMAGES_PATH',
    
    # Logging
    'LOG_LEVEL',
    'LOG_FORMAT',
]
