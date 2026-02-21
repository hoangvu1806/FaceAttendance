"""
Database modules for Face Attendance System
"""

from .embedding_db import EmbeddingDB
from .attendance_log import AttendanceLogger

__all__ = [
    'EmbeddingDB',
    'AttendanceLogger',
]
