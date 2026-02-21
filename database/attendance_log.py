import csv
import os
from datetime import datetime
import cv2


class AttendanceLogger:
    def __init__(self, log_file='data/attendance.csv', img_dir='data/attendance_images'):
        self.log_file = log_file
        self.img_dir = img_dir
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Name', 'Score', 'Image'])
    
    def log(self, name, score, face_image=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        img_filename = None
        if face_image is not None:
            img_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            img_path = os.path.join(self.img_dir, img_filename)
            cv2.imwrite(img_path, face_image)
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, name, f"{score:.3f}", img_filename or ''])
        
        return timestamp
    
    def get_recent(self, n=10):
        if not os.path.exists(self.log_file):
            return []
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            return rows[-n:] if len(rows) > 1 else []
    
    def get_today_count(self):
        if not os.path.exists(self.log_file):
            return {}
        
        today = datetime.now().strftime('%Y-%m-%d')
        counts = {}
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].startswith(today):
                    name = row[1]
                    counts[name] = counts.get(name, 0) + 1
        
        return counts
