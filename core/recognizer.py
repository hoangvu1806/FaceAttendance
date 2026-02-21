"""
Face Recognition with temporal voting and tracking
"""

import time
import numpy as np
from collections import deque, Counter
from config import thresholds


class FaceRecognizer:
    """Face recognizer with temporal voting"""
    
    def __init__(self, embedder, database):
        self.embedder = embedder
        self.database = database
        
        # Load thresholds
        self.t_reject = thresholds.T_REJECT
        self.t_accept = thresholds.T_ACCEPT
        self.min_margin = thresholds.MIN_MARGIN
        self.voting_frames = thresholds.VOTING_FRAMES
        self.voting_threshold = thresholds.VOTING_THRESHOLD
        
        # Frame buffer for voting
        self.frame_buffer = {}
        
        # Face tracking
        self.face_tracks = {}
        self.next_track_id = 0
        self.max_track_age = 30
        self.iou_threshold = 0.3
        
        # Cache
        self.confirmed_cache_time = 3.0
    
    def recognize(self, face_img, is_frontal=True, track_id=None, liveness_score=1.0):
        """
        Recognize face with temporal voting
        
        Args:
            face_img: Aligned face image (160x160x3)
            is_frontal: Whether face is frontal
            track_id: Track ID for temporal voting
            liveness_score: Liveness score
            
        Returns:
            dict with recognition result
        """
        # Check cache
        if track_id is not None and track_id in self.face_tracks:
            track_data = self.face_tracks[track_id]
            
            if (track_data.get('confirmed_name') and 
                track_data.get('confirmed_time') and
                (time.time() - track_data['confirmed_time']) < self.confirmed_cache_time):
                
                return {
                    "name": track_data['confirmed_name'],
                    "score": track_data['confirmed_score'],
                    "status": "confirmed",
                    "message": f"{track_data['confirmed_name']} - CONFIRMED",
                    "color": (0, 255, 0),
                    "voting_info": {
                        "decision": "accepted",
                        "voted_name": track_data['confirmed_name'],
                        "avg_score": track_data['confirmed_score'],
                        "vote_count": self.voting_frames,
                        "total_frames": self.voting_frames
                    },
                    "liveness_score": liveness_score
                }
        
        # Extract embedding
        emb = self.embedder.extract(face_img, check_quality=False)
        if not emb:
            return {
                "name": None,
                "score": 0.0,
                "status": "error",
                "message": "Error",
                "color": (128, 128, 128),
                "voting_info": None
            }
        
        # Match with database
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
        
        # Temporal voting
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
                # Cache result
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
            else:  # pending
                if score < self.t_reject:
                    status_msg = f"Unknown ({score:.2f})"
                    color = (0, 0, 255)
                elif score > self.t_accept:
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
            # Single frame recognition
            if score < self.t_reject:
                status_msg = f"Unknown ({score:.2f})"
                color = (0, 0, 255)
            elif score > self.t_accept:
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
    
    def match(self, embedding):
        """Match embedding with database"""
        db = self.database.get_all()
        name, decision, s1, s2, margin = self.embedder.match_with_margin(embedding, db)
        
        if decision == "accept":
            return name, s1
        elif decision == "uncertain":
            return name, s1
        else:  # unknown
            return None, s1
    
    def add_vote(self, track_id, name, score):
        """Add vote for temporal voting"""
        if track_id not in self.frame_buffer:
            self.frame_buffer[track_id] = deque(maxlen=self.voting_frames)
        
        self.frame_buffer[track_id].append({
            "name": name,
            "score": score,
            "time": time.time()
        })
    
    def get_voting_result(self, track_id):
        """Get voting result"""
        if track_id not in self.frame_buffer:
            return "pending", None, 0.0, 0
        
        buffer = self.frame_buffer[track_id]
        if len(buffer) < self.voting_frames:
            return "pending", None, 0.0, len(buffer)
        
        names = [vote["name"] for vote in buffer]
        scores = [vote["score"] for vote in buffer]
        
        name_counts = Counter(names)
        most_common_name, vote_count = name_counts.most_common(1)[0]
        
        name_scores = [s for n, s in zip(names, scores) if n == most_common_name]
        avg_score = np.mean(name_scores)
        min_score = np.min(name_scores)
        score_std = np.std(name_scores)
        
        consistency_ratio = vote_count / len(buffer)
        
        if (vote_count >= self.voting_threshold and 
            avg_score >= self.t_accept and 
            min_score >= (self.t_accept - 0.08) and
            score_std < thresholds.MIN_SCORE_VARIANCE and
            consistency_ratio >= thresholds.MIN_CONSISTENCY):
            return "accepted", most_common_name, avg_score, vote_count
        elif len(buffer) >= self.voting_frames:
            return "rejected", most_common_name, avg_score, vote_count
        else:
            return "pending", most_common_name, avg_score, vote_count
    
    def update_tracks(self, faces):
        """Update face tracks"""
        current_time = time.time()
        
        # Age existing tracks
        for track_id in list(self.face_tracks.keys()):
            self.face_tracks[track_id]['age'] += 1
            if self.face_tracks[track_id]['age'] > self.max_track_age:
                del self.face_tracks[track_id]
                if track_id in self.frame_buffer:
                    del self.frame_buffer[track_id]
        
        # Match faces with tracks
        matched_tracks = set()
        face_assignments = {}
        
        for face_idx, (box, _) in enumerate(faces):
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
                self.face_tracks[best_track_id].update({
                    'last_box': box,
                    'age': 0,
                    'last_seen': current_time
                })
                matched_tracks.add(best_track_id)
                face_assignments[face_idx] = best_track_id
            else:
                new_track_id = self.next_track_id
                self.next_track_id += 1
                
                self.face_tracks[new_track_id] = {
                    'last_box': box,
                    'age': 0,
                    'last_seen': current_time,
                    'confirmed_name': None,
                    'confirmed_score': 0.0,
                    'confirmed_time': None,
                    'attendance_status': None,
                    'attendance_time': None
                }
                face_assignments[face_idx] = new_track_id
        
        return face_assignments
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
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
    
    def cleanup_old_tracks(self, max_age=5.0):
        """Remove old tracks"""
        current_time = time.time()
        
        to_remove = []
        for track_id, buffer in self.frame_buffer.items():
            if buffer and (current_time - buffer[-1]["time"]) > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.frame_buffer[track_id]
        
        to_remove_tracks = []
        for track_id, track_data in self.face_tracks.items():
            if (current_time - track_data['last_seen']) > max_age:
                to_remove_tracks.append(track_id)
        
        for track_id in to_remove_tracks:
            del self.face_tracks[track_id]
