"""
Liveness Tracker with Voting Mechanism
Tracks faces across frames and uses voting to improve liveness detection accuracy
"""
import numpy as np
from collections import deque, defaultdict
import time


class FaceTrack:
    """Represents a tracked face with voting history"""
    
    def __init__(self, track_id, bbox, max_history=10):
        self.track_id = track_id
        self.bbox = bbox
        self.last_seen = time.time()
        self.age = 0  # Number of frames tracked
        
        # Voting history
        self.max_history = max_history
        self.liveness_scores = deque(maxlen=max_history)
        self.liveness_labels = deque(maxlen=max_history)
        
        # Statistics
        self.real_count = 0
        self.fake_count = 0
        self.total_predictions = 0
        
        # Confidence tracking
        self.confidence_scores = deque(maxlen=max_history)
        
    def update(self, bbox, liveness_score, liveness_label):
        """Update track with new detection"""
        self.bbox = bbox
        self.last_seen = time.time()
        self.age += 1
        
        # Add to history
        self.liveness_scores.append(liveness_score)
        self.liveness_labels.append(liveness_label)
        self.confidence_scores.append(liveness_score)
        
        # Update counts
        self.total_predictions += 1
        if liveness_label == 'real':
            self.real_count += 1
        else:
            self.fake_count += 1
    
    def get_voted_result(self, method='majority', min_samples=3):
        """
        Get voted liveness result
        
        Args:
            method: 'majority', 'weighted', 'confidence_threshold'
            min_samples: Minimum samples needed for reliable voting
            
        Returns:
            (is_real, confidence, vote_info)
        """
        if len(self.liveness_labels) < min_samples:
            # Not enough samples, return latest
            if len(self.liveness_labels) > 0:
                return (
                    self.liveness_labels[-1] == 'real',
                    self.liveness_scores[-1],
                    {'method': 'insufficient_samples', 'samples': len(self.liveness_labels)}
                )
            return False, 0.0, {'method': 'no_samples'}
        
        if method == 'majority':
            return self._vote_majority()
        elif method == 'weighted':
            return self._vote_weighted()
        elif method == 'confidence_threshold':
            return self._vote_confidence_threshold()
        else:
            return self._vote_majority()
    
    def _vote_majority(self):
        """Simple majority voting"""
        real_ratio = self.real_count / self.total_predictions
        is_real = real_ratio >= 0.5
        
        # Average confidence of the winning class
        if is_real:
            real_scores = [s for s, l in zip(self.liveness_scores, self.liveness_labels) if l == 'real']
            confidence = np.mean(real_scores) if real_scores else 0.5
        else:
            fake_scores = [s for s, l in zip(self.liveness_scores, self.liveness_labels) if l == 'fake']
            confidence = np.mean(fake_scores) if fake_scores else 0.5
        
        vote_info = {
            'method': 'majority',
            'real_count': self.real_count,
            'fake_count': self.fake_count,
            'real_ratio': real_ratio,
            'samples': self.total_predictions
        }
        
        return is_real, confidence, vote_info
    
    def _vote_weighted(self):
        """Weighted voting - recent predictions have more weight"""
        weights = np.linspace(0.5, 1.0, len(self.liveness_labels))
        
        weighted_real = 0
        weighted_fake = 0
        
        for weight, label, score in zip(weights, self.liveness_labels, self.liveness_scores):
            if label == 'real':
                weighted_real += weight * score
            else:
                weighted_fake += weight * score
        
        total_weight = np.sum(weights)
        weighted_real /= total_weight
        weighted_fake /= total_weight
        
        is_real = weighted_real > weighted_fake
        confidence = weighted_real if is_real else weighted_fake
        
        vote_info = {
            'method': 'weighted',
            'weighted_real': weighted_real,
            'weighted_fake': weighted_fake,
            'samples': len(self.liveness_labels)
        }
        
        return is_real, confidence, vote_info
    
    def _vote_confidence_threshold(self, high_conf_threshold=0.8, low_conf_threshold=0.3):
        """
        Voting based on confidence levels
        High confidence predictions have more weight
        """
        high_conf_real = 0
        high_conf_fake = 0
        low_conf_count = 0
        
        for score, label in zip(self.liveness_scores, self.liveness_labels):
            if score >= high_conf_threshold:
                if label == 'real':
                    high_conf_real += 1
                else:
                    high_conf_fake += 1
            elif score <= low_conf_threshold:
                low_conf_count += 1
        
        # If we have high confidence predictions, use them
        if high_conf_real + high_conf_fake > 0:
            is_real = high_conf_real > high_conf_fake
            confidence = 0.9 if is_real else 0.9
        else:
            # Fall back to majority voting
            return self._vote_majority()
        
        vote_info = {
            'method': 'confidence_threshold',
            'high_conf_real': high_conf_real,
            'high_conf_fake': high_conf_fake,
            'low_conf_count': low_conf_count,
            'samples': len(self.liveness_labels)
        }
        
        return is_real, confidence, vote_info
    
    def get_stability_score(self):
        """
        Calculate stability score (0-1)
        Higher score means more consistent predictions
        """
        if len(self.liveness_labels) < 2:
            return 0.0
        
        # Check label consistency
        labels_array = np.array([1 if l == 'real' else 0 for l in self.liveness_labels])
        label_std = np.std(labels_array)
        label_stability = 1.0 - label_std
        
        # Check score consistency
        score_std = np.std(self.liveness_scores)
        score_stability = 1.0 - min(score_std, 1.0)
        
        # Combined stability
        stability = (label_stability * 0.7 + score_stability * 0.3)
        return stability
    
    def is_expired(self, timeout=2.0):
        """Check if track has expired"""
        return (time.time() - self.last_seen) > timeout


class LivenessTracker:
    """
    Tracks faces across frames and maintains voting history
    """
    
    def __init__(self, 
                 max_history=10,
                 iou_threshold=0.5,
                 track_timeout=2.0,
                 voting_method='weighted',
                 min_samples_for_voting=3):
        """
        Args:
            max_history: Maximum number of predictions to keep per track
            iou_threshold: IoU threshold for matching detections to tracks
            track_timeout: Seconds before a track is considered expired
            voting_method: 'majority', 'weighted', or 'confidence_threshold'
            min_samples_for_voting: Minimum samples needed for reliable voting
        """
        self.max_history = max_history
        self.iou_threshold = iou_threshold
        self.track_timeout = track_timeout
        self.voting_method = voting_method
        self.min_samples_for_voting = min_samples_for_voting
        
        self.tracks = {}  # track_id -> FaceTrack
        self.next_track_id = 0
        
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _match_detection_to_track(self, bbox):
        """
        Match a detection to existing tracks
        Returns: track_id or None
        """
        best_iou = 0
        best_track_id = None
        
        for track_id, track in self.tracks.items():
            iou = self._calculate_iou(bbox, track.bbox)
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_track_id = track_id
        
        return best_track_id
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of (bbox, liveness_score, liveness_label)
            
        Returns:
            List of (track_id, bbox, is_real, confidence, vote_info)
        """
        # Remove expired tracks
        expired_ids = [tid for tid, track in self.tracks.items() if track.is_expired(self.track_timeout)]
        for tid in expired_ids:
            del self.tracks[tid]
        
        results = []
        matched_tracks = set()
        
        # Match detections to existing tracks
        for bbox, liveness_score, liveness_label in detections:
            track_id = self._match_detection_to_track(bbox)
            
            if track_id is not None:
                # Update existing track
                self.tracks[track_id].update(bbox, liveness_score, liveness_label)
                matched_tracks.add(track_id)
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                track = FaceTrack(track_id, bbox, self.max_history)
                track.update(bbox, liveness_score, liveness_label)
                self.tracks[track_id] = track
                matched_tracks.add(track_id)
            
            # Get voted result
            track = self.tracks[track_id]
            is_real, confidence, vote_info = track.get_voted_result(
                method=self.voting_method,
                min_samples=self.min_samples_for_voting
            )
            
            # Add stability score
            vote_info['stability'] = track.get_stability_score()
            vote_info['age'] = track.age
            
            results.append((track_id, bbox, is_real, confidence, vote_info))
        
        return results
    
    def get_track_info(self, track_id):
        """Get detailed information about a track"""
        if track_id not in self.tracks:
            return None
        
        track = self.tracks[track_id]
        return {
            'track_id': track_id,
            'age': track.age,
            'total_predictions': track.total_predictions,
            'real_count': track.real_count,
            'fake_count': track.fake_count,
            'stability': track.get_stability_score(),
            'last_seen': track.last_seen,
            'bbox': track.bbox
        }
    
    def reset(self):
        """Reset all tracks"""
        self.tracks.clear()
        self.next_track_id = 0
