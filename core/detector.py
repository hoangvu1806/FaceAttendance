"""
Face Detector with PROPER alignment using 5-point similarity transform
"""
import cv2
import numpy as np
import mediapipe as mp


class FaceDetector:
    """MediaPipe-based face detector with PROPER 5-point alignment"""
    
    def __init__(self, min_detection_confidence=0.7):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )
        
        # max_num_faces=2 for better performance (access control usually 1 person)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Standard 5-point template for 112x112 (ArcFace/MobileFaceNet)
        self.template_112 = np.array([
            [38.2946, 51.6963],  # Left eye
            [73.5318, 51.5014],  # Right eye
            [56.0252, 71.7366],  # Nose tip
            [41.5493, 92.3655],  # Left mouth corner
            [70.7299, 92.2041]   # Right mouth corner
        ], dtype=np.float32)
        
        # Standard 5-point template for 160x160 (FaceNet)
        self.template_160 = np.array([
            [54.706573, 73.85186],   # Left eye
            [105.045425, 73.573425], # Right eye
            [80.036115, 102.48086],  # Nose tip
            [59.356144, 131.95071],  # Left mouth corner
            [101.04271, 131.72014]   # Right mouth corner
        ], dtype=np.float32)
    
    def detect(self, image):
        """Detect faces and return bounding boxes"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Clamp to image boundaries
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append((x, y, width, height))
        
        return faces
    
    def get_landmarks(self, image):
        """Get facial landmarks (468 points per face)"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        landmarks_list = []
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def match_landmarks_to_bbox(self, bbox, landmarks_list):
        """
        Match landmarks to bbox using center distance
        Returns: best matching landmarks or None
        """
        if not landmarks_list:
            return None
        
        x, y, w, h = bbox
        bbox_center = np.array([x + w/2, y + h/2])
        
        best_landmarks = None
        min_distance = float('inf')
        
        for landmarks in landmarks_list:
            # Calculate landmarks center (use nose tip as reference)
            if len(landmarks) >= 468:
                nose_tip = np.array(landmarks[1])  # Landmark 1 is nose tip
                distance = np.linalg.norm(nose_tip - bbox_center)
                
                # Check if nose is inside bbox (with margin)
                margin = max(w, h) * 0.3
                if (x - margin < nose_tip[0] < x + w + margin and
                    y - margin < nose_tip[1] < y + h + margin):
                    if distance < min_distance:
                        min_distance = distance
                        best_landmarks = landmarks
        
        return best_landmarks
    
    def get_5_points(self, landmarks):
        """
        Extract 5 key points from 468 landmarks
        Returns: np.array of shape (5, 2) - [left_eye, right_eye, nose, left_mouth, right_mouth]
        """
        if not landmarks or len(landmarks) < 468:
            return None
        
        # MediaPipe landmark indices for 5 key points
        left_eye = landmarks[33]      # Left eye outer corner
        right_eye = landmarks[263]    # Right eye outer corner
        nose_tip = landmarks[1]       # Nose tip
        left_mouth = landmarks[61]    # Left mouth corner
        right_mouth = landmarks[291]  # Right mouth corner
        
        points = np.array([
            left_eye,
            right_eye,
            nose_tip,
            left_mouth,
            right_mouth
        ], dtype=np.float32)
        
        return points
    
    def similarity_transform(self, src_points, dst_points):
        """
        Calculate similarity transform matrix (rotation + scale + translation)
        Args:
            src_points: Source 5 points (Nx2)
            dst_points: Destination 5 points (Nx2)
        Returns:
            M: 2x3 affine transform matrix
        """
        # Use cv2.estimateAffinePartial2D for similarity transform
        M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
        return M
    
    def align_face(self, image, landmarks, output_size=160):
        """
        Align face using 5-point similarity transform
        Args:
            image: Input image
            landmarks: 468 landmarks
            output_size: Output size (112 or 160)
        Returns:
            Aligned face image
        """
        # Get 5 key points
        src_points = self.get_5_points(landmarks)
        if src_points is None:
            return None
        
        # Select template based on output size
        if output_size == 112:
            dst_points = self.template_112
        else:  # 160
            dst_points = self.template_160
        
        # Calculate similarity transform
        M = self.similarity_transform(src_points, dst_points)
        if M is None:
            return None
        
        # Warp image to align face
        aligned = cv2.warpAffine(image, M, (output_size, output_size), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        
        return aligned
    
    def crop_face_simple(self, image, bbox):
        """
        Simple crop with proper padding (fallback when no landmarks)
        """
        x, y, w, h = bbox
        
        # Use max(w, h) for square crop
        size = max(w, h)
        pad = int(size * 0.3)
        
        # Calculate center
        cx = x + w // 2
        cy = y + h // 2
        
        # Calculate crop region
        half_size = (size + 2 * pad) // 2
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(image.shape[1], cx + half_size)
        y2 = min(image.shape[0], cy + half_size)
        
        # Crop
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face = cv2.resize(face, (160, 160))
        return face
    
    def detect_and_crop(self, image, output_size=160):
        """
        Detect faces, match landmarks, align and crop
        Args:
            image: Input image
            output_size: Output size (112 or 160)
        Returns:
            List of (bbox, aligned_face)
        """
        # Detect faces
        faces = self.detect(image)
        if not faces:
            return []
        
        # Get landmarks for all faces
        landmarks_list = self.get_landmarks(image)
        
        result = []
        for bbox in faces:
            # Match landmarks to this bbox
            landmarks = self.match_landmarks_to_bbox(bbox, landmarks_list)
            
            if landmarks is not None:
                # Align face using 5-point transform
                aligned_face = self.align_face(image, landmarks, output_size)
                if aligned_face is not None:
                    result.append((bbox, aligned_face))
                else:
                    # Fallback to simple crop
                    face = self.crop_face_simple(image, bbox)
                    if face is not None:
                        result.append((bbox, face))
            else:
                # No landmarks, use simple crop
                face = self.crop_face_simple(image, bbox)
                if face is not None:
                    result.append((bbox, face))
        
        return result
    
    def get_head_pose(self, landmarks):
        """
        Estimate head pose using solvePnP (PROPER method)
        Returns: dict with yaw, pitch, roll in degrees
        """
        if not landmarks or len(landmarks) < 468:
            return None
        
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        image_points = np.array([
            landmarks[1],    # Nose tip
            landmarks[152],  # Chin
            landmarks[33],   # Left eye
            landmarks[263],  # Right eye
            landmarks[61],   # Left mouth
            landmarks[291]   # Right mouth
        ], dtype=np.float64)
        
        # Camera internals (approximate)
        focal_length = 1.0
        center = (0.5, 0.5)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        # Calculate Euler angles
        sy = np.sqrt(rotation_mat[0, 0]**2 + rotation_mat[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        return {
            "pitch": np.degrees(pitch),
            "yaw": np.degrees(yaw),
            "roll": np.degrees(roll)
        }
    
    def is_frontal_face(self, landmarks, yaw_threshold=20, pitch_threshold=20, roll_threshold=25):
        """Check if face is frontal using proper pose estimation"""
        pose = self.get_head_pose(landmarks)
        if pose is None:
            return False
        
        return (abs(pose["yaw"]) < yaw_threshold and 
                abs(pose["pitch"]) < pitch_threshold and 
                abs(pose["roll"]) < roll_threshold)
    
    def __del__(self):
        try:
            if hasattr(self, 'face_detection') and self.face_detection is not None:
                self.face_detection.close()
        except:
            pass
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.face_mesh.close()
        except:
            pass
