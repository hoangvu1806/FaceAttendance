"""
Face Recognition Model using FaceNet (Pretrained)
Enhanced with quality filtering and strict matching
"""
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch
from config import settings, thresholds


class FaceEmbedder:
    """Face embedder with STRICT quality control and matching"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        print("Loading FaceNet pretrained model (VGGFace2)...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load PRETRAINED FaceNet
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        print(f"✓ FaceNet loaded (device: {self.device})")
        
        # Load thresholds from config
        self.T_reject = thresholds.T_REJECT
        self.T_accept = thresholds.T_ACCEPT
        self.min_margin = thresholds.MIN_MARGIN
        
        # Quality thresholds from config
        self.min_sharpness = settings.MIN_SHARPNESS
        self.min_brightness = settings.MIN_BRIGHTNESS
        self.max_brightness = settings.MAX_BRIGHTNESS
        self.min_contrast = settings.MIN_CONTRAST
        
        self._initialized = True
    
    def assess_quality(self, face_image):
        if face_image is None or face_image.size == 0:
            return False, 0.0, "Empty image"
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        if sharpness < self.min_sharpness:
            return False, sharpness, f"Blurry (sharpness={sharpness:.1f})"
        
        # 2. Brightness
        brightness = np.mean(gray)
        
        if brightness < self.min_brightness:
            return False, brightness, f"Too dark (brightness={brightness:.1f})"
        
        if brightness > self.max_brightness:
            return False, brightness, f"Too bright (brightness={brightness:.1f})"
        
        # 3. Contrast
        contrast = np.std(gray)
        if contrast < self.min_contrast:
            return False, contrast, f"Low contrast (contrast={contrast:.1f})"
        
        # Quality score (0-1)
        quality_score = min(1.0, (sharpness / 200.0) * (contrast / 50.0))
        
        return True, quality_score, "Good"
    
    def preprocess_face(self, face_image):
        """
        Preprocess face for FaceNet
        - Resize to 160x160
        - Convert to RGB
        - Normalize to [-1, 1]
        """
        # Already 160x160 from detector
        if face_image.shape[:2] != (160, 160):
            face = cv2.resize(face_image, (160, 160))
        else:
            face = face_image
        
        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        face = torch.from_numpy(face).float()
        face = face.permute(2, 0, 1)  # HWC to CHW
        face = (face - 127.5) / 128.0  # Normalize to [-1, 1]
        
        return face.unsqueeze(0).to(self.device)
    
    def extract(self, face_image, check_quality=True):
        """
        Extract 512-dim embedding with quality check
        """
        if face_image is None or face_image.size == 0:
            return None
        
        # Quality check
        if check_quality:
            is_good, quality_score, reason = self.assess_quality(face_image)
            if not is_good:
                print(f"Quality check failed: {reason}")
                return None
        
        try:
            # Preprocess
            face_tensor = self.preprocess_face(face_image)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
            
            # Convert to numpy
            embedding = embedding.cpu().numpy()[0]
            
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            else:
                return None
            
            return embedding.tolist()
            
        except Exception as e:
            print(f"Extract error: {e}")
            return None
    
    def extract_multiple(self, face_images, check_quality=True):
        """
        Extract embeddings for multiple faces (batch processing)
        Returns: List of embeddings (None for failed extractions)
        """
        if not face_images:
            return []
        
        embeddings = []
        valid_indices = []
        valid_tensors = []
        
        # Filter by quality and prepare batch
        for i, face_img in enumerate(face_images):
            if face_img is None or face_img.size == 0:
                embeddings.append(None)
                continue
            
            if check_quality:
                is_good, _, reason = self.assess_quality(face_img)
                if not is_good:
                    embeddings.append(None)
                    continue
            
            try:
                face_tensor = self.preprocess_face(face_img)
                valid_tensors.append(face_tensor)
                valid_indices.append(i)
                embeddings.append(None)  # Placeholder
            except:
                embeddings.append(None)
        
        if not valid_tensors:
            return embeddings
        
        # Batch inference
        try:
            batch = torch.cat(valid_tensors, dim=0)
            with torch.no_grad():
                batch_embeddings = self.model(batch)
            
            # Normalize and assign
            for i, idx in enumerate(valid_indices):
                emb = batch_embeddings[i].cpu().numpy()
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
                    embeddings[idx] = emb.tolist()
        except Exception as e:
            print(f"Batch extract error: {e}")
        
        return embeddings
    
    def calculate_similarity(self, emb1, emb2):
        """
        Calculate cosine similarity between two embeddings
        """
        return np.dot(np.array(emb1), np.array(emb2))
    
    def match_with_margin(self, query_embedding, database):
        """
        STRICT matching with margin-based decision
        
        Decision rules:
        1. s1 < T_reject (0.7): Unknown - too different
        2. s1 >= T_accept (0.85) AND margin >= 0.15: Accept - clear winner
        3. Otherwise: Uncertain - need more frames
        
        Returns: (name, decision, s1, s2, margin)
        """
        if not database or not query_embedding:
            return None, "unknown", 0.0, 0.0, 0.0
        
        query = np.array(query_embedding)
        similarities = []
        
        # Calculate similarities with ALL stored embeddings
        for name, stored_data in database.items():
            # Handle both formats
            if isinstance(stored_data[0], list):
                stored_embeddings = stored_data
            else:
                stored_embeddings = [stored_data]
            
            # Calculate similarity with EACH stored embedding
            person_sims = []
            for stored_emb in stored_embeddings:
                stored = np.array(stored_emb)
                if len(stored) == len(query):
                    sim = np.dot(query, stored)
                    person_sims.append(sim)
            
            if person_sims:
                # Use MEDIAN instead of MAX (more robust against outliers)
                median_sim = np.median(person_sims)
                max_sim = np.max(person_sims)
                
                # Require both median and max to be high
                # This ensures consistency across multiple stored embeddings
                final_sim = 0.7 * median_sim + 0.3 * max_sim
                
                similarities.append((name, final_sim, median_sim, max_sim))
        
        if not similarities:
            return None, "unknown", 0.0, 0.0, 0.0
        
        # Sort by final similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        s1 = similarities[0][1]  # Best match (weighted)
        s2 = similarities[1][1] if len(similarities) > 1 else 0.0  # Second best
        margin = s1 - s2
        best_name = similarities[0][0]
        
        # Additional check: median and max should both be reasonable
        median_s1 = similarities[0][2]
        max_s1 = similarities[0][3]
        
        # STRICT decision logic
        if s1 < self.T_reject:
            decision = "unknown"
        elif (s1 >= self.T_accept and 
              margin >= self.min_margin and
              median_s1 >= (self.T_accept - 0.05) and  # Median also high
              max_s1 >= self.T_accept):                # Max also high
            decision = "accept"
        else:
            decision = "uncertain"
        
        return best_name, decision, s1, s2, margin
    
    def verify_pair(self, emb1, emb2, threshold=0.85):
        """
        Verify if two embeddings are from the same person
        Returns: (is_same, similarity)
        """
        if emb1 is None or emb2 is None:
            return False, 0.0
        
        sim = self.calculate_similarity(emb1, emb2)
        return sim >= threshold, sim
