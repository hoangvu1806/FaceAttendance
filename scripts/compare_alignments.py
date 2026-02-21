# -*- coding: utf-8 -*-
"""
So sánh các phương pháp alignment: 5-point vs 68-point vs 468-point
"""

import cv2
import numpy as np
import time
from core.detector import FaceDetector
from core.embedder import FaceEmbedder
import matplotlib.pyplot as plt


class AlignmentComparison:
    def __init__(self):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        
    def get_68_points_from_468(self, landmarks):
        """Extract 68 key points từ 468 MediaPipe landmarks"""
        if not landmarks or len(landmarks) < 468:
            return None
        
        # Mapping MediaPipe 468 → Dlib 68
        indices_68 = [
            # Jaw line (17)
            152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 
            454, 356, 389, 251, 284, 332, 297,
            # Right eyebrow (5)
            70, 63, 105, 66, 107,
            # Left eyebrow (5)
            336, 296, 334, 293, 300,
            # Nose bridge (4)
            168, 6, 197, 195,
            # Nose bottom (5)
            98, 97, 2, 326, 327,
            # Right eye (6)
            33, 160, 158, 133, 153, 144,
            # Left eye (6)
            362, 385, 387, 263, 373, 380,
            # Outer mouth (12)
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
            # Inner mouth (8)
            78, 191, 80, 81, 82, 13, 312, 311
        ]
        
        points = [landmarks[i] for i in indices_68]
        return np.array(points, dtype=np.float32)
    
    def align_5point(self, image, landmarks):
        """Standard 5-point alignment"""
        return self.detector.align_face(image, landmarks, output_size=160)
    
    def align_68point(self, image, landmarks):
        """68-point alignment"""
        points_68 = self.get_68_points_from_468(landmarks)
        if points_68 is None:
            return None
        
        # Simple template for 68 points (centered)
        # In practice, you'd use a proper template
        template_68 = self._get_template_68()
        
        # Affine transform
        M, _ = cv2.estimateAffinePartial2D(points_68, template_68)
        if M is None:
            return None
        
        aligned = cv2.warpAffine(image, M, (160, 160))
        return aligned
    
    def align_dense(self, image, landmarks):
        """Dense 468-point alignment (simplified)"""
        # For demo purposes, just use all 468 points
        # In practice, this is overkill
        points_468 = np.array(landmarks, dtype=np.float32)
        
        # Create a simple grid template
        template_468 = self._get_template_468()
        
        # Use thin plate spline or similar for dense alignment
        # For simplicity, we'll just use affine with subset
        M, _ = cv2.estimateAffinePartial2D(points_468[:50], template_468[:50])
        if M is None:
            return None
        
        aligned = cv2.warpAffine(image, M, (160, 160))
        return aligned
    
    def _get_template_68(self):
        """Get 68-point template for 160x160"""
        # Simplified template (you'd use a proper one in production)
        template = np.zeros((68, 2), dtype=np.float32)
        
        # Jaw (17) - outline
        for i in range(17):
            angle = np.pi * (i / 16.0)
            template[i] = [80 + 60 * np.sin(angle), 120 - 60 * np.cos(angle)]
        
        # Eyebrows, eyes, nose, mouth - simplified positions
        template[17:22] = [[50, 50], [60, 45], [70, 45], [80, 45], [90, 50]]  # Right eyebrow
        template[22:27] = [[110, 50], [100, 45], [90, 45], [80, 45], [70, 50]]  # Left eyebrow
        template[27:31] = [[80, 60], [80, 70], [80, 80], [80, 85]]  # Nose bridge
        template[31:36] = [[70, 90], [75, 92], [80, 93], [85, 92], [90, 90]]  # Nose bottom
        template[36:42] = [[50, 60], [45, 58], [45, 62], [50, 64], [55, 62], [55, 58]]  # Right eye
        template[42:48] = [[110, 60], [105, 58], [105, 62], [110, 64], [115, 62], [115, 58]]  # Left eye
        template[48:60] = [[60, 110], [65, 115], [70, 117], [75, 118], [80, 118], [85, 118], 
                          [90, 117], [95, 115], [100, 110], [95, 112], [85, 113], [75, 113]]  # Outer mouth
        template[60:68] = [[70, 115], [75, 116], [80, 116], [85, 116], [85, 114], [80, 114], 
                          [75, 114], [70, 114]]  # Inner mouth
        
        return template
    
    def _get_template_468(self):
        """Get 468-point template (simplified)"""
        # For demo, create a simple grid
        template = np.zeros((468, 2), dtype=np.float32)
        for i in range(468):
            row = i // 22
            col = i % 22
            template[i] = [20 + col * 6, 20 + row * 6]
        return template
    
    def compare_methods(self, image_path):
        """So sánh 3 phương pháp alignment"""
        print("="*70)
        print("ALIGNMENT METHODS COMPARISON")
        print("="*70)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot load image: {image_path}")
            return
        
        print(f"\nImage: {image_path}")
        print(f"Size: {image.shape}")
        
        # Detect face and landmarks
        print("\n[1/4] Detecting face and landmarks...")
        faces = self.detector.detect(image)
        if not faces:
            print("No face detected!")
            return
        
        landmarks_list = self.detector.get_landmarks(image)
        if not landmarks_list:
            print("No landmarks detected!")
            return
        
        bbox = faces[0]
        landmarks = landmarks_list[0]
        print(f"✓ Face detected: {bbox}")
        print(f"✓ Landmarks: {len(landmarks)} points")
        
        # Test each method
        results = {}
        
        # 5-point
        print("\n[2/4] Testing 5-point alignment...")
        start = time.time()
        aligned_5 = self.align_5point(image, landmarks)
        time_5 = (time.time() - start) * 1000
        
        if aligned_5 is not None:
            emb_5 = self.embedder.extract(aligned_5, check_quality=False)
            results['5-point'] = {
                'image': aligned_5,
                'time': time_5,
                'embedding': emb_5,
                'success': emb_5 is not None
            }
            print(f"✓ Time: {time_5:.2f}ms")
            print(f"✓ Embedding: {'Success' if emb_5 else 'Failed'}")
        else:
            print("✗ Alignment failed")
        
        # 68-point
        print("\n[3/4] Testing 68-point alignment...")
        start = time.time()
        aligned_68 = self.align_68point(image, landmarks)
        time_68 = (time.time() - start) * 1000
        
        if aligned_68 is not None:
            emb_68 = self.embedder.extract(aligned_68, check_quality=False)
            results['68-point'] = {
                'image': aligned_68,
                'time': time_68,
                'embedding': emb_68,
                'success': emb_68 is not None
            }
            print(f"✓ Time: {time_68:.2f}ms")
            print(f"✓ Embedding: {'Success' if emb_68 else 'Failed'}")
        else:
            print("✗ Alignment failed")
        
        # 468-point (dense)
        print("\n[4/4] Testing 468-point alignment...")
        start = time.time()
        aligned_468 = self.align_dense(image, landmarks)
        time_468 = (time.time() - start) * 1000
        
        if aligned_468 is not None:
            emb_468 = self.embedder.extract(aligned_468, check_quality=False)
            results['468-point'] = {
                'image': aligned_468,
                'time': time_468,
                'embedding': emb_468,
                'success': emb_468 is not None
            }
            print(f"✓ Time: {time_468:.2f}ms")
            print(f"✓ Embedding: {'Success' if emb_468 else 'Failed'}")
        else:
            print("✗ Alignment failed")
        
        # Compare embeddings
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        
        print(f"\n{'Method':<15} {'Time (ms)':<12} {'Speed':<10} {'Status':<10}")
        print("-"*70)
        
        baseline_time = results.get('5-point', {}).get('time', 1)
        
        for method, data in results.items():
            time_ms = data['time']
            speedup = baseline_time / time_ms
            status = "✓ OK" if data['success'] else "✗ FAIL"
            print(f"{method:<15} {time_ms:>8.2f}    {speedup:>6.2f}x    {status}")
        
        # Embedding similarity
        if '5-point' in results and '68-point' in results:
            emb_5 = results['5-point']['embedding']
            emb_68 = results['68-point']['embedding']
            
            if emb_5 and emb_68:
                sim = np.dot(emb_5, emb_68)
                print(f"\nSimilarity (5-point vs 68-point): {sim:.4f}")
                print(f"Difference: {(1-sim)*100:.2f}%")
        
        # Visualize
        self.visualize_results(image, bbox, landmarks, results)
        
        return results
    
    def visualize_results(self, original, bbox, landmarks, results):
        """Visualize comparison"""
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(4*(n_methods+1), 4))
        
        # Original with landmarks
        img_with_landmarks = original.copy()
        x, y, w, h = bbox
        cv2.rectangle(img_with_landmarks, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw 5 key points
        points_5 = self.detector.get_5_points(landmarks)
        if points_5 is not None:
            for i, point in enumerate(points_5):
                cv2.circle(img_with_landmarks, tuple(point.astype(int)), 5, (0, 0, 255), -1)
        
        axes[0].imshow(cv2.cvtColor(img_with_landmarks, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original\n(with 5 key points)', fontsize=10)
        axes[0].axis('off')
        
        # Aligned faces
        for i, (method, data) in enumerate(results.items()):
            aligned = data['image']
            time_ms = data['time']
            
            axes[i+1].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f'{method}\n{time_ms:.1f}ms', fontsize=10)
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig('alignment_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: alignment_comparison.png")
        plt.show()


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compare_alignment_methods.py <image_path>")
        print("\nExample:")
        print("  python compare_alignment_methods.py test_face.jpg")
        print("  python compare_alignment_methods.py Silent-Face-Anti-Spoofing/images/sample/image_F1.jpg")
        return
    
    image_path = sys.argv[1]
    
    comparator = AlignmentComparison()
    comparator.compare_methods(image_path)


if __name__ == "__main__":
    main()
