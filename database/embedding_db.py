import json
import os
import numpy as np

class EmbeddingDB:
    def __init__(self, db_path='data/embeddings.json', img_dir='data/faces'):
        self.db_path = db_path
        self.img_dir = img_dir
        self.data = self.load()
        os.makedirs(self.img_dir, exist_ok=True)
    
    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        if os.path.exists(self.db_path):
            import shutil
            from datetime import datetime
            backup_path = self.db_path.replace('.json', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            shutil.copy2(self.db_path, backup_path)
            
            backup_dir = os.path.join(os.path.dirname(self.db_path), 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            backup_file = os.path.join(backup_dir, os.path.basename(backup_path))
            shutil.move(backup_path, backup_file)
        
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def add_person(self, person_id, embeddings, face_images=None):
        if not embeddings:
            print(f"No valid embeddings for {person_id}")
            return
        
        # Filter out None embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        if not valid_embeddings:
            print(f"No valid embeddings after filtering for {person_id}")
            return
        
        # Store multiple embeddings instead of average (better discrimination)
        # Keep top embeddings based on quality/variance
        if len(valid_embeddings) > 10:
            # Select diverse embeddings
            emb_array = np.array(valid_embeddings)
            
            # Calculate pairwise distances to select diverse samples
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(emb_array, metric='cosine'))
            
            # Select embeddings that are diverse from each other
            selected_indices = [0]  # Start with first
            for _ in range(min(9, len(valid_embeddings) - 1)):  # Select up to 10 total
                # Find embedding most different from selected ones
                min_distances = np.min(distances[selected_indices], axis=0)
                next_idx = np.argmax(min_distances)
                if next_idx not in selected_indices:
                    selected_indices.append(next_idx)
            
            selected_embeddings = [valid_embeddings[i] for i in selected_indices]
        else:
            selected_embeddings = valid_embeddings
        
        # Normalize each embedding
        normalized_embeddings = []
        for emb in selected_embeddings:
            emb_array = np.array(emb)
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                normalized_embeddings.append((emb_array / norm).tolist())
        
        if not normalized_embeddings:
            print(f"No valid normalized embeddings for {person_id}")
            return
        
        # Store as list of embeddings
        self.data[person_id] = normalized_embeddings
        self.save()
        
        print(f"Added {person_id} with {len(normalized_embeddings)} diverse embeddings")
        
        if face_images:
            import cv2
            person_dir = os.path.join(self.img_dir, person_id)
            os.makedirs(person_dir, exist_ok=True)
            
            # Only save high-quality images
            saved_count = 0
            for i, img in enumerate(face_images):
                if img is not None and img.size > 0:
                    img_path = os.path.join(person_dir, f"{saved_count:03d}.jpg")
                    cv2.imwrite(img_path, img)
                    saved_count += 1
    
    def get_all(self):
        return self.data
    
    def remove_person(self, person_id):
        if person_id in self.data:
            del self.data[person_id]
            self.save()
            
            import shutil
            person_dir = os.path.join(self.img_dir, person_id)
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
    
    def list_persons(self):
        return list(self.data.keys())
    
    def get_person_images(self, person_id):
        person_dir = os.path.join(self.img_dir, person_id)
        if not os.path.exists(person_dir):
            return []
        import cv2
        images = []
        for f in sorted(os.listdir(person_dir)):
            if f.endswith('.jpg'):
                img = cv2.imread(os.path.join(person_dir, f))
                if img is not None:
                    images.append(img)
        return images
