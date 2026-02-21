import json
import os
import numpy as np

class EmbeddingDB:
    def __init__(self, db_path='data/embeddings.json', img_dir='data/faces'):
        self.db_path = db_path
        self.img_dir = img_dir
        self.last_mtime = 0
        self.data = self.load()
        os.makedirs(self.img_dir, exist_ok=True)
    
    def load(self):
        if os.path.exists(self.db_path):
            current_mtime = os.path.getmtime(self.db_path)
            self.last_mtime = current_mtime
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading DB: {e}")
                return {}
        return {}
    
    def save(self):
        try:
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f)
            if os.path.exists(self.db_path):
                self.last_mtime = os.path.getmtime(self.db_path)
        except Exception as e:
            print(f"Error saving DB: {e}")

    def add_person(self, name, embeddings, face_imgs):
        # Convert embeddings to list if they are numpy arrays
        emb_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        
        self.data[name] = emb_list
        self.save()
        
        # Save images
        person_dir = os.path.join(self.img_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        import cv2
        for i, img in enumerate(face_imgs):
            cv2.imwrite(os.path.join(person_dir, f"{i}.jpg"), img)

    def get_all(self):
        # Auto-reload if file changed on disk by another process (e.g. app.py)
        if os.path.exists(self.db_path):
            current_mtime = os.path.getmtime(self.db_path)
            if current_mtime > self.last_mtime:
                print("DB file changed, reloading...")
                self.data = self.load()
        return self.data
    
    def remove_person(self, person_id):
        if person_id in self.data:
            del self.data[person_id]
            self.save()
            
            import shutil
            person_dir = os.path.join(self.img_dir, person_id)
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
    
    def delete_person(self, person_id):
        if person_id not in self.data:
            return False
        self.remove_person(person_id)
        return True
    
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
