from PIL import Image  
import cv2
import numpy as np
from datetime import datetime
from torchvision import transforms
import torch
from models.proposed_model import Proposed_model
import os
from facenet_pytorch import MTCNN
from pathlib import Path

class image_model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=60, device=self.device)
        self.image_size = 384

        # Resolve paths relative to backend/ so launching from any cwd works.
        self.base_dir = Path(__file__).resolve().parents[1]

        checkpoint_path = self.base_dir / 'weights' / 'proposed_model.pth'
    
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if os.path.getsize(checkpoint_path) == 0:
            raise ValueError("Checkpoint file is empty (0 bytes)")
        
        try:
     
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except RuntimeError:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, encoding='latin1')
            except RuntimeError:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict' key")
        
        self.model = Proposed_model(num_classes=2)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        boxes, probs = self.mtcnn.detect(image_np)
        
        if boxes is None or probs[0] < 0.9: 
            return "Face not found"
        
        x1, y1, x2, y2 = boxes[0].astype(int)
        
        height, width = image_np.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return "Face not found"
        
        face_region = image_np[y1:y2, x1:x2]
        face_pil = Image.fromarray(face_region)
        
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(face_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        output = "Real" if prediction.argmax().item() == 0 else "Fake"
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = self.base_dir / 'output' / 'image'
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'run_{timestamp}__{output}.png')
        
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, image_bgr)
        
        return output