from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms, models

import cv2
import base64
from io import BytesIO
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageManager:
    
    def __init__(self):
        self.face_detector = None
    
    def deserialize(self, image_serialized: str) -> Image:
        
        image_data = base64.b64decode(image_serialized)
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
    
    def serialize(self, image: Image) -> str:
        
        # Write image to buffer as jpeg
        buffer = BytesIO()
        image.save(buffer, format='jpeg')
        
        # Encode image into bytearray and convert it to string format
        image_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_encoded
        
    
    def resize(self, image: Image) -> Image:
        """
        Aligns face and generates resulting resized output of 224 X 224 Image.
        
        """
        # Instantiate face detector to extract resized image from source
        if self.face_detector is None:
            self.face_detector = MTCNN(image_size = 224, margin = 20, post_process = False, device = DEVICE)
        
        # Obtain face tensor
        face_tensor = self.face_detector(image)
        
        # Return None if no face is found
        if face_tensor is None:
            return None
        
        to_pil = transforms.ToPILImage()
        
        # Causing an error when not sent to CPU
        resized_image = to_pil(face_tensor.cpu())
        
        return resized_image

