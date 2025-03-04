import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from facenet_pytorch import MTCNN
from app.image_manager import ImageManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGE_MODEL_PATH = "age_regression_model.pth"
GENDER_MODEL_PATH = "gender_classification_model.pth"


class ImageModel:
    
    def __init__(self):
        
        # Placeholders to reference age and gender models
        self.age_model = None
        self.gender_model = None
        
        # ImageNet standards (Efficientnet was initially trained on ImageNet)
        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])
        
        # PREPARE AGE MODEL FOR INFERENCE
        
        # Load pre-trained model architecture (EfficientNet) to serve as the base
        self.age_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Get neurons feeding into the classifier (fully connected layer)
        input_features_age = self.age_model.classifier[1].in_features
        # Replace second layer of fully connected layer and modify it to do regression (one output)
        self.age_model.classifier[1] = nn.Linear(input_features_age, 1)
        # Send model architecture to device (GPU or CPU)
        self.age_model.to(DEVICE)
        # Load local training result model weights onto architecture
        self.age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
        # Place model in evaluation mode for inference
        self.age_model.eval()
        
        # PREPARE GENDER MODEL FOR INFERENCE
        
        # Load pre-trained model architecture (EfficientNet) to serve as the base
        self.gender_model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Get neurons feeding into the classifier (fully connected layer)
        input_features_gender = self.gender_model.classifier[1].in_features
        # Replace second layer of fully connected layer and modify it for 2 outputs (Male, Female)
        self.gender_model.classifier[1] = nn.Linear(input_features_gender, 2)
        # Send model architecture to device (GPU or CPU)
        self.gender_model.to(DEVICE)
        # Load local training result model weights onto architecture
        self.gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
        # Place model in evaluation mode for inference
        self.gender_model.eval()
        
        
    def get_age(self, image_full: Image) -> int:
        #image_array = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(image_array)
        img_manager = ImageManager()
        face = img_manager.resize(image_full)
        
        if face is not None:
            # Preprocess image
            input_tensor = self.data_transform(face).unsqueeze(0).to(DEVICE)
            
            # Inference for age regression
            with torch.no_grad():
                age = self.age_model(input_tensor).item()
            
            return round(age)
        else:
            return None
    
    def get_gender(self, image_full: Image) -> str:
        #image_array = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(image_array)
        img_manager = ImageManager()
        face = img_manager.resize(image_full)
        
        if face is not None:
            
            # Preprocess image
            input_tensor = self.data_transform(face).unsqueeze(0).to(DEVICE)
            
            # Inference for gender classification
            with torch.no_grad():
                gender_scores = self.gender_model(input_tensor)
                
            gender_id = torch.argmax(gender_scores, dim=1).item()
            
            if gender_id == 0:
                return "Male"
            else:
                return "Female"
            
        else:
            return None
            
            
