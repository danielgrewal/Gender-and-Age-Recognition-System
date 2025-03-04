from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # Added this import
from io import BytesIO  # Added this import
import numpy as np
from ecapa_gender import ECAPA_gender
import base64
import io
import tensorflow as tf
import torchaudio
import torch
from PIL import Image
import uvicorn
from fastapi.staticfiles import StaticFiles
from torchvision import transforms, models
import torch.nn as nn
from facenet_pytorch import MTCNN
import subprocess
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGE_MODEL_PATH = "age_regression_model.pth"
GENDER_MODEL_PATH = "gender_classification_model.pth"
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
])

def detect_and_align(image, mtcnn):
    """
    Detects and aligns a face in the provided image using MTCNN.
    
    Args:
        image (PIL.Image): The input image.
        mtcnn (MTCNN): An initialized MTCNN face detector.
    
    Returns:
        PIL.Image or None: The cropped & aligned face image or None if no face is detected.
    """
    face_tensor = mtcnn(image)
    if face_tensor is None:
        return None
    face_image = transforms.ToPILImage()(face_tensor.cpu())
    return face_image

def get_age_regression_model():
    """
    Loads the EfficientNet_B4 model for age regression and modifies its classifier.
    
    Returns:
        torch.nn.Module: The age regression model.
    """
    model = models.efficientnet_b4(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model

def get_gender_classification_model():
    """
    Loads the EfficientNet_B4 model for gender classification and modifies its classifier.
    
    Returns:
        torch.nn.Module: The gender classification model.
    """
    model = models.efficientnet_b4(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model

def load_models():
    """
    Loads pre-trained age regression and gender classification models from disk.
    
    Returns:
        tuple: (age_model, gender_model) loaded to the DEVICE.
    """
    age_model = get_age_regression_model().to(DEVICE)
    gender_model = get_gender_classification_model().to(DEVICE)
    
    age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE))
    gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE))
    
    age_model.eval()
    gender_model.eval()
    
    return age_model, gender_model




app = FastAPI()

# Enable CORS so frontend can communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/process_frame")
async def detect_age_gender(request: Request):
    print("Received API request!")  # Debugging step
    data = await request.json()

    try:
        # Decode base64 image
        image_data = base64.b64decode(data["image"].split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        frame = np.array(image)
        
        mtcnn = MTCNN(image_size=224, margin=20, post_process=False, device=DEVICE)
        age_model, gender_model = load_models()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # Run face detection and alignment
        face = detect_and_align(pil_img, mtcnn)
        if face is not None:
            # Preprocess face
            input_tensor = data_transform(face).unsqueeze(0).to(DEVICE)
            
            # Inference for age regression
            with torch.no_grad():
                age_pred = age_model(input_tensor)
            age_value = age_pred.item()
            
            # Inference for gender classification
            with torch.no_grad():
                gender_logits = gender_model(input_tensor)
            gender_idx = torch.argmax(gender_logits, dim=1).item()
            gender_value = "Male" if gender_idx == 0 else "Female"

         # Call the function from face_gender_age_logic.py
       # result_text, age_predictions, gender_predictions = predict_age_gender(image)

        if face is None:
            return {"error": "No faces detected"}

        # For now, just return the last detected face's predictions (you can modify as needed)
        result = {
            "age": round(age_value),
            "gender": gender_value,
        }

        return result

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
@app.post("/api/detect_audio")
async def detect_audio(audio: UploadFile = File(...)):
    try:
        print("Received audio request!")  
        print(f"Filename: {audio.filename}")
        print(f"Content-Type: {audio.content_type}")

        # Ensure the audio format is valid
        if audio.content_type not in ["audio/wav", "audio/webm"]:
            return JSONResponse(content={"error": "Invalid audio format"}, status_code=400)

        # Save the uploaded audio file temporarily
        original_audio_path = "temp_recording.webm"
        with open(original_audio_path, "wb") as f:
            file_content = await audio.read()
            f.write(file_content)

        print(f"File saved: {original_audio_path}, Size: {len(file_content)} bytes")

        # Convert WebM to WAV if necessary
        converted_audio_path = "temp_recording.wav"
        if audio.content_type == "audio/webm":
            conversion_command = [
                "ffmpeg", "-i", original_audio_path,
                "-ac", "1", "-ar", "16000",  # Mono channel, 16kHz (adjust if needed)
                "-y", converted_audio_path  # Overwrite existing file
            ]
            conversion_result = subprocess.run(conversion_command, capture_output=True, text=True)

            print(f"FFmpeg stdout: {conversion_result.stdout}")
            print(f"FFmpeg stderr: {conversion_result.stderr}")

            if conversion_result.returncode != 0:
                return JSONResponse(content={"error": "Audio format conversion failed"}, status_code=400)

        # Call the audio processing script
        result = subprocess.run(
            ["python", "audioTest.py", converted_audio_path],
            capture_output=True, text=True
        )

        print(f"Subprocess return code: {result.returncode}")
        print(f"Subprocess stdout: {result.stdout}")
        print(f"Subprocess stderr: {result.stderr}")

        # If the script failed, return an error
        if result.returncode != 0 or not result.stdout.strip():
            return JSONResponse(content={"error": "Audio processing failed"}, status_code=400)

        # Extract gender from script output
        gender = result.stdout.split(":")[-1].strip()
        return JSONResponse(content={"gender": gender})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
