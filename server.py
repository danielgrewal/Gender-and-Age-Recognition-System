from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # Added this import
from io import BytesIO  # Added this import
import numpy as np
from model import ECAPA_gender
import base64
import io
import tensorflow as tf
import torchaudio
import torch
from PIL import Image
import uvicorn
from fastapi.staticfiles import StaticFiles
from live_age_gender_detection_WITH_AVG import predict_age_gender
import subprocess

app = FastAPI()

# Enable CORS so frontend can communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/api/detect")
async def detect_age_gender(image: str = Form(...)):
    print("Received API request!")  # Debugging step

    try:
        # Decode base64 image
        image_data = base64.b64decode(image.split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)

         # Call the function from face_gender_age_logic.py
        result_text, age_predictions, gender_predictions = predict_age_gender(image)

        if len(age_predictions) == 0:
            return {"error": "No faces detected"}

        # For now, just return the last detected face's predictions (you can modify as needed)
        result = {
            "age": age_predictions[-1],
            "gender": gender_predictions[-1],
            "result_text": result_text
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
