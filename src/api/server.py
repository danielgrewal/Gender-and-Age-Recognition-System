from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.image_manager import ImageManager
from app.image_model import ImageModel
from app.filter_detector import FilterDetector
from app.audio_model import AudioModel
from app.session_manager import SessionManager
from pathlib import Path
from PIL import Image
import os
import time
import csv
import datetime
import uvicorn
import subprocess




CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/process_frame")
async def process_frame(request: Request):
    # Start timer to measure performance
    start_time = time.time()
    
    img_manager = ImageManager()
    data = await request.json()
    serialized_image = data["image"].split(",")[1]
    image_rgb = img_manager.deserialize(serialized_image)
    
    img_model = ImageModel()
    filter_detector = FilterDetector(image_rgb)
    filter_removed, image = filter_detector.remove_filter()
    
    age = img_model.get_age(image)
    gender = img_model.get_gender(image)

    if age is None:
        return {"error": "No faces detected"}
    
    result = { "age": age, "gender": gender, "frame": "data:image/jpeg;base64," + img_manager.serialize(image) }
    
    # Record time taken for the request
    elapsed_time = time.time() - start_time   
    current_datetime = datetime.datetime.now().isoformat() 
    
    # Write performance data to a CSV file
    csv_file = os.path.join(CURRENT_DIR, "prediction_performance.csv")
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([current_datetime, elapsed_time])
    print(f"Logged timing: {current_datetime}, {elapsed_time:.2f} seconds")
    
    return result      

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
        original_audio_path = os.path.join("temp_recording.webm")
        with open(original_audio_path, "wb") as f:
            file_content = await audio.read()
            f.write(file_content)

        print(f"File saved: {original_audio_path}, Size: {len(file_content)} bytes")

        # Convert WebM to WAV if necessary
        converted_audio_path = os.path.join("temp_recording.wav")
        if audio.content_type == "audio/webm":
            conversion_command = [
                "ffmpeg", "-i", original_audio_path,
                "-ac", "1", "-ar", "16000",  # Mono channel, 16kHz (adjust if needed)
                "-y", converted_audio_path  # Overwrite existing file
            ]
            conversion_result = subprocess.run(conversion_command, capture_output=True, text=True)

            #print(f"FFmpeg stdout: {conversion_result.stdout}")
            #print(f"FFmpeg stderr: {conversion_result.stderr}")

            if conversion_result.returncode != 0:
                return JSONResponse(content={"error": "Audio format conversion failed"}, status_code=400)

        model = AudioModel()
        gender = model.get_gender(converted_audio_path)
        
        if gender is None:
            return JSONResponse(content={"gender": "--"})
        
        return JSONResponse(content={"gender": gender})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code = 500)

@app.post("/api/consent")
async def process_frame(request: Request):
    data = await request.json()
    serialized_image = data["image"].split(",")[1]
    session_manager = SessionManager()
    img_manager = ImageManager()
    image = img_manager.deserialize(serialized_image)
    session_manager.connect()
    result = session_manager.log_session(image, data["age"], data["gender"], True)
    session_manager.disconnect()
    print(f"Writing metadata to database: {data["age"]} - {data["gender"]}")
    return JSONResponse(content={"message": "Data Saved!"})
     
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)