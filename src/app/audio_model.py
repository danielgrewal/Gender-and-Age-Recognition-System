import io
import os
import numpy as np
import soundfile as sf
import torch
from torchaudio import models
from app.ecapa_gender import ECAPA_gender

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioModel:
    
    def __init__(self):
        
        # Download the model from the huggingface model hub
        self.model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
        # Set model into evaluation mode
        self.model.eval()
        self.model.to(DEVICE)
    
    def get_gender(self, path: str):
        
        audio_segment, sample_rate = sf.read(path, dtype="float32")
    
        if audio_segment.ndim > 1 and audio_segment.shape[1] > 1:
            audio_segment = np.mean(audio_segment, axis=1)
        
        with torch.no_grad():
            # Write the audio to in-memory buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio_segment, sample_rate, format="WAV")
            buffer.seek(0)
            gender = self.model.predict(buffer, device = DEVICE)
            return gender
        