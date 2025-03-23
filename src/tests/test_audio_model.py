from app.audio_model import AudioModel
import os


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def test_audio_model_get_gender_returns_gender():
    
    # Path to test file
    path = os.path.join(CURRENT_DIR, "../app/media", "recording.wav")
    
    # Initialize audio model and make prediction
    model = AudioModel()
    gender = model.get_gender(path)
    assert gender == "male"