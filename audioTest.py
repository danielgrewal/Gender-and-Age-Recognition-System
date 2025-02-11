import sys
import torch
import sounddevice as sd 
import soundfile as sf
from model import ECAPA_gender

# Directly download the model from the huggingface model hub
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.eval()

# If using GPU or not.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Modify to accept a file path from command line argument
def predict_audio(file_path):
    with torch.no_grad():
        output = model.predict(file_path, device=device)
        return output

# Check if file path is passed as argument and process it
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = predict_audio(file_path)
        print("Gender: ", result)  # Print result which will be captured by the server
    else:
        print("No audio file provided.")
