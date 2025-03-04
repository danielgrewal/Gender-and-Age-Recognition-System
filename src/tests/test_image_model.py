from PIL import Image
from app.image_model import ImageModel
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def test_image_model_get_age_returns_age():
    
    # Load image from file
    path = os.path.join(CURRENT_DIR, "../app/media", "image.jpg")
    image = Image.open(path)
    
    # Instantiate ImageModel
    model = ImageModel()
    
    # Run inference on image
    result = model.get_age(image)
    
    # Verify a valid age was returned (Between 10 and 83 inclusive are possible)
    assert 10 <= result <= 83
    
def test_image_model_get_gender_returns_gender():
    # Load image from file
    path = os.path.join(CURRENT_DIR, "../app/media", "image.jpg")
    image = Image.open(path)
    
    # Instantiate ImageModel
    model = ImageModel()
    
    # Run inference on image
    result = model.get_gender(image)
    
    assert result == "Male"