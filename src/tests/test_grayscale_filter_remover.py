import os
from PIL import Image
from app.grayscale_filter_remover import GrayscaleFilterRemover

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORIZED_IMAGE_PATH = os.path.join(CURRENT_DIR, "../app/media", "image_colorized.jpg")
GS_IMAGE_PATH = os.path.join(CURRENT_DIR, "../app/media", "image_gs.jpg")

def test_remove_returns_colored_image():
    image = Image.open(GS_IMAGE_PATH)
    
    model = GrayscaleFilterRemover(image = image)
    result = model.remove_filter()
    
    result.save(COLORIZED_IMAGE_PATH)
    