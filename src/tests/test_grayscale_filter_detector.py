import os
import pytest
from PIL import Image
from app.grayscale_filter_detector import GrayscaleFilterDetector

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COLORED_IMAGE_PATH = os.path.join(CURRENT_DIR, "../app/media", "image.jpg")
GS_IMAGE_PATH = os.path.join(CURRENT_DIR, "../app/media", "image_gs.jpg")

def test_detect_colored_image_identified_as_colored():
    """
    Test if a coloured image is correctly identified.
    """
    detector = GrayscaleFilterDetector(COLORED_IMAGE_PATH)
    result = detector.detect() 
    assert result is False

def test_detect_grayscale_image_identified_as_grayscale():
    """
    Test if a grayscale image is correctly identified.
    """
    detector = GrayscaleFilterDetector(GS_IMAGE_PATH)
    result = detector.detect() 
    assert result is True