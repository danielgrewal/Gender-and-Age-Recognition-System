from PIL import Image

class GrayscaleFilterDetector:
    
    def __init__(self, image):
        self.image = image
    
    def detect(self):
        """Detects the presence of a grayscale filter"""
        # img = Image.open(self.image)
        num_colors = self.image.getcolors()
        if num_colors:
            return True
        else:
            return False
