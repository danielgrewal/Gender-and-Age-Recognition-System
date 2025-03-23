from .grayscale_filter_detector import GrayscaleFilterDetector
from .grayscale_filter_remover import GrayscaleFilterRemover

FILTER_DETECTORS = {
    GrayscaleFilterDetector: GrayscaleFilterRemover 
}

class FilterDetector:
    
    def __init__(self, image):
        self.image = image
        self.filter_removed = False

    def remove_filter(self):
        """
        Detect if a filter has been applied and remove it.
        """
        for FilterDetectorType, FilterRemoverType in FILTER_DETECTORS.items():
            detector = FilterDetectorType(self.image)
            has_filter = detector.detect()
            
            if has_filter:
                remover = FilterRemoverType(self.image)
                self.image = remover.remove_filter()
                self.filter_removed = True
        
        return self.filter_removed, self.image
                
        
            
            
            