from app.image_manager import ImageManager
from io import BytesIO
import os
from PIL import Image
import base64

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def test_serialize_image_is_serialized():
    
    # Create mock image
    image = Image.new(mode="RGB", size=(224, 224))
    manager = ImageManager()
    result = manager.serialize(image)

    # Check if the the image was converted to a string of bytes
    assert isinstance(result, str)
    
    # Decode the base64 string back to byte array
    decoded_bytes = base64.b64decode(result)

    # Convert bytes back to an image
    image_buffer = BytesIO(decoded_bytes)
    image_decoded = Image.open(image_buffer)
    
    # Check that the image was properly decoded after encoding
    assert image_decoded.mode == "RGB"
    assert image_decoded.size == (224, 224) 
    
def test_deserialize_image_is_deserialized():
    """
    Test if image serialized as byte string is correctly deserialized and decoded
    """    
    # Create mock image
    image = Image.new(mode="RGB", size=(224, 224))
    
    # Write image to buffer
    buffer = BytesIO()
    image.save(buffer, format='jpeg')
    image_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    # Instantiate ImageManager and deserialize image
    manager = ImageManager()
    result = manager.deserialize(image_encoded)
    
    # Check result
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGB'
    assert result.size == (224, 224)
   

    
def test_resize_image_returns_correct_dimensions():
    
    # Create mock image
    image_blank = Image.new(mode="RGB", size=(224, 224))
    
    # Instantiate ImageManager and resize image
    manager = ImageManager()
    result = manager.resize(image_blank)
    
    # Since mock image has no face, expecting None
    assert result is None
    
    # Load image from file
    path = os.path.join(CURRENT_DIR, "../app/media", "image.jpg")
    image = Image.open(path)
    
    # Since the image is of a subject, expecting resized image
    resized_image = manager.resize(image)
    
    # Check if the correct dimensions are obtained
    assert resized_image.size == (224, 224)