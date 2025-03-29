import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io


def preprocess_image(image_data, target_size=(224, 224)):
    """
    Properly handles both file objects and bytes
    """
    try:
        # If it's a file object (from request.files)
        if hasattr(image_data, 'read'):
            image = Image.open(image_data)
            image_data.seek(0)  # Rewind the file pointer
        # If it's bytes
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            raise ValueError("Unsupported image input type")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize and normalize
        image = image.resize(target_size)
        img_array = np.array(image) / 255.0
        
        return img_array
        
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")
    
def apply_image_enhancement(image):
    """
    Apply additional image enhancements
    
    Args:
        image: PIL Image object
        
    Returns:
        Enhanced PIL Image
    """


    img = np.array(image)
    
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    

    return Image.fromarray(img)
