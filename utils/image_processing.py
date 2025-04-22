import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, List

def preprocess_image(
    image_path: Union[str, Path], 
    input_size: Tuple[int, int]
) -> np.ndarray:
    """
    Preprocess an image for model input.
    
    Args:
        image_path: Path to the image file
        input_size: Tuple of (height, width) for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def load_calibration_images(
    image_folder: Union[str, Path],
    input_size: Tuple[int, int],
    max_images: int = 100
) -> List[np.ndarray]:
    """
    Load and preprocess calibration images.
    
    Args:
        image_folder: Directory containing calibration images
        input_size: Tuple of (height, width) for resizing
        max_images: Maximum number of images to load
        
    Returns:
        List of preprocessed images
    """
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    image_files = image_files[:max_images]
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
        
    return [preprocess_image(img_path, input_size) for img_path in image_files] 