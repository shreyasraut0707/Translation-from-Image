"""
Image preprocessing utilities for OCR
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import config


def enhance_image(image):
    """
    Apply image enhancement techniques
    
    Args:
        image: Input image (numpy array or PIL Image)
        
    Returns:
        Enhanced image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)
    
    return np.array(image)


def remove_noise(image):
    """
    Remove noise from image using morphological operations
    
    Args:
        image: Grayscale image
        
    Returns:
        Denoised image
    """
    # Apply median blur to remove salt-and-pepper noise
    denoised = cv2.medianBlur(image, 3)
    
    # Apply morphological opening to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    
    return denoised


def binarize_image(image):
    """
    Convert image to binary using adaptive thresholding
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image
    """
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary


def deskew_image(image):
    """
    Correct skew in image
    
    Args:
        image: Binary image
        
    Returns:
        Deskewed image
    """
    # Find all white pixels
    coords = np.column_stack(np.where(image > 0))
    
    if len(coords) == 0:
        return image
    
    # Find minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    # Rotate image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), 
                             flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def detect_text_regions(image):
    """
    Detect text regions in image using MSER or contours
    
    Args:
        image: Input image
        
    Returns:
        List of bounding boxes (x, y, w, h)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by size (remove very small regions)
        if w > 10 and h > 10:
            boxes.append((x, y, w, h))
    
    return boxes


def augment_image(image, mode='random'):
    """
    Apply data augmentation to image
    
    Args:
        image: Input image
        mode: Augmentation mode ('random', 'rotation', 'noise', etc.)
        
    Returns:
        Augmented image
    """
    if mode == 'random' or mode == 'rotation':
        # Random rotation (-5 to 5 degrees)
        angle = np.random.uniform(-5, 5)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    
    if mode == 'random' or mode == 'noise':
        # Add Gaussian noise
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
    
    if mode == 'random' or mode == 'brightness':
        # Random brightness adjustment
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    return image


def preprocess_for_ocr(image_path):
    """
    Complete preprocessing pipeline for OCR
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image ready for OCR
    """
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance image
    enhanced = enhance_image(gray)
    
    # Remove noise
    denoised = remove_noise(enhanced)
    
    # Binarize
    binary = binarize_image(denoised)
    
    # Deskew if needed
    deskewed = deskew_image(binary)
    
    return deskewed


if __name__ == "__main__":
    # Test preprocessing
    import os
    
    test_image = os.path.join(config.TEST_DIR, "1002_1.png")
    if os.path.exists(test_image):
        processed = preprocess_for_ocr(test_image)
        print(f"Processed image shape: {processed.shape}")
