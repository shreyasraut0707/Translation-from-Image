"""
Utility functions for the application
"""
import os
import re
import string
from datetime import datetime


def validate_file_path(file_path):
    """
    Validate that file exists and is readable
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid, False otherwise
    """
    return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


def is_image_file(file_path):
    """
    Check if file is an image
    
    Args:
        file_path: Path to file
        
    Returns:
        True if image file, False otherwise
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in image_extensions


def is_video_file(file_path):
    """
    Check if file is a video
    
    Args:
        file_path: Path to file
        
    Returns:
        True if video file, False otherwise
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions


def clean_text(text):
    """
    Clean and normalize extracted text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char isprintable or char.isspace())
    
    # Trim
    text = text.strip()
    
    return text


def contains_english(text):
    """
    Check if text contains significant English characters
    
    Args:
        text: Input text
        
    Returns:
        True if contains English, False otherwise
    """
    if not text:
        return False
    
    # Count ASCII letters
    ascii_letters = sum(1 for char in text if char in string.ascii_letters)
    total_chars = sum(1 for char in text if not char.isspace())
    
    if total_chars == 0:
        return False
    
    # At least 70% ASCII letters
    return (ascii_letters / total_chars) > 0.7


def format_confidence(confidence):
    """
    Format confidence score as percentage
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        Formatted string
    """
    return f"{confidence * 100:.1f}%"


def get_timestamp():
    """
    Get current timestamp
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_text_to_file(text, output_path):
    """
    Save text to file
    
    Args:
        text: Text to save
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Error saving text: {e}")
        return False


def setup_logging(log_file='app.log'):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
    """
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    test_text = "  Hello   World!  \n  "
    print(f"Original: '{test_text}'")
    print(f"Cleaned: '{clean_text(test_text)}'")
    
    print(f"\nContains English: {contains_english('Hello World')}")
    print(f"Contains English: {contains_english('你好世界')}")
    
    print(f"\nConfidence: {format_confidence(0.856)}")
    print(f"Timestamp: {get_timestamp()}")
