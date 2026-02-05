# TrOCR Engine - High accuracy OCR using Microsoft TrOCR from Hugging Face
# This model is downloaded from huggingface and runs locally

import os
import sys
import cv2
import numpy as np
from PIL import Image

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TrOCREngine:
    """
    High-accuracy OCR using Microsoft TrOCR model
    
    Available models:
    - microsoft/trocr-base-printed: for printed text (recommended)
    - microsoft/trocr-large-printed: higher accuracy but slower
    - microsoft/trocr-base-handwritten: for handwritten text
    """
    
    def __init__(self, model_name="microsoft/trocr-base-printed"):
        """Initialize TrOCR with specified model"""
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load TrOCR model from Hugging Face"""
        try:
            print(f"Loading TrOCR model: {self.model_name}")
            print("This may take a few minutes on first run (downloading model)...")
            
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            # check for GPU
            if torch.cuda.is_available():
                self.device = "cuda"
                print("GPU detected - using CUDA for faster inference")
            else:
                print("No GPU detected - using CPU (slower but works)")
            
            # load the model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print("TrOCR model loaded successfully!")
            print(f"  Model: {self.model_name}")
            print(f"  Device: {self.device}")
            
        except ImportError as e:
            print(f"Error: Required packages not installed: {e}")
            print("Run: pip install transformers torch")
            self.is_loaded = False
            
        except Exception as e:
            print(f"Error loading TrOCR model: {e}")
            self.is_loaded = False
    
    def extract_text_from_image(self, pil_image):
        """Extract text from a PIL image"""
        if not self.is_loaded:
            return ""
        
        try:
            import torch
            
            # process image
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=128)
            
            # decode
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error in TrOCR inference: {e}")
            return ""
    
    def extract_text(self, image_path, use_regions=False):
        """
        Extract text from image file
        
        Args:
            image_path: path to image file
            use_regions: whether to detect text regions first
            
        Returns:
            tuple of (text, confidence)
        """
        if not self.is_loaded:
            print("TrOCR model not loaded!")
            return "", 0.0
        
        try:
            # load image
            image = cv2.imread(image_path)
            if image is None:
                return "", 0.0
            
            # convert to RGB PIL image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # extract text
            text = self.extract_text_from_image(pil_image)
            
            # TrOCR doesnt provide confidence scores
            confidence = 0.95 if text else 0.0
            
            return text, confidence
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return "", 0.0
    
    def is_english(self, text):
        """Check if text is in English"""
        if not text or not text.strip():
            return False
        
        # check ASCII ratio
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        
        if ascii_chars / max(total_chars, 1) > 0.7:
            return True
        
        try:
            from langdetect import detect
            return detect(text) == 'en'
        except:
            return ascii_chars / max(total_chars, 1) > 0.5


def download_model():
    """Download and cache the TrOCR model"""
    print("=" * 60)
    print("Downloading TrOCR Model from Hugging Face")
    print("=" * 60)
    print()
    
    engine = TrOCREngine()
    
    if engine.is_loaded:
        print()
        print("=" * 60)
        print("Model downloaded and ready to use!")
        print("=" * 60)
    else:
        print("Failed to download model")
    
    return engine


if __name__ == "__main__":
    print("Testing TrOCR Engine...")
    
    engine = TrOCREngine()
    
    if engine.is_loaded:
        test_image = os.path.join(config.TEST_DIR, "1002_1.png")
        if os.path.exists(test_image):
            text, confidence = engine.extract_text(test_image)
            print(f"\nExtracted text: {text}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Is English: {engine.is_english(text)}")
        else:
            print(f"Test image not found: {test_image}")
    else:
        print("TrOCR engine failed to load")
