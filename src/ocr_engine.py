# OCR Engine - handles text extraction from images
# Uses EasyOCR as primary, with Tesseract and TrOCR as alternatives

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from preprocessing import preprocess_for_ocr, detect_text_regions
from models.crnn_model import build_crnn_model


class OCREngine:
    """Main OCR engine for extracting text from images"""
    
    def __init__(self, use_custom_model=False, use_tesseract=False, use_easyocr=True, use_trocr=True):
        """
        Initialize the OCR engine with different backends
        
        Args:
            use_custom_model: whether to use our trained CRNN model
            use_tesseract: whether to use Tesseract OCR
            use_easyocr: whether to use EasyOCR (recommended)
            use_trocr: whether to use TrOCR from huggingface
        """
        self.use_custom_model = use_custom_model
        self.use_tesseract = use_tesseract
        self.use_easyocr = use_easyocr
        self.use_trocr = use_trocr
        
        self.custom_model = None
        self.tesseract_available = False
        self.easyocr_reader = None
        self.trocr_engine = None
        
        # try to load TrOCR (high accuracy model from huggingface)
        if use_trocr:
            try:
                from trocr_engine import TrOCREngine
                self.trocr_engine = TrOCREngine()
                if self.trocr_engine.is_loaded:
                    print("TrOCR initialized (High Accuracy Mode)")
                else:
                    print("TrOCR failed to load, using fallback methods")
                    self.trocr_engine = None
            except ImportError as e:
                print(f"TrOCR not available: {e}")
            except Exception as e:
                print(f"TrOCR initialization error: {e}")
        
        # load our custom trained model if available
        if use_custom_model and os.path.exists(config.WEIGHTS_PATH):
            try:
                print("Loading custom CRNN model...")
                self.custom_model = build_crnn_model()
                self.custom_model.load_weights(config.WEIGHTS_PATH)
                print("Custom model loaded successfully")
            except Exception as e:
                print(f"Error loading custom model: {e}")
                self.custom_model = None
        
        # initialize Tesseract
        if use_tesseract:
            try:
                import pytesseract
                self.tesseract = pytesseract
                self.tesseract_available = True
                print("Tesseract OCR initialized")
            except ImportError:
                print("Tesseract not available")
        
        # initialize EasyOCR (main OCR engine)
        if use_easyocr:
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                print("EasyOCR initialized")
            except ImportError as e:
                print(f"EasyOCR not available: {e}")
            except Exception as e:
                print(f"EasyOCR initialization error: {e}")
    
    def extract_text_custom(self, image):
        """Extract text using our trained CRNN model"""
        if self.custom_model is None:
            return ""
        
        try:
            # convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # resize to model input size
            img_resized = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_input = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)
            
            # run prediction
            prediction = self.custom_model.predict(img_input, verbose=0)
            text = self._decode_prediction(prediction[0])
            
            return text
        except Exception as e:
            print(f"Error in custom OCR: {e}")
            return ""
    
    def extract_text_tesseract(self, image):
        """Extract text using Tesseract OCR"""
        if not self.tesseract_available:
            return ""
        
        try:
            # configure for english only
            config_str = '--oem 3 --psm 6 -l eng'
            text = self.tesseract.image_to_string(image, config=config_str)
            return text.strip()
        except Exception as e:
            print(f"Error in Tesseract OCR: {e}")
            return ""
    
    def extract_text_easyocr(self, image):
        """Extract text using EasyOCR - best for multi-line text"""
        if self.easyocr_reader is None:
            return ""
        
        try:
            results = self.easyocr_reader.readtext(image)
            
            if not results:
                return ""
            
            # sort results by position (top to bottom, left to right)
            def get_position(result):
                bbox = result[0]
                top_y = min(point[1] for point in bbox)
                left_x = min(point[0] for point in bbox)
                return (top_y, left_x)
            
            sorted_results = sorted(results, key=get_position)
            
            # group text by lines based on y position
            lines = []
            current_line = []
            current_y = None
            y_threshold = 20  # pixels
            
            for result in sorted_results:
                bbox = result[0]
                top_y = min(point[1] for point in bbox)
                
                if current_y is None:
                    current_y = top_y
                    current_line.append(result)
                elif abs(top_y - current_y) < y_threshold:
                    current_line.append(result)
                else:
                    # new line
                    current_line.sort(key=lambda r: min(p[0] for p in r[0]))
                    lines.append(current_line)
                    current_line = [result]
                    current_y = top_y
            
            # add the last line
            if current_line:
                current_line.sort(key=lambda r: min(p[0] for p in r[0]))
                lines.append(current_line)
            
            # combine all text
            text_lines = []
            for line in lines:
                line_text = ' '.join([r[1] for r in line])
                text_lines.append(line_text)
            
            text = '\n'.join(text_lines)
            return text.strip()
        except Exception as e:
            print(f"Error in EasyOCR: {e}")
            return ""
    
    def extract_text(self, image_path, detect_regions=False):
        """
        Main method to extract text from an image
        Tries multiple OCR engines and picks the best result
        
        Args:
            image_path: path to the image file
            detect_regions: whether to detect text regions first
            
        Returns:
            tuple of (extracted_text, confidence_score)
        """
        # load the image
        image = cv2.imread(image_path)
        if image is None:
            return "", 0.0
        
        results = []
        
        # try EasyOCR first (best for documents and screenshots)
        if self.easyocr_reader is not None:
            try:
                text = self.extract_text_easyocr(image)
                if text:
                    results.append(('easyocr', text, len(text)))
                    print(f"EasyOCR extracted ({len(text)} chars): {text[:80]}...")
            except Exception as e:
                print(f"EasyOCR error: {e}")
        
        # try Tesseract as backup
        if self.tesseract_available:
            try:
                text = self.extract_text_tesseract(image)
                if text:
                    results.append(('tesseract', text, len(text)))
                    print(f"Tesseract extracted ({len(text)} chars): {text[:80]}...")
            except Exception as e:
                print(f"Tesseract error: {e}")
        
        # try TrOCR (good for single words)
        if self.trocr_engine is not None:
            try:
                text, confidence = self.trocr_engine.extract_text(image_path)
                if text:
                    results.append(('trocr', text, len(text)))
                    print(f"TrOCR extracted ({len(text)} chars): {text[:80]}...")
            except Exception as e:
                print(f"TrOCR error: {e}")
        
        # try custom model
        if self.custom_model is not None:
            try:
                text = self.extract_text_custom(image)
                if text:
                    results.append(('custom', text, len(text)))
            except Exception as e:
                print(f"Custom model error: {e}")
        
        # pick the result with most text (works best for documents)
        if results:
            results.sort(key=lambda x: x[2], reverse=True)
            method, text, length = results[0]
            confidence = 0.90 if method == 'easyocr' else 0.85
            print(f"Using {method} OCR ({length} chars)")
            return text, confidence
        
        return "", 0.0
    
    def _decode_prediction(self, prediction):
        """Decode model output to text"""
        indices = np.argmax(prediction, axis=-1)
        
        # remove duplicates and blanks
        chars = []
        prev_idx = -1
        for idx in indices:
            if idx != prev_idx and idx > 0 and idx < len(config.CHARACTERS):
                chars.append(config.CHARACTERS[idx])
            prev_idx = idx
        
        return ''.join(chars)
    
    def is_english(self, text):
        """Check if the text is in English"""
        if not text or not text.strip():
            return False
        
        # check if mostly ASCII characters
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        total_chars = len(text)
        
        if ascii_chars / max(total_chars, 1) > 0.7:
            return True
        
        # try language detection as backup
        try:
            from langdetect import detect, LangDetectException
            language = detect(text)
            return language == 'en'
        except (LangDetectException, Exception) as e:
            print(f"Language detection fallback: {e}")
            return ascii_chars / max(total_chars, 1) > 0.5


if __name__ == "__main__":
    # test the OCR engine
    engine = OCREngine(use_custom_model=False, use_tesseract=True)
    
    test_image = os.path.join(config.TEST_DIR, "1002_1.png")
    if os.path.exists(test_image):
        text, confidence = engine.extract_text(test_image)
        print(f"\nExtracted text: {text}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Is English: {engine.is_english(text)}")
