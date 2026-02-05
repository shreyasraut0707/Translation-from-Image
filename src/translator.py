# Translator module - handles English to Hindi translation
# Uses Google Translate (free, no API key needed)

import sys
import os

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class Translator:
    """Handles translation from English to target language"""
    
    def __init__(self, target_language=None):
        """
        Initialize the translator
        
        Args:
            target_language: language code like 'hi' for hindi
        """
        self.target_language = target_language or config.TARGET_LANGUAGE
        self.translator = None
        self.translator_type = None
        
        # try google translate first
        try:
            from googletrans import Translator as GoogleTranslator
            self.translator = GoogleTranslator()
            self.translator_type = 'googletrans'
            print(f"Google Translate initialized (target: {self.target_language})")
        except ImportError:
            print("googletrans not available")
        
        # fallback to deep-translator if googletrans fails
        if self.translator is None:
            try:
                from deep_translator import GoogleTranslator as DeepGoogleTranslator
                self.translator = DeepGoogleTranslator(
                    source='en',
                    target=self.target_language
                )
                self.translator_type = 'deep_translator'
                print(f"Deep Translator initialized (target: {self.target_language})")
            except ImportError:
                print("deep_translator not available")
    
    def translate(self, text):
        """
        Translate English text to target language
        
        Args:
            text: english text to translate
            
        Returns:
            translated text
        """
        if not text or not text.strip():
            return ""
        
        if self.translator is None:
            print("No translator available")
            return text
        
        try:
            if self.translator_type == 'googletrans':
                result = self.translator.translate(
                    text,
                    src='en',
                    dest=self.target_language
                )
                return result.text
            
            elif self.translator_type == 'deep_translator':
                return self.translator.translate(text)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def translate_batch(self, texts):
        """Translate multiple texts at once"""
        return [self.translate(text) for text in texts]
    
    def detect_language(self, text):
        """Detect the language of given text"""
        try:
            if self.translator_type == 'googletrans':
                result = self.translator.detect(text)
                return result.lang
            else:
                from langdetect import detect
                return detect(text)
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'unknown'
    
    def is_english(self, text):
        """Check if text is in English"""
        if not text:
            return False
        
        lang = self.detect_language(text)
        return lang == 'en'
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        if self.translator_type == 'googletrans':
            from googletrans import LANGUAGES
            return LANGUAGES
        else:
            # common languages
            return {
                'en': 'english',
                'hi': 'hindi',
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'zh-cn': 'chinese (simplified)',
                'ja': 'japanese',
                'ko': 'korean',
                'ar': 'arabic',
                'ru': 'russian'
            }


if __name__ == "__main__":
    # test the translator
    translator = Translator(target_language='hi')
    
    test_text = "Hello, how are you?"
    print(f"\nOriginal: {test_text}")
    
    translated = translator.translate(test_text)
    print(f"Translated: {translated}")
    
    print(f"\nLanguage of '{test_text}': {translator.detect_language(test_text)}")
    print(f"Is English: {translator.is_english(test_text)}")
