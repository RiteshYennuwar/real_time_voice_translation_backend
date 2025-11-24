import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MTModel:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing MT models on {self.device}")
        
        # Model cache
        self.models = {}
        self.tokenizers = {}
        
        # Language pair configurations
        self.model_map = {
            ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
            ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi',
            ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
            ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en',
            ('es', 'hi'): None,  # Will use pivot through English
            ('hi', 'es'): None,  # Will use pivot through English
        }
        
        # Preload common models
        self._preload_models()
        
        logger.info("MT Models initialized successfully")
    
    def _preload_models(self):
        """Preload most common translation pairs"""
        # Load English <-> Spanish and English <-> Hindi
        priority_pairs = [('en', 'es'), ('es', 'en'), ('en', 'hi'), ('hi', 'en')]
        
        for src, tgt in priority_pairs:
            model_name = self.model_map.get((src, tgt))
            if model_name:
                self._load_model(src, tgt, model_name)
    
    def _load_model(self, src: str, tgt: str, model_name: str):
        key = f"{src}-{tgt}"
        
        if key not in self.models:
            logger.info(f"Loading model: {model_name}")
            
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                
                # Move to device and optimize
                model = model.to(self.device)
                if self.device == "cuda":
                    model = model.half()  # FP16 for efficiency
                
                model.eval()  # Set to evaluation mode
                
                self.tokenizers[key] = tokenizer
                self.models[key] = model
                
                logger.info(f"Model {key} loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
    
    def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        max_length: int = 512
    ) -> Dict[str, any]:
        if not text or not text.strip():
            return {
                'translated_text': '',
                'source_lang': source_lang,
                'target_lang': target_lang
            }
        
        # Same language - return as is
        if source_lang == target_lang:
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang
            }
        
        try:
            # Check if direct translation is available
            key = f"{source_lang}-{target_lang}"
            model_name = self.model_map.get((source_lang, target_lang))
            
            if model_name:
                # Direct translation
                if key not in self.models:
                    self._load_model(source_lang, target_lang, model_name)
                
                translated = self._translate_direct(text, key, max_length)
            else:
                # Pivot translation through English
                translated = self._translate_pivot(text, source_lang, target_lang, max_length)
            
            return {
                'translated_text': translated,
                'source_lang': source_lang,
                'target_lang': target_lang
            }
            
        except Exception as e:
            logger.error(f"Translation error ({source_lang}->{target_lang}): {e}")
            return {
                'translated_text': text,  # Return original on error
                'source_lang': source_lang,
                'target_lang': target_lang,
                'error': str(e)
            }
    
    def _translate_direct(self, text: str, key: str, max_length: int) -> str:
        tokenizer = self.tokenizers[key]
        model = self.models[key]
        
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,  # Beam search for better quality
                early_stopping=True
            )
        
        # Decode
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()
    
    def _translate_pivot(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        max_length: int
    ) -> str:
        logger.info(f"Using pivot translation: {source_lang} -> en -> {target_lang}")
        
        # First translate to English
        if source_lang != 'en':
            intermediate = self._translate_direct(
                text, 
                f"{source_lang}-en", 
                max_length
            )
        else:
            intermediate = text
        
        # Then translate from English to target
        if target_lang != 'en':
            final = self._translate_direct(
                intermediate, 
                f"en-{target_lang}", 
                max_length
            )
        else:
            final = intermediate
        
        return final
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        max_length: int = 512
    ) -> List[Dict[str, any]]:
        results = []
        for text in texts:
            result = self.translate(text, source_lang, target_lang, max_length)
            results.append(result)
        
        return results
    
    def get_supported_pairs(self) -> List[tuple]:
        return list(self.model_map.keys())
    
    def get_model_info(self) -> dict:
        return {
            'device': self.device,
            'supported_languages': ['en', 'es', 'hi'],
            'loaded_models': list(self.models.keys()),
            'supported_pairs': [f"{src}->{tgt}" for src, tgt in self.get_supported_pairs()]
        }


# Test function
if __name__ == "__main__":
    # Initialize MT model
    mt = MTModel()
    
    # Test translations
    test_cases = [
        ("Hello, how are you?", "en", "es"),
        ("Hello, how are you?", "en", "hi"),
        ("¿Cómo estás?", "es", "en"),
    ]
    
    print("MT Model Test:\n")
    for text, src, tgt in test_cases:
        result = mt.translate(text, src, tgt)
        print(f"{src.upper()} -> {tgt.upper()}")
        print(f"Input: {text}")
        print(f"Output: {result['translated_text']}")
        print("-" * 50)
    
    print(f"\nModel Info: {mt.get_model_info()}")