"""
Text-to-Speech Module using Coqui TTS
Supports English, Spanish, and Hindi with natural voices
"""

import torch
import numpy as np
from TTS.api import TTS
from typing import Optional, Dict
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSModel:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize TTS models for multi-language support
        
        Args:
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing TTS models on {self.device}")
        
        # Language-specific TTS models
        self.models = {}
        
        # Model configurations for each language
        self.model_config = {
            'en': 'tts_models/en/ljspeech/tacotron2-DDC',  # English
            'es': 'tts_models/es/mai/tacotron2-DDC',       # Spanish
            'hi': 'tts_models/hi/nsk/vits'                  # Hindi (VITS is lighter)
        }
        
        # Preload models
        self._preload_models()
        
        logger.info("TTS Models initialized successfully")
    
    def _preload_models(self):
        """Preload TTS models for all supported languages"""
        for lang, model_name in self.model_config.items():
            try:
                logger.info(f"Loading TTS model for {lang}: {model_name}")
                
                tts = TTS(model_name=model_name, progress_bar=False)
                tts.to(self.device)
                
                self.models[lang] = tts
                logger.info(f"TTS model for {lang} loaded successfully")
                
            except Exception as e:
                logger.error(f"Error loading TTS model for {lang}: {e}")
                # Fallback to a simpler model if available
                try:
                    logger.info(f"Trying fallback model for {lang}")
                    fallback = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                    fallback.to(self.device)
                    self.models[lang] = fallback
                except:
                    logger.error(f"Failed to load fallback model for {lang}")
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker: Optional[str] = None,
        speed: float = 1.0
    ) -> Dict[str, any]:
        """
        Synthesize speech from text
        
        Args:
            text: Input text to synthesize
            language: Language code ('en', 'es', 'hi')
            speaker: Speaker ID (if model supports multiple speakers)
            speed: Speech speed (1.0 = normal)
        
        Returns:
            dict: {'audio': np.ndarray, 'sample_rate': int, 'language': str}
        """
        if not text or not text.strip():
            return {
                'audio': np.array([]),
                'sample_rate': 22050,
                'language': language
            }
        
        # Get model for language
        if language not in self.models:
            logger.warning(f"No model for language {language}, using English")
            language = 'en'
        
        try:
            model = self.models[language]
            
            # Generate speech
            # Note: TTS.tts() returns audio as numpy array
            audio = model.tts(text=text, speed=speed)
            
            # Convert to numpy array if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Ensure float32 format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize to [-1, 1]
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.abs(audio).max()
            
            return {
                'audio': audio,
                'sample_rate': model.synthesizer.output_sample_rate if hasattr(model, 'synthesizer') else 22050,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis error for {language}: {e}")
            # Return silence on error
            return {
                'audio': np.zeros(22050, dtype=np.float32),  # 1 second silence
                'sample_rate': 22050,
                'language': language,
                'error': str(e)
            }
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        speed: float = 1.0
    ) -> bool:
        """
        Synthesize speech and save to file
        
        Args:
            text: Input text
            output_path: Output file path (.wav)
            language: Language code
            speed: Speech speed
        
        Returns:
            bool: Success status
        """
        try:
            model = self.models.get(language, self.models['en'])
            model.tts_to_file(text=text, file_path=output_path, speed=speed)
            logger.info(f"Audio saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False
    
    def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        chunk_size: int = 2048
    ):
        """
        Synthesize speech in streaming chunks (for real-time playback)
        
        Args:
            text: Input text
            language: Language code
            chunk_size: Size of audio chunks
        
        Yields:
            np.ndarray: Audio chunks
        """
        result = self.synthesize(text, language)
        audio = result['audio']
        
        # Yield audio in chunks
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) > 0:
                yield chunk
    
    def get_available_speakers(self, language: str = "en") -> list:
        """
        Get available speakers for a language
        
        Args:
            language: Language code
        
        Returns:
            list: Available speaker IDs
        """
        if language not in self.models:
            return []
        
        try:
            model = self.models[language]
            if hasattr(model, 'speakers') and model.speakers:
                return model.speakers
        except:
            pass
        
        return []
    
    def estimate_duration(self, text: str, language: str = "en") -> float:
        """
        Estimate audio duration in seconds
        
        Args:
            text: Input text
            language: Language code
        
        Returns:
            float: Estimated duration in seconds
        """
        # Rough estimation: ~150 words per minute for most languages
        word_count = len(text.split())
        words_per_second = 2.5  # 150 WPM = 2.5 words/sec
        
        return word_count / words_per_second
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'device': self.device,
            'supported_languages': list(self.model_config.keys()),
            'loaded_models': list(self.models.keys()),
            'models': {
                lang: self.model_config[lang] 
                for lang in self.models.keys()
            }
        }


# Test function
if __name__ == "__main__":
    # Initialize TTS model
    tts = TTSModel()
    
    # Test synthesis
    test_cases = [
        ("Hello, this is a test.", "en"),
        ("Hola, esto es una prueba.", "es"),
        ("नमस्ते, यह एक परीक्षण है।", "hi"),
    ]
    
    print("TTS Model Test:\n")
    for text, lang in test_cases:
        print(f"Language: {lang.upper()}")
        print(f"Text: {text}")
        
        result = tts.synthesize(text, lang)
        print(f"Audio shape: {result['audio'].shape}")
        print(f"Sample rate: {result['sample_rate']}")
        print(f"Duration: {len(result['audio']) / result['sample_rate']:.2f}s")
        print(f"Estimated duration: {tts.estimate_duration(text, lang):.2f}s")
        print("-" * 50)
    
    print(f"\nModel Info: {tts.get_model_info()}")