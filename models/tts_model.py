import torch
import numpy as np
from TTS.api import TTS
from typing import Optional, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSModel:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize TTS models with multilingual support using separate models per language
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing TTS models on {self.device}")
        
        self.models = {}
        
        # Use separate models for each language (simpler and more reliable)
        self.model_config = {
            'en': 'tts_models/en/ljspeech/vits',
            'es': 'tts_models/es/css10/vits',
            # For Hindi, we'll use edge-tts (Microsoft neural voices)
        }
        
        self._preload_models()
        logger.info("TTS Models initialized successfully")
    
    def _preload_models(self):
        """Preload TTS models for each language"""
        # Load English model
        try:
            logger.info("Loading English VITS model...")
            self.models['en'] = TTS(model_name=self.model_config['en'], progress_bar=True)
            self.models['en'].to(self.device)
            logger.info("✓ English model loaded")
        except Exception as e:
            logger.error(f"Failed to load English model: {e}")
            self._load_english_fallback()
        
        # Load Spanish model
        try:
            logger.info("Loading Spanish VITS model...")
            self.models['es'] = TTS(model_name=self.model_config['es'], progress_bar=True)
            self.models['es'].to(self.device)
            logger.info("✓ Spanish model loaded")
        except Exception as e:
            logger.error(f"Failed to load Spanish model: {e}")
        
        # For Hindi, we'll use edge-tts (external service)
        self.models['hi'] = None  # Will use edge-tts
        logger.info("Hindi TTS will use edge-tts (Microsoft neural voices)")
    
    def _load_english_fallback(self):
        """Load English-only model as fallback"""
        try:
            tts = TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC', progress_bar=False)
            tts.to(self.device)
            self.models['en'] = tts
            logger.info("✓ English fallback model loaded")
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
    
    def _synthesize_hindi_edge_tts(self, text: str) -> Dict[str, any]:
        """Synthesize Hindi using gTTS (Google Text-to-Speech) as primary method"""
        import tempfile
        import soundfile as sf
        
        try:
            from gtts import gTTS
            from pydub import AudioSegment
            
            # Use gTTS for Hindi
            tts = gTTS(text=text, lang='hi', slow=False)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                tmp_path = tmp.name
            
            tts.save(tmp_path)
            
            # Convert MP3 to WAV using pydub
            audio_segment = AudioSegment.from_mp3(tmp_path)
            
            # Set to mono and resample to 22050 Hz for consistency
            audio_segment = audio_segment.set_channels(1).set_frame_rate(22050)
            
            # Export as WAV
            wav_path = tmp_path.replace('.mp3', '.wav')
            audio_segment.export(wav_path, format='wav')
            
            # Read the audio
            audio, sr = sf.read(wav_path)
            
            # Cleanup
            os.unlink(tmp_path)
            os.unlink(wav_path)
            
            return {
                'audio': audio.astype(np.float32),
                'sample_rate': sr,
                'language': 'hi'
            }
            
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            # Return silent audio as fallback
            return {
                'audio': np.zeros(22050, dtype=np.float32),
                'sample_rate': 22050,
                'language': 'hi',
                'error': f'Hindi TTS failed: {e}'
            }
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker: Optional[str] = None,
        speed: float = 1.0
    ) -> Dict[str, any]:
        """
        Synthesize speech from text
        """
        if not text or not text.strip():
            return {
                'audio': np.array([]),
                'sample_rate': 22050,
                'language': language
            }
        
        try:
            logger.info(f"Synthesizing {language} text: {text[:50]}...")
            
            # Handle Hindi separately (gTTS - Google Text-to-Speech)
            if language == 'hi':
                return self._synthesize_hindi_edge_tts(text)
            
            # Get the model for this language
            model = self.models.get(language)
            if model is None:
                logger.warning(f"No model for {language}, using English")
                model = self.models.get('en')
                if model is None:
                    raise ValueError("No TTS models available")
            
            # Synthesize audio
            audio = model.tts(text=text)
            
            # Convert to numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            
            # Ensure float32 format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize to [-1, 1]
            if len(audio) > 0 and (np.abs(audio).max() > 1.0):
                audio = audio / np.abs(audio).max()
            
            # VITS models typically use 22050 Hz
            sample_rate = 22050
            
            logger.info(f"✓ Generated {len(audio)/sample_rate:.2f}s of audio")
            
            return {
                'audio': audio,
                'sample_rate': sample_rate,
                'language': language
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis error for {language}: {e}")
            return {
                'audio': np.zeros(22050, dtype=np.float32),
                'sample_rate': 22050,
                'language': language,
                'error': str(e)
            }
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "en",
        speed: float = 1.0,
        speaker: Optional[str] = None
    ) -> bool:
        """Synthesize speech and save to file"""
        try:
            result = self.synthesize(text, language, speaker, speed)
            
            import soundfile as sf
            sf.write(output_path, result['audio'], result['sample_rate'])
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
        """Synthesize speech in streaming chunks"""
        result = self.synthesize(text, language)
        audio = result['audio']
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) > 0:
                yield chunk
    
    def get_available_speakers(self, language: str = "en") -> list:
        """Get available speakers"""
        if language == 'hi':
            return ['hi-IN-SwaraNeural', 'hi-IN-MadhurNeural']
        return ['default']
    
    def estimate_duration(self, text: str, language: str = "en") -> float:
        """Estimate audio duration in seconds"""
        word_count = len(text.split())
        return word_count / 2.5
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'device': self.device,
            'supported_languages': ['en', 'es', 'hi'],
            'loaded_models': [k for k, v in self.models.items() if v is not None],
            'models': {
                'en': 'VITS (LJSpeech)',
                'es': 'VITS (CSS10)',
                'hi': 'edge-tts (Microsoft Neural)'
            },
            'hindi_voices': ['hi-IN-SwaraNeural', 'hi-IN-MadhurNeural']
        }


if __name__ == "__main__":
    tts = TTSModel()
    
    test_cases = [
        ("Hello, this is a test.", "en"),
        ("Hola, esto es una prueba.", "es"),
        ("नमस्ते, यह एक परीक्षण है।", "hi"),
    ]
    
    print("\nTTS Model Test:\n")
    for text, lang in test_cases:
        print(f"Language: {lang.upper()}")
        print(f"Text: {text}")
        
        result = tts.synthesize(text, lang)
        print(f"Audio shape: {result['audio'].shape}")
        print(f"Sample rate: {result['sample_rate']}")
        print(f"Duration: {len(result['audio']) / result['sample_rate']:.2f}s")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            tts.synthesize_to_file(text, f"test_{lang}.wav", lang)
            print(f"Saved to: test_{lang}.wav")
        print("-" * 50)
    
    print("\nModel Info:")
    print(tts.get_model_info())
