import torch
import whisper
import numpy as np
from typing import Optional, Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASRModel:
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper ASR model
        
        Args:
            model_size: 'tiny', 'base', 'small', 'medium' (base recommended for 8GB VRAM)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading Whisper {model_size} model on {self.device}")
        
        # Load model with FP16 for GPU efficiency
        self.model = whisper.load_model(model_size, device=self.device)
        
        if self.device == "cuda":
            self.model = self.model.half()  # FP16 for faster inference
        
        logger.info(f"ASR Model loaded successfully")
        
        # Supported languages
        self.supported_languages = {
            'en': 'english',
            'es': 'spanish',
            'hi': 'hindi'
        }
    
    def transcribe(
        self, 
        audio_data: np.ndarray, 
        source_language: str = "en",
        sample_rate: int = 16000
    ) -> dict:
        """
        Transcribe audio to text
        
        Args:
            audio_data: Audio array (must be 16kHz mono)
            source_language: Language code ('en', 'es', 'hi')
            sample_rate: Sample rate (must be 16000)
        
        Returns:
            dict: {'text': str, 'language': str, 'confidence': float}
        """
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1]
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # Whisper expects audio at 16kHz
            if sample_rate != 16000:
                raise ValueError("Audio must be sampled at 16kHz")
            
            # Transcribe with language hint
            lang_code = self.supported_languages.get(source_language, 'english')
            
            result = self.model.transcribe(
                audio_data,
                language=lang_code,
                fp16=(self.device == "cuda"),
                verbose=False
            )
            
            return {
                'text': result['text'].strip(),
                'language': result.get('language', source_language),
                'confidence': self._calculate_confidence(result)
            }
            
        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
            return {'text': '', 'language': source_language, 'confidence': 0.0}
    
    def transcribe_streaming(
        self,
        audio_chunks: Generator[np.ndarray, None, None],
        source_language: str = "en",
        chunk_duration: float = 3.0
    ) -> Generator[dict, None, None]:
        """
        Transcribe streaming audio chunks
        
        Args:
            audio_chunks: Generator yielding audio chunks
            source_language: Language code
            chunk_duration: Duration of each chunk in seconds
        
        Yields:
            dict: Transcription results
        """
        buffer = np.array([], dtype=np.float32)
        chunk_size = int(16000 * chunk_duration)  # 3 seconds at 16kHz
        
        for chunk in audio_chunks:
            buffer = np.concatenate([buffer, chunk])
            
            # Process when buffer is large enough
            if len(buffer) >= chunk_size:
                result = self.transcribe(buffer[:chunk_size], source_language)
                yield result
                
                # Keep last 0.5s for context (overlap)
                overlap_size = int(16000 * 0.5)
                buffer = buffer[chunk_size - overlap_size:]
        
        # Process remaining audio
        if len(buffer) > 16000:  # At least 1 second
            result = self.transcribe(buffer, source_language)
            yield result
    
    def _calculate_confidence(self, result: dict) -> float:
        """
        Calculate confidence score from Whisper result
        
        Args:
            result: Whisper transcription result
        
        Returns:
            float: Confidence score [0, 1]
        """
        # Whisper doesn't provide direct confidence, estimate from segments
        if 'segments' in result and result['segments']:
            confidences = []
            for segment in result['segments']:
                if 'avg_logprob' in segment:
                    # Convert log probability to confidence
                    conf = np.exp(segment['avg_logprob'])
                    confidences.append(conf)
            
            if confidences:
                return float(np.mean(confidences))
        
        return 0.8  # Default confidence if no segments
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_type': 'whisper',
            'device': self.device,
            'supported_languages': list(self.supported_languages.keys()),
            'sample_rate': 16000
        }


# Test function
if __name__ == "__main__":
    # Initialize model
    asr = ASRModel(model_size="base")
    
    # Test with dummy audio (1 second of silence)
    test_audio = np.zeros(16000, dtype=np.float32)
    result = asr.transcribe(test_audio, source_language="en")
    
    print("ASR Model Test:")
    print(f"Text: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nModel Info: {asr.get_model_info()}")