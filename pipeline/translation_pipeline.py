import numpy as np
import time
from typing import Dict, Generator, Optional
import logging
from dataclasses import dataclass
from collections import deque

# Import your models (ensure these files are in the correct location)
from models.asr_model import ASRModel
from models.mt_model import MTModel
from models.tts_model import TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    audio: np.ndarray
    source_lang: str
    target_lang: str
    sample_rate: int
    latency_ms: float
    asr_confidence: float
    timestamp: float


class TranslationPipeline:
    def __init__(
        self,
        asr_model=None,
        mt_model=None,
        tts_model=None,
        device: Optional[str] = None
    ):
        logger.info("Initializing Translation Pipeline")
        
        self.device = device or "cuda"
        
        # Initialize models
        self.asr = asr_model
        self.mt = mt_model
        self.tts = tts_model
        
        # Performance tracking
        self.metrics = {
            'total_translations': 0,
            'avg_latency': 0.0,
            'avg_asr_time': 0.0,
            'avg_mt_time': 0.0,
            'avg_tts_time': 0.0
        }
        
        # Buffer for streaming
        self.audio_buffer = deque(maxlen=5)  # Keep last 5 chunks
        
        logger.info("Translation Pipeline initialized successfully")
    
    def translate(
        self,
        audio_data: np.ndarray,
        source_lang: str,
        target_lang: str,
        sample_rate: int = 16000
    ) -> TranslationResult:
        
        start_time = time.time()
        
        try:
            # Step 1: ASR (Speech to Text)
            asr_start = time.time()
            asr_result = self.asr.transcribe(audio_data, source_lang, sample_rate)
            asr_time = (time.time() - asr_start) * 1000
            
            original_text = asr_result['text']
            asr_confidence = asr_result['confidence']
            
            logger.info(f"ASR ({source_lang}): '{original_text}' (conf: {asr_confidence:.2f})")
            
            if not original_text:
                logger.warning("No speech detected")
                return self._create_empty_result(source_lang, target_lang)
            
            # Step 2: Machine Translation
            mt_start = time.time()
            mt_result = self.mt.translate(original_text, source_lang, target_lang)
            mt_time = (time.time() - mt_start) * 1000
            
            translated_text = mt_result['translated_text']
            logger.info(f"MT ({source_lang}->{target_lang}): '{translated_text}'")
            
            # Step 3: TTS (Text to Speech)
            tts_start = time.time()
            tts_result = self.tts.synthesize(translated_text, target_lang)
            tts_time = (time.time() - tts_start) * 1000
            
            output_audio = tts_result['audio']
            output_sample_rate = tts_result['sample_rate']
            
            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000
            
            logger.info(f"Pipeline latency: {total_latency:.0f}ms "
                       f"(ASR: {asr_time:.0f}ms, MT: {mt_time:.0f}ms, TTS: {tts_time:.0f}ms)")
            
            # Update metrics
            self._update_metrics(total_latency, asr_time, mt_time, tts_time)
            
            # Create result
            result = TranslationResult(
                original_text=original_text,
                translated_text=translated_text,
                audio=output_audio,
                source_lang=source_lang,
                target_lang=target_lang,
                sample_rate=output_sample_rate,
                latency_ms=total_latency,
                asr_confidence=asr_confidence,
                timestamp=time.time()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return self._create_empty_result(source_lang, target_lang)
    
    def translate_streaming(
        self,
        audio_chunks: Generator[np.ndarray, None, None],
        source_lang: str,
        target_lang: str,
        chunk_duration: float = 3.0
    ) -> Generator[TranslationResult, None, None]:

        buffer = np.array([], dtype=np.float32)
        chunk_size = int(16000 * chunk_duration)
        
        for chunk in audio_chunks:
            # Add to buffer
            buffer = np.concatenate([buffer, chunk])
            
            # Process when buffer is large enough
            if len(buffer) >= chunk_size:
                # Process this chunk
                result = self.translate(
                    buffer[:chunk_size],
                    source_lang,
                    target_lang
                )
                
                yield result
                
                # Keep overlap for context
                overlap_size = int(16000 * 0.5)  # 0.5 second overlap
                buffer = buffer[chunk_size - overlap_size:]
        
        # Process remaining audio
        if len(buffer) > 16000:  # At least 1 second
            result = self.translate(buffer, source_lang, target_lang)
            yield result
    
    def translate_file(
        self,
        input_path: str,
        output_path: str,
        source_lang: str,
        target_lang: str
    ) -> TranslationResult:

        import librosa
        import soundfile as sf
        
        # Load audio
        audio_data, sr = librosa.load(input_path, sr=16000, mono=True)
        
        # Translate
        result = self.translate(audio_data, source_lang, target_lang)
        
        # Save output audio
        sf.write(output_path, result.audio, result.sample_rate)
        logger.info(f"Translated audio saved to {output_path}")
        
        return result
    
    def _create_empty_result(
        self,
        source_lang: str,
        target_lang: str
    ) -> TranslationResult:
        """Create empty result for error cases"""
        return TranslationResult(
            original_text="",
            translated_text="",
            audio=np.zeros(16000, dtype=np.float32),
            source_lang=source_lang,
            target_lang=target_lang,
            sample_rate=22050,
            latency_ms=0.0,
            asr_confidence=0.0,
            timestamp=time.time()
        )
    
    def _update_metrics(
        self,
        total_latency: float,
        asr_time: float,
        mt_time: float,
        tts_time: float
    ):
        """Update performance metrics"""
        n = self.metrics['total_translations']
        
        self.metrics['avg_latency'] = (
            (self.metrics['avg_latency'] * n + total_latency) / (n + 1)
        )
        self.metrics['avg_asr_time'] = (
            (self.metrics['avg_asr_time'] * n + asr_time) / (n + 1)
        )
        self.metrics['avg_mt_time'] = (
            (self.metrics['avg_mt_time'] * n + mt_time) / (n + 1)
        )
        self.metrics['avg_tts_time'] = (
            (self.metrics['avg_tts_time'] * n + tts_time) / (n + 1)
        )
        
        self.metrics['total_translations'] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """Get pipeline performance metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            'total_translations': 0,
            'avg_latency': 0.0,
            'avg_asr_time': 0.0,
            'avg_mt_time': 0.0,
            'avg_tts_time': 0.0
        }
    
    def get_pipeline_info(self) -> dict:
        """Get pipeline information"""
        info = {
            'device': self.device,
            'components': {
                'asr': self.asr.get_model_info() if self.asr else None,
                'mt': self.mt.get_model_info() if self.mt else None,
                'tts': self.tts.get_model_info() if self.tts else None
            },
            'metrics': self.get_metrics()
        }
        return info


if __name__ == "__main__":
    print("Translation Pipeline Test")
    print("=" * 50)
    
    from models.asr_model import ASRModel
    from models.mt_model import MTModel
    from models.tts_model import TTSModel
    
    print("\nInitializing models...")
    asr = ASRModel(model_size="base")
    mt = MTModel()
    tts = TTSModel()
    
    print("\nCreating pipeline...")
    pipeline = TranslationPipeline(asr, mt, tts)
    
    # Test with dummy audio (3 seconds of random noise)
    print("\nTesting with dummy audio...")
    test_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
    
    result = pipeline.translate(test_audio, "en", "es")
    
    print(f"\nResults:")
    print(f"Original: {result.original_text}")
    print(f"Translated: {result.translated_text}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Confidence: {result.asr_confidence:.2%}")
    print(f"\nMetrics: {pipeline.get_metrics()}")
    
    print("\n✓ Pipeline test complete!")