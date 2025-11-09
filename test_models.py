from models.asr_model import ASRModel
from models.mt_model import MTModel
from models.tts_model import TTSModel

print("Testing ASR Model...")
asr = ASRModel(model_size="base")
print("✓ ASR Model loaded")

print("Testing MT Model...")
mt = MTModel()
print("✓ MT Model loaded")
print("Testing TTS Model...")
tts = TTSModel()
print("✓ TTS Model loaded")
print("\nAll models loaded successfully.")