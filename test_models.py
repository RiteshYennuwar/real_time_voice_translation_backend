import torch
from models.asr_model import ASRModel
from models.mt_model import MTModel
from models.tts_model import TTSModel

print("Testing Torch installation...")
print(f"✓ Torch version: {torch.__version__}")
print("verifying GPU availability...")
print(f"✓ GPU available: {torch.cuda.is_available()}")

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