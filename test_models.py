from models.asr_model import ASRModel
from models.mt_model import MTModel

print("Testing ASR Model...")
asr = ASRModel(model_size="base")
print("✓ ASR Model loaded")

print("Testing MT Model...")
mt = MTModel()
print("✓ MT Model loaded")