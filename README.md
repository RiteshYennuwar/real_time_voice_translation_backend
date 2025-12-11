# Real-Time Voice Translation Backend

Backend API server for the real-time voice translation system. Supports speech recognition (ASR), machine translation (MT), and text-to-speech (TTS) with automatic quality evaluation.

## Features

- **Automatic Speech Recognition (ASR)**: Whisper model for multilingual transcription
- **Machine Translation (MT)**: Helsinki-NLP models for language translation
- **Text-to-Speech (TTS)**: Coqui TTS for high-quality voice synthesis
- **Real-time Processing**: WebSocket support via Flask-SocketIO
- **Evaluation Metrics**: WER, BLEU, and MOS scoring
- **Multi-language Support**: English, Spanish, Hindi

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- GPU optional (for faster processing)

## Installation

# IMPORTANT NOTE PLEASE MAKE SURE YOU HAVE PYTHNO VERSION 3.10 OR 3.11 BEFORE RUNNING THIS PROJECT, THIS PROJECT WILL NOT RUN ON PYTHON 3.12 OR HIGHER VERSIONS

### 1. Clone the Repository

```bash
git clone https://github.com/RiteshYennuwar/real_time_voice_translation_backend.git
cd real_time_voice_translation_backend
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Installation may take 10-15 minutes as it downloads large ML models and dependencies.

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-change-this-in-production

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000

# Model Configuration
MODEL_CACHE_DIR=./model_cache
```

#### Environment Variables Explained

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FLASK_ENV` | Flask environment mode | `development` | No |
| `FLASK_DEBUG` | Enable Flask debug mode | `True` | No |
| `SECRET_KEY` | Secret key for session security | - | **Yes** |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | `*` | No |
| `MODEL_CACHE_DIR` | Directory to cache downloaded models | `./model_cache` | No |

**Security Note**: Always change `SECRET_KEY` to a strong random string in production. Generate one using:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

## Running the Server

### Development Mode

```bash
cd api
python main.py
```

The server will start on `http://localhost:5000`


# AT THIS POINT PLEASE VISIT https://github.com/RiteshYennuwar/real_time_voice_translation_frontend and install the frontend

### Production Mode

```bash
# Set production environment
export FLASK_ENV=production  # Linux/Mac
set FLASK_ENV=production     # Windows CMD
$env:FLASK_ENV="production"  # Windows PowerShell

# Run with production settings
cd api
python main.py
```

## API Endpoints

### Translation Endpoints

#### WebSocket Connection
```
ws://localhost:5000/socket.io/
```

**Events:**
- `audio_chunk` - Send audio data for real-time translation
- `translation_result` - Receive translation results

#### REST Endpoints

- `GET /` - Health check
- `GET /api/health` - Detailed health status
- `POST /api/translate` - Translate audio file

### Evaluation Endpoints

- `GET /api/evaluation/summary` - Get evaluation metrics summary
- `POST /api/evaluation/wer` - Calculate Word Error Rate
- `POST /api/evaluation/bleu` - Calculate BLEU score
- `POST /api/evaluation/mos` - Record Mean Opinion Score
- `POST /api/evaluation/auto-track` - Auto-track metrics
- `POST /api/evaluation/reset` - Reset all metrics
- `GET /api/evaluation/full` - Get detailed evaluation report

## Project Structure

```
real_time_voice_translation_backend/
├── api/
│   └── main.py              # Flask API server
├── models/
│   ├── asr_model.py         # Speech recognition model
│   ├── mt_model.py          # Machine translation model
│   ├── tts_model.py         # Text-to-speech model
│   └── evaluation_metrics.py # Evaluation metrics
├── pipeline/
│   └── translation_pipeline.py # Translation pipeline
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
├── evaluate_pipeline.py     # Evaluation test suite
└── README.md               # This file
```

## Testing

### Run Model Tests
```bash
python test_models.py
```

### Run Pipeline Tests
```bash
python test_pipeline.py
```

### Run Evaluation Tests
```bash
# Quick evaluation (simulated)
python evaluate_pipeline.py --mode quick

# Full evaluation (with real models)
python evaluate_pipeline.py --mode full
```

### Test API Endpoints
```bash
# Check server health
curl http://localhost:5000/api/health

# Get evaluation summary
curl http://localhost:5000/api/evaluation/summary
```

## Supported Languages

| Language | Code | ASR | MT | TTS |
|----------|------|-----|----|----|
| English | `en` | ✅ | ✅ | ✅ |
| Spanish | `es` | ✅ | ✅ | ✅ |
| Hindi | `hi` | ✅ | ✅ | ✅ |

## Evaluation Metrics

The system automatically tracks three quality metrics:

1. **WER (Word Error Rate)**: ASR transcription accuracy (0-100%, lower is better)
2. **BLEU Score**: Translation quality (0-100, higher is better)
3. **MOS (Mean Opinion Score)**: Voice synthesis quality (1-5, higher is better)

Access metrics at: `http://localhost:5000/api/evaluation/summary`

## Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure virtual environment is activated
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

**2. CUDA/GPU Errors**
```bash
# Force CPU-only installation
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**3. Port Already in Use**
```python
# Change port in api/main.py
socketio.run(app, host='0.0.0.0', port=5001, debug=True)
```

**4. Model Download Issues**
- Check internet connection
- Models are auto-downloaded on first run (~2-3GB)
- Clear cache: `rm -rf model_cache/` and restart

**5. Memory Issues**
- Use smaller models (edit `models/*.py` files)
- Reduce audio chunk sizes
- Close other applications

## Performance Optimization

### For CPU-Only Systems
- Use Whisper base/small models instead of large
- Reduce audio sample rate (8000 or 16000 Hz)
- Process in larger chunks

### For GPU Systems
- Install CUDA-enabled PyTorch
- Set `device='cuda'` in model files
- Increase batch sizes for better throughput
