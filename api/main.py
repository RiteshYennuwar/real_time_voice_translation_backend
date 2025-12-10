from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import base64
import io
import librosa
import soundfile as sf
import logging
import traceback
from datetime import datetime
import os
import sys
import tempfile
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.asr_model import ASRModel
    from models.mt_model import MTModel
    from models.tts_model import TTSModel
    from pipeline.translation_pipeline import TranslationPipeline
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import models: {e}")
    MODELS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS - allow all localhost origins for development
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins for development
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False,
        "max_age": 3600
    }
})

# Global pipeline instance
pipeline = None

# Supported languages configuration
SUPPORTED_LANGUAGES = ['en', 'es', 'hi']
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'hi': 'Hindi'
}

# Application state
app_state = {
    'initialized': False,
    'models_loaded': False,
    'error': None,
    'startup_time': None
}

#initialize models and pipeline
def initialize_pipeline():
    global pipeline, app_state
    
    try:
        logger.info("=" * 60)
        logger.info("Initializing Voice Translation Pipeline")
        logger.info("=" * 60)
        
        if not MODELS_AVAILABLE:
            logger.error("Models not available. Please install required packages.")
            app_state['error'] = "Models not available"
            return False
        
        # Initialize ASR Model
        logger.info("\n[1/4] Loading ASR Model (Whisper)...")
        asr = ASRModel(model_size="base")
        logger.info("✓ ASR Model loaded successfully")
        
        # Initialize MT Model
        logger.info("\n[2/4] Loading MT Models (MarianMT)...")
        mt = MTModel()
        logger.info("✓ MT Models loaded successfully")
        
        # Initialize TTS Model
        logger.info("\n[3/4] Loading TTS Models (Coqui)...")
        tts = TTSModel()
        logger.info("✓ TTS Models loaded successfully")
        
        # Create pipeline
        logger.info("\n[4/4] Creating Translation Pipeline...")
        pipeline = TranslationPipeline(asr, mt, tts)
        logger.info("✓ Pipeline created successfully")
        
        # Update state
        app_state['initialized'] = True
        app_state['models_loaded'] = True
        app_state['startup_time'] = datetime.now().isoformat()
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL SYSTEMS READY")
        logger.info("=" * 60)
        logger.info(f"Supported Languages: {', '.join(LANGUAGE_NAMES.values())}")
        logger.info("API Server is ready to accept requests")
        logger.info("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Failed to initialize pipeline: {e}")
        logger.error(traceback.format_exc())
        app_state['error'] = str(e)
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    Returns system status and readiness
    """
    logger.info(f"Health check request from {request.remote_addr}")
    return jsonify({
        'status': 'healthy' if app_state['initialized'] else 'initializing',
        'pipeline_ready': pipeline is not None,
        'models_loaded': app_state['models_loaded'],
        'error': app_state['error'],
        'startup_time': app_state['startup_time'],
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200 if pipeline else 503


@app.route('/status', methods=['GET'])
def get_status():
    """
    Detailed status endpoint
    Returns comprehensive system information
    """
    status = {
        'system': {
            'initialized': app_state['initialized'],
            'models_loaded': app_state['models_loaded'],
            'error': app_state['error'],
            'startup_time': app_state['startup_time']
        },
        'supported_languages': LANGUAGE_NAMES,
        'endpoints': {
            'health': '/health',
            'translate': '/api/translate',
            'languages': '/api/languages',
            'metrics': '/api/metrics'
        }
    }
    
    if pipeline:
        try:
            status['pipeline'] = pipeline.get_pipeline_info()
        except:
            pass
    
    return jsonify(status)

@app.route('/api/languages', methods=['GET'])
def get_languages():
    return jsonify({
        'languages': [
            {
                'code': code,
                'name': LANGUAGE_NAMES[code],
                'flag': {'en': '🇺🇸', 'es': '🇪🇸', 'hi': '🇮🇳'}.get(code, '')
            }
            for code in SUPPORTED_LANGUAGES
        ],
        'count': len(SUPPORTED_LANGUAGES)
    })


@app.route('/api/language-pairs', methods=['GET'])
def get_language_pairs():
    if not pipeline:
        return jsonify({'error': 'Pipeline not initialized'}), 503
    
    try:
        pairs = []
        for src in SUPPORTED_LANGUAGES:
            for tgt in SUPPORTED_LANGUAGES:
                if src != tgt:
                    pairs.append({
                        'source': {'code': src, 'name': LANGUAGE_NAMES[src]},
                        'target': {'code': tgt, 'name': LANGUAGE_NAMES[tgt]}
                    })
        
        return jsonify({
            'pairs': pairs,
            'count': len(pairs)
        })
    except Exception as e:
        logger.error(f"Error getting language pairs: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/translate', methods=['POST'])
def translate_audio():
    """
    Translate audio file
    
    Expected form data:
    - audio: audio file (wav, mp3, etc.)
    - source_lang: source language code (en, es, hi)
    - target_lang: target language code (en, es, hi)
    
    Returns:
    - original_text: transcribed text
    - translated_text: translated text
    - audio_base64: synthesized audio in base64
    - metrics: latency, confidence, etc.
    """
    try:
        if not pipeline:
            return jsonify({
                'error': 'Pipeline not initialized',
                'status': 'service_unavailable'
            }), 503
        
        # Get form data
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        source_lang = request.form.get('source_lang', 'en')
        target_lang = request.form.get('target_lang', 'es')
        
        # Validate languages
        if source_lang not in SUPPORTED_LANGUAGES:
            return jsonify({'error': f'Unsupported source language: {source_lang}'}), 400
        
        if target_lang not in SUPPORTED_LANGUAGES:
            return jsonify({'error': f'Unsupported target language: {target_lang}'}), 400
        
        if source_lang == target_lang:
            return jsonify({'error': 'Source and target languages must be different'}), 400
        
        logger.info(f"Translation request: {source_lang} → {target_lang}")
        
        # Load audio - handle various formats including WebM from browser
        audio_bytes = audio_file.read()
        filename = audio_file.filename or 'audio.webm'
        
        try:
            # Try loading directly with librosa first (works for WAV, MP3, etc.)
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        except Exception as e:
            logger.info(f"Direct load failed ({e}), trying pydub conversion...")
            
            # Use pydub to convert from WebM/Opus to WAV
            try:
                # Determine format from filename
                if filename.endswith('.webm'):
                    audio_format = 'webm'
                elif filename.endswith('.ogg'):
                    audio_format = 'ogg'
                elif filename.endswith('.mp3'):
                    audio_format = 'mp3'
                else:
                    audio_format = 'webm'  # Default for browser recordings
                
                # Convert using pydub
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                
                # Convert to mono and set sample rate
                audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                
                # Export to WAV in memory
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format='wav')
                wav_buffer.seek(0)
                
                # Now load with librosa
                audio_data, sr = librosa.load(wav_buffer, sr=16000, mono=True)
                
            except Exception as conv_error:
                logger.error(f"Audio conversion failed: {conv_error}")
                raise ValueError(f"Could not process audio file. Format: {filename}. Error: {conv_error}")
        
        logger.info(f"Audio loaded: {len(audio_data)} samples, {sr} Hz, duration: {len(audio_data)/sr:.2f}s")
        
        # Translate
        result = pipeline.translate(audio_data, source_lang, target_lang, sr)
        
        # Convert audio to base64
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, result.audio, result.sample_rate, format='WAV')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        logger.info(f"Translation completed in {result.latency_ms:.0f}ms")
        
        return jsonify({
            'success': True,
            'original_text': result.original_text,
            'translated_text': result.translated_text,
            'source_lang': result.source_lang,
            'target_lang': result.target_lang,
            'audio_base64': audio_base64,
            'sample_rate': result.sample_rate,
            'metrics': {
                'latency_ms': result.latency_ms,
                'confidence': result.asr_confidence,
                'audio_duration_sec': len(result.audio) / result.sample_rate,
                'timestamp': result.timestamp
            }
        })
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500


@app.route('/api/translate-text', methods=['POST'])
def translate_text_only():
    """
    Translate text without audio (MT only)
    
    Expected JSON:
    {
        "text": "Hello world",
        "source_lang": "en",
        "target_lang": "es"
    }
    """
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'es')
        
        # Validate
        if source_lang not in SUPPORTED_LANGUAGES or target_lang not in SUPPORTED_LANGUAGES:
            return jsonify({'error': 'Unsupported language'}), 400
        
        # Translate using MT model only
        result = pipeline.mt.translate(text, source_lang, target_lang)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'translated_text': result['translated_text'],
            'source_lang': source_lang,
            'target_lang': target_lang
        })
        
    except Exception as e:
        logger.error(f"Text translation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/synthesize', methods=['POST'])
def synthesize_speech():
    """
    Synthesize speech from text (TTS only)
    
    Expected JSON:
    {
        "text": "Hello world",
        "language": "en",
        "speed": 1.0
    }
    """
    try:
        if not pipeline:
            return jsonify({'error': 'Pipeline not initialized'}), 503
        
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        language = data.get('language', 'en')
        speed = data.get('speed', 1.0)
        
        # Validate
        if language not in SUPPORTED_LANGUAGES:
            return jsonify({'error': f'Unsupported language: {language}'}), 400
        
        # Synthesize using TTS model only
        result = pipeline.tts.synthesize(text, language, speed=speed)
        
        # Convert to base64
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, result['audio'], result['sample_rate'], format='WAV')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'text': text,
            'language': language,
            'audio_base64': audio_base64,
            'sample_rate': result['sample_rate'],
            'duration_sec': len(result['audio']) / result['sample_rate']
        })
        
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        return jsonify({'error': str(e)}), 500

# api for metrics and monitoring
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    Get pipeline performance metrics
    Returns aggregated statistics
    """
    if not pipeline:
        return jsonify({'error': 'Pipeline not initialized'}), 503
    
    try:
        metrics = pipeline.get_metrics()
        return jsonify({
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics/reset', methods=['POST'])
def reset_metrics():
    """Reset pipeline metrics"""
    if not pipeline:
        return jsonify({'error': 'Pipeline not initialized'}), 503
    
    try:
        pipeline.reset_metrics()
        return jsonify({
            'success': True,
            'message': 'Metrics reset successfully'
        })
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API documentation"""
    return jsonify({
        'name': 'Voice Translation API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': {
                'url': '/health',
                'method': 'GET',
                'description': 'Health check'
            },
            'status': {
                'url': '/status',
                'method': 'GET',
                'description': 'Detailed system status'
            },
            'languages': {
                'url': '/api/languages',
                'method': 'GET',
                'description': 'List supported languages'
            },
            'translate': {
                'url': '/api/translate',
                'method': 'POST',
                'description': 'Translate audio file'
            },
            'translate_text': {
                'url': '/api/translate-text',
                'method': 'POST',
                'description': 'Translate text only'
            },
            'synthesize': {
                'url': '/api/synthesize',
                'method': 'POST',
                'description': 'Synthesize speech from text'
            },
            'metrics': {
                'url': '/api/metrics',
                'method': 'GET',
                'description': 'Get performance metrics'
            }
        },
        'documentation': 'https://github.com/RiteshYennuwar/real_time_voice_translation_backend',
        'frontend': 'https://github.com/RiteshYennuwar/real_time_voice_translation_frontend'
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 404,
        'available_endpoints': [
            '/',
            '/health',
            '/status',
            '/api/languages',
            '/api/translate',
            '/api/translate-text',
            '/api/synthesize',
            '/api/metrics'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'status': 500
    }), 500



if __name__ == '__main__':
    # Initialize pipeline before starting server
    logger.info("Starting Voice Translation API Server...")
    
    if not initialize_pipeline():
        logger.error("Failed to initialize pipeline")
        logger.warning("Starting server anyway (will return 503 for requests)")
    
    # Run Flask server
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'  # Changed default to False
    
    logger.info(f"\nServer Configuration:")
    logger.info(f"  - Port: {port}")
    logger.info(f"  - Debug: {debug}")
    logger.info(f"  - Host: 0.0.0.0")
    logger.info(f"\nServer starting at http://localhost:{port}")
    logger.info("Press CTRL+C to quit\n")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True,
        use_reloader=False  # Disable reloader to avoid threading issues
    )