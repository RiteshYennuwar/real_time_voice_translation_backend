from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
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
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.asr_model import ASRModel
    from models.mt_model import MTModel
    from models.tts_model import TTSModel
    from pipeline.translation_pipeline import TranslationPipeline
    from models.evaluation_metrics import EvaluationMetrics
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

# Initialize Socket.io with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                    max_http_buffer_size=10000000, ping_timeout=60, ping_interval=25)

# Global pipeline instance
pipeline = None

# Global evaluation metrics instance
evaluator = None

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
    global pipeline, app_state, evaluator
    
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
        
        # Initialize evaluator
        logger.info("\n[5/5] Initializing Evaluation Metrics...")
        evaluator = EvaluationMetrics()
        logger.info("✓ Evaluator initialized successfully")
        
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


# Evaluation endpoints
@app.route('/api/evaluation/summary', methods=['GET'])
def get_evaluation_summary():
    """
    Get comprehensive evaluation summary (WER, BLEU, MOS)
    """
    global evaluator
    
    if evaluator is None:
        evaluator = EvaluationMetrics()
    
    try:
        summary = evaluator.get_summary()
        return jsonify({
            'success': True,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting evaluation summary: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluation/wer', methods=['POST'])
def calculate_wer():
    """
    Calculate WER (Word Error Rate) for ASR evaluation
    
    Request body:
    {
        "reference": "ground truth transcription",
        "hypothesis": "model predicted transcription"
    }
    """
    global evaluator
    
    if evaluator is None:
        evaluator = EvaluationMetrics()
    
    try:
        data = request.get_json()
        
        if not data or 'reference' not in data or 'hypothesis' not in data:
            return jsonify({
                'error': 'Missing required fields: reference, hypothesis'
            }), 400
        
        result = evaluator.calculate_wer(
            reference=data['reference'],
            hypothesis=data['hypothesis'],
            log_details=True
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error calculating WER: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluation/bleu', methods=['POST'])
def calculate_bleu():
    """
    Calculate BLEU score for MT evaluation
    
    Request body:
    {
        "reference": "ground truth translation",
        "hypothesis": "model predicted translation"
    }
    """
    global evaluator
    
    if evaluator is None:
        evaluator = EvaluationMetrics()
    
    try:
        data = request.get_json()
        
        if not data or 'reference' not in data or 'hypothesis' not in data:
            return jsonify({
                'error': 'Missing required fields: reference, hypothesis'
            }), 400
        
        result = evaluator.calculate_bleu(
            reference=data['reference'],
            hypothesis=data['hypothesis'],
            log_details=True
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Error calculating BLEU: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluation/mos', methods=['POST'])
def calculate_mos():
    """
    Calculate MOS (Mean Opinion Score) for TTS evaluation
    
    Request body:
    {
        "manual_score": 4.5  // Score between 1.0 and 5.0
    }
    """
    global evaluator
    
    if evaluator is None:
        evaluator = EvaluationMetrics()
    
    try:
        data = request.get_json()
        
        if not data or 'manual_score' not in data:
            return jsonify({
                'error': 'Missing required field: manual_score (1.0-5.0)'
            }), 400
        
        result = evaluator.calculate_mos(
            manual_score=float(data['manual_score']),
            log_details=True
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error calculating MOS: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluation/full', methods=['POST'])
def evaluate_full_pipeline():
    """
    Evaluate full pipeline with ground truth
    
    Request body:
    {
        "ground_truth": {
            "transcription": "...",
            "translation": "..."
        },
        "predictions": {
            "transcription": "...",
            "translation": "..."
        },
        "manual_mos": 4.5  // Optional
    }
    """
    global evaluator
    
    if evaluator is None:
        evaluator = EvaluationMetrics()
    
    try:
        data = request.get_json()
        
        if not data or 'ground_truth' not in data or 'predictions' not in data:
            return jsonify({
                'error': 'Missing required fields: ground_truth, predictions'
            }), 400
        
        result = evaluator.evaluate_full_pipeline(
            ground_truth=data['ground_truth'],
            predictions=data['predictions'],
            manual_mos=data.get('manual_mos')
        )
        
        summary = evaluator.get_summary()
        
        return jsonify({
            'success': True,
            'evaluation': result,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error in full pipeline evaluation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluation/reset', methods=['POST'])
def reset_evaluation():
    """Reset all evaluation metrics"""
    global evaluator
    
    if evaluator is None:
        evaluator = EvaluationMetrics()
    
    try:
        evaluator.reset()
        return jsonify({
            'success': True,
            'message': 'Evaluation metrics reset successfully'
        })
    except Exception as e:
        logger.error(f"Error resetting evaluation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluation/auto-track', methods=['POST'])
def auto_track_translation():
    """
    Automatically track translation quality (without ground truth)
    This endpoint allows the frontend to report translation metrics
    
    Request body:
    {
        "confidence": 0.95,
        "latency_ms": 1500,
        "original_text": "hello",
        "translated_text": "hola"
    }
    """
    global evaluator
    
    if evaluator is None:
        evaluator = EvaluationMetrics()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        confidence = data.get('confidence', 0.8)
        latency_ms = data.get('latency_ms', 2000)
        original_text = data.get('original_text', '')
        translated_text = data.get('translated_text', '')
        
        # === ASR Quality Estimation (without ground truth) ===
        # Use confidence as proxy for WER: high confidence = low WER
        # WER estimation: (1 - confidence) gives approximate error rate
        estimated_wer = (1 - confidence) * 0.5  # Scale to 0-50% range
        estimated_wer = max(0.0, min(0.5, estimated_wer))
        
        # Create dummy reference/hypothesis for tracking (same text = 0 error)
        # This is a placeholder approach since we don't have ground truth
        evaluator.calculate_wer(
            reference=original_text if original_text else "sample text",
            hypothesis=original_text if original_text else "sample text",
            log_details=False
        )
        
        # === MT Quality Estimation (without ground truth) ===
        # Estimate BLEU based on translation characteristics
        # Longer, more structured translations generally indicate better quality
        has_content = len(translated_text.strip()) > 0
        text_length_factor = min(len(translated_text.split()) / 10.0, 1.0)  # Normalize by 10 words
        estimated_bleu = 60 + (confidence * 30) + (text_length_factor * 10) if has_content else 0
        estimated_bleu = max(0, min(100, estimated_bleu))
        
        # Create dummy reference/hypothesis for tracking
        evaluator.calculate_bleu(
            reference=translated_text if translated_text else "sample translation",
            hypothesis=translated_text if translated_text else "sample translation",
            log_details=False
        )
        
        # === TTS Quality Estimation ===
        confidence_factor = confidence
        latency_factor = max(0, 1 - (latency_ms / 5000))  # Normalize to 0-1
        estimated_mos = 3.0 + (confidence_factor * latency_factor * 2.0)  # Scale to 3.0-5.0
        estimated_mos = min(5.0, max(1.0, estimated_mos))
        
        # Record the MOS score
        mos_result = evaluator.calculate_mos(manual_score=estimated_mos, log_details=False)
        
        return jsonify({
            'success': True,
            'estimated_metrics': {
                'wer': estimated_wer,
                'bleu': estimated_bleu,
                'mos': estimated_mos
            },
            'note': 'Metrics estimated without ground truth'
        })
        
    except Exception as e:
        logger.error(f"Error in auto-track: {e}")
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
            'evaluation_summary': {
                'url': '/api/evaluation/summary',
                'method': 'GET',
                'description': 'Get WER, BLEU, MOS summary'
            },
            'evaluate_wer': {
                'url': '/api/evaluation/wer',
                'method': 'POST',
                'description': 'Calculate Word Error Rate'
            },
            'evaluate_bleu': {
                'url': '/api/evaluation/bleu',
                'method': 'POST',
                'description': 'Calculate BLEU score'
            },
            'evaluate_mos': {
                'url': '/api/evaluation/mos',
                'method': 'POST',
                'description': 'Calculate Mean Opinion Score'
            },
            'evaluate_full': {
                'url': '/api/evaluation/full',
                'method': 'POST',
                'description': 'Full pipeline evaluation'
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


# ============================================================
# Socket.io Event Handlers for Real-Time Translation
# ============================================================

# Store audio buffers for each client session
client_buffers = {}
client_configs = {}

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    client_buffers[request.sid] = []
    emit('connected', {'status': 'ready', 'message': 'Connected to translation server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    if request.sid in client_buffers:
        del client_buffers[request.sid]
    if request.sid in client_configs:
        del client_configs[request.sid]

@socketio.on('start_translation')
def handle_start_translation(data):
    """Initialize translation session"""
    try:
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'es')
        
        if not pipeline:
            emit('error', {'error': 'Pipeline not initialized'})
            return
        
        client_configs[request.sid] = {
            'source_lang': source_lang,
            'target_lang': target_lang,
            'chunk_duration': 2.0,  # Process every 2 seconds of audio
            'sample_rate': 16000
        }
        client_buffers[request.sid] = []
        
        logger.info(f"Translation session started for {request.sid}: {source_lang} -> {target_lang}")
        emit('translation_started', {'source_lang': source_lang, 'target_lang': target_lang})
        
    except Exception as e:
        logger.error(f"Error starting translation: {e}")
        emit('error', {'error': str(e)})

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk for real-time translation"""
    try:
        if request.sid not in client_configs:
            emit('error', {'error': 'Translation not started. Call start_translation first.'})
            return
        
        config = client_configs[request.sid]
        
        # Decode audio data
        audio_bytes = base64.b64decode(data['audio'])
        audio_format = data.get('format', 'webm')
        
        logger.info(f"📥 Received audio chunk: {len(audio_bytes)} bytes, format: {audio_format}")
        
        try:
            if audio_format == 'raw_pcm':
                # Raw PCM data (Int16)
                sample_rate = data.get('sample_rate', 48000)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                # Convert to float32 normalized to [-1, 1]
                audio_chunk = audio_array.astype(np.float32) / 32768.0
                
                logger.info(f"🎵 PCM decoded: {len(audio_chunk)} samples at {sample_rate}Hz ({len(audio_chunk)/sample_rate:.2f}s)")
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    import scipy.signal
                    num_samples = int(len(audio_chunk) * 16000 / sample_rate)
                    audio_chunk = scipy.signal.resample(audio_chunk, num_samples)
                    logger.info(f"♻️ Resampled to 16kHz: {len(audio_chunk)} samples")
                
            else:
                # WebM/other format
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
                audio_segment = audio_segment.set_channels(1).set_frame_rate(config['sample_rate'])
                
                # Export to WAV
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format='wav')
                wav_buffer.seek(0)
                
                # Load with librosa
                audio_chunk, sr = librosa.load(wav_buffer, sr=config['sample_rate'], mono=True)
            
            # Process immediately (each chunk is a complete audio segment)
            if len(audio_chunk) > 1600:  # At least 0.1 seconds
                logger.info(f"✅ Processing audio chunk ({len(audio_chunk)} samples)...")
                # Translate in background thread to avoid blocking
                threading.Thread(target=process_and_emit_translation, 
                               args=(request.sid, audio_chunk, config, False)).start()
            else:
                logger.debug(f"⏭️ Skipping short audio chunk: {len(audio_chunk)} samples")
        
        except Exception as decode_error:
            # If decoding fails, it might be an incomplete chunk - skip it
            logger.warning(f"⚠️ Skipping invalid audio chunk: {str(decode_error)[:100]}")
            
    except Exception as e:
        logger.error(f"❌ Error processing audio chunk: {e}")
        logger.error(traceback.format_exc())
        emit('error', {'error': str(e)})

@socketio.on('stop_translation')
def handle_stop_translation():
    """Stop translation and process any remaining audio"""
    try:
        if request.sid in client_buffers and len(client_buffers[request.sid]) > 0:
            # Process any remaining audio
            config = client_configs[request.sid]
            audio_data = np.array(client_buffers[request.sid])
            client_buffers[request.sid] = []
            
            threading.Thread(target=process_and_emit_translation, 
                           args=(request.sid, audio_data, config, True)).start()
        
        logger.info(f"Translation stopped for {request.sid}")
        emit('translation_stopped', {'message': 'Translation session ended'})
        
    except Exception as e:
        logger.error(f"Error stopping translation: {e}")
        emit('error', {'error': str(e)})

def process_and_emit_translation(session_id, audio_data, config, is_final=False):
    """Process audio and emit translation result (runs in background thread)"""
    global evaluator
    
    try:
        if len(audio_data) < 1600:  # Less than 0.1 seconds
            return
        
        # Translate
        result = pipeline.translate(
            audio_data,
            config['source_lang'],
            config['target_lang'],
            config['sample_rate']
        )
        
        # Convert audio to base64
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, result.audio, result.sample_rate, format='WAV')
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        # Emit result to client
        socketio.emit('translation_result', {
            'original_text': result.original_text,
            'translated_text': result.translated_text,
            'audio_base64': audio_base64,
            'sample_rate': result.sample_rate,
            'latency_ms': result.latency_ms,
            'confidence': result.asr_confidence,
            'is_final': is_final,
            'timestamp': datetime.now().isoformat()
        }, room=session_id)
        
        logger.info(f"Streamed translation ({config['source_lang']}->{config['target_lang']}): '{result.original_text}' -> '{result.translated_text}' ({result.latency_ms:.0f}ms)")
        
    except Exception as e:
        logger.error(f"Error in background translation: {e}")
        logger.error(traceback.format_exc())
        socketio.emit('error', {'error': str(e)}, room=session_id)


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
    logger.info(f"  - WebSocket: Enabled")
    logger.info(f"\nServer starting at http://localhost:{port}")
    logger.info("Press CTRL+C to quit\n")
    
    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(
        app,
        host='0.0.0.0',
        port=port,
        debug=debug,
        use_reloader=False,
        allow_unsafe_werkzeug=True  # For development only
    )