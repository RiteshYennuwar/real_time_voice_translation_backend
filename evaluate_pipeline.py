"""
Test dataset and evaluation runner for the translation pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.evaluation_metrics import EvaluationMetrics
from pipeline.translation_pipeline import TranslationPipeline
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test Dataset with Ground Truth
# Format: (audio_description, ground_truth_transcription, ground_truth_translation)
TEST_DATASET_EN_ES = [
    {
        "id": 1,
        "audio_text": "Hello, how are you today?",
        "ground_truth": {
            "transcription": "Hello, how are you today?",
            "translation": "Hola, ¿cómo estás hoy?"
        },
        "source_lang": "en",
        "target_lang": "es"
    },
    {
        "id": 2,
        "audio_text": "The weather is beautiful today.",
        "ground_truth": {
            "transcription": "The weather is beautiful today.",
            "translation": "El clima está hermoso hoy."
        },
        "source_lang": "en",
        "target_lang": "es"
    },
    {
        "id": 3,
        "audio_text": "I love learning new languages.",
        "ground_truth": {
            "transcription": "I love learning new languages.",
            "translation": "Me encanta aprender nuevos idiomas."
        },
        "source_lang": "en",
        "target_lang": "es"
    },
    {
        "id": 4,
        "audio_text": "Can you help me with this problem?",
        "ground_truth": {
            "transcription": "Can you help me with this problem?",
            "translation": "¿Puedes ayudarme con este problema?"
        },
        "source_lang": "en",
        "target_lang": "es"
    },
    {
        "id": 5,
        "audio_text": "Thank you very much for your assistance.",
        "ground_truth": {
            "transcription": "Thank you very much for your assistance.",
            "translation": "Muchas gracias por su ayuda."
        },
        "source_lang": "en",
        "target_lang": "es"
    }
]

TEST_DATASET_ES_EN = [
    {
        "id": 6,
        "audio_text": "Buenos días, ¿cómo estás?",
        "ground_truth": {
            "transcription": "Buenos días, ¿cómo estás?",
            "translation": "Good morning, how are you?"
        },
        "source_lang": "es",
        "target_lang": "en"
    },
    {
        "id": 7,
        "audio_text": "Me gusta mucho esta ciudad.",
        "ground_truth": {
            "transcription": "Me gusta mucho esta ciudad.",
            "translation": "I really like this city."
        },
        "source_lang": "es",
        "target_lang": "en"
    },
    {
        "id": 8,
        "audio_text": "¿Dónde está la biblioteca?",
        "ground_truth": {
            "transcription": "¿Dónde está la biblioteca?",
            "translation": "Where is the library?"
        },
        "source_lang": "es",
        "target_lang": "en"
    }
]


def generate_synthetic_audio(text: str, sample_rate: int = 16000, duration: float = 2.0) -> np.ndarray:
    """
    Generate synthetic audio for testing (placeholder)
    In production, use real audio recordings
    """
    # Generate silence for now
    # In real evaluation, you would use actual audio recordings
    num_samples = int(sample_rate * duration)
    audio = np.zeros(num_samples, dtype=np.float32)
    
    logger.warning(f"⚠️  Using synthetic audio placeholder for: '{text[:30]}...'")
    return audio


def run_evaluation(use_real_pipeline: bool = False):
    """
    Run comprehensive evaluation on test dataset
    
    Args:
        use_real_pipeline: If True, run through actual pipeline (requires audio)
                          If False, use simulated predictions for demonstration
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting Translation Pipeline Evaluation")
    logger.info("=" * 80)
    
    evaluator = EvaluationMetrics()
    
    if use_real_pipeline:
        # Initialize actual pipeline
        try:
            pipeline = TranslationPipeline()
            logger.info("✅ Translation pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return
    
    # Combine all test datasets
    all_tests = TEST_DATASET_EN_ES + TEST_DATASET_ES_EN
    
    logger.info(f"\n📊 Evaluating {len(all_tests)} test samples\n")
    
    for idx, test_case in enumerate(all_tests, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Test Case {idx}/{len(all_tests)} (ID: {test_case['id']})")
        logger.info(f"Direction: {test_case['source_lang']} → {test_case['target_lang']}")
        logger.info(f"Input Text: {test_case['audio_text']}")
        logger.info(f"{'=' * 60}")
        
        if use_real_pipeline:
            # Run through actual pipeline
            try:
                # Generate audio (in real scenario, load from file)
                audio = generate_synthetic_audio(test_case['audio_text'])
                
                # Process through pipeline
                result = pipeline.process(
                    audio_data=audio,
                    source_language=test_case['source_lang'],
                    target_language=test_case['target_lang']
                )
                
                predictions = {
                    'transcription': result['transcription'],
                    'translation': result['translation'],
                    'audio': result.get('audio_data')
                }
                
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                continue
        else:
            # Simulate predictions with slight variations for demonstration
            predictions = simulate_predictions(test_case)
        
        # Evaluate
        eval_result = evaluator.evaluate_full_pipeline(
            ground_truth=test_case['ground_truth'],
            predictions=predictions,
            manual_mos=4.2  # Simulated MOS score
        )
        
        logger.info(f"\n📈 Results:")
        if 'asr' in eval_result:
            logger.info(f"   ASR - WER: {eval_result['asr']['wer']:.2%}, CER: {eval_result['asr']['cer']:.2%}")
        if 'mt' in eval_result:
            logger.info(f"   MT  - BLEU: {eval_result['mt']['bleu']:.2f}")
        if 'tts' in eval_result:
            logger.info(f"   TTS - MOS: {eval_result['tts']['mos']:.2f}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    summary = evaluator.get_summary()
    
    logger.info("\n🎤 ASR (Automatic Speech Recognition):")
    logger.info(f"   Average WER: {summary['asr']['avg_wer']:.2%} - {summary['asr']['interpretation']}")
    logger.info(f"   Average CER: {summary['asr']['avg_cer']:.2%}")
    logger.info(f"   Total Samples: {summary['asr']['total_samples']}")
    
    logger.info("\n🌐 MT (Machine Translation):")
    logger.info(f"   Average BLEU: {summary['mt']['avg_bleu']:.2f} - {summary['mt']['interpretation']}")
    logger.info(f"   Total Samples: {summary['mt']['total_samples']}")
    
    logger.info("\n🔊 TTS (Text-to-Speech):")
    logger.info(f"   Average MOS: {summary['tts']['avg_mos']:.2f} - {summary['tts']['interpretation']}")
    logger.info(f"   Total Samples: {summary['tts']['total_samples']}")
    
    logger.info("\n⭐ Overall Quality:")
    logger.info(f"   Score: {summary['overall_quality']['score']:.2f}/100")
    logger.info(f"   Assessment: {summary['overall_quality']['interpretation']}")
    
    # Export results
    output_file = "evaluation_results.json"
    evaluator.export_results(output_file)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Evaluation complete! Results saved to {output_file}")
    logger.info("=" * 80)
    
    return evaluator


def simulate_predictions(test_case: dict) -> dict:
    """
    Simulate model predictions for demonstration
    In real evaluation, these would come from the actual pipeline
    """
    # Simulate ASR with slight variations
    transcription = test_case['ground_truth']['transcription']
    
    # Simulate occasional ASR errors (realistic)
    if test_case['id'] % 3 == 0:
        # Introduce minor error
        transcription = transcription.replace('.', '')
    
    # Simulate MT with slight variations
    translation = test_case['ground_truth']['translation']
    
    # Simulate occasional MT variations (realistic)
    if test_case['id'] % 4 == 0:
        # Use a synonym or slightly different phrasing
        translation = translation.replace('?', ' ?')
    
    return {
        'transcription': transcription,
        'translation': translation,
        'audio': None  # TTS audio would be here
    }


def run_quick_test():
    """Quick test of evaluation metrics"""
    logger.info("🧪 Running quick evaluation test\n")
    
    evaluator = EvaluationMetrics()
    
    # Test ASR
    logger.info("1️⃣  Testing ASR (WER/CER):")
    evaluator.calculate_wer(
        reference="Hello how are you today",
        hypothesis="Hello how are you today",
    )
    
    evaluator.calculate_wer(
        reference="The quick brown fox",
        hypothesis="The quik brown fox",  # 1 error
    )
    
    # Test MT
    logger.info("\n2️⃣  Testing MT (BLEU):")
    evaluator.calculate_bleu(
        reference="Hola, ¿cómo estás hoy?",
        hypothesis="Hola, ¿cómo estás hoy?",
    )
    
    evaluator.calculate_bleu(
        reference="El clima está hermoso hoy",
        hypothesis="El tiempo está hermoso hoy",
    )
    
    # Test TTS
    logger.info("\n3️⃣  Testing TTS (MOS):")
    evaluator.calculate_mos(manual_score=4.5)
    evaluator.calculate_mos(manual_score=4.0)
    
    # Summary
    logger.info("\n" + "=" * 60)
    summary = evaluator.get_summary()
    logger.info("📊 Summary:")
    logger.info(f"   ASR: WER={summary['asr']['avg_wer']:.2%} ({summary['asr']['interpretation']})")
    logger.info(f"   MT:  BLEU={summary['mt']['avg_bleu']:.2f} ({summary['mt']['interpretation']})")
    logger.info(f"   TTS: MOS={summary['tts']['avg_mos']:.2f} ({summary['tts']['interpretation']})")
    logger.info(f"   Overall: {summary['overall_quality']['score']:.2f}/100 ({summary['overall_quality']['interpretation']})")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate translation pipeline")
    parser.add_argument(
        '--mode',
        choices=['quick', 'full', 'real'],
        default='full',
        help='Evaluation mode: quick (basic test), full (simulated), real (actual pipeline)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test()
    elif args.mode == 'full':
        run_evaluation(use_real_pipeline=False)
    elif args.mode == 'real':
        run_evaluation(use_real_pipeline=True)
