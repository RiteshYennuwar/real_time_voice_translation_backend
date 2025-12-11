"""
Evaluation Metrics for Real-Time Voice Translation System
Implements WER (Word Error Rate), BLEU score, and MOS (Mean Opinion Score)
"""

import numpy as np
from typing import Dict, List, Optional
import logging
from jiwer import wer, cer
from sacrebleu.metrics import BLEU
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for voice translation pipeline
    
    Metrics:
    - WER (Word Error Rate): ASR quality - lower is better (0-1)
    - CER (Character Error Rate): ASR quality - lower is better (0-1)
    - BLEU: Translation quality - higher is better (0-100)
    - MOS (Mean Opinion Score): TTS quality - higher is better (1-5)
    """
    
    def __init__(self):
        self.bleu = BLEU()
        
        # Storage for evaluation results
        self.asr_evaluations = []
        self.mt_evaluations = []
        self.tts_evaluations = []
        
        # Aggregate metrics
        self.metrics_summary = {
            'asr': {
                'wer': [],
                'cer': [],
                'avg_wer': 0.0,
                'avg_cer': 0.0,
                'total_samples': 0
            },
            'mt': {
                'bleu': [],
                'avg_bleu': 0.0,
                'total_samples': 0
            },
            'tts': {
                'mos': [],
                'avg_mos': 0.0,
                'total_samples': 0
            }
        }
    
    def calculate_wer(
        self, 
        reference: str, 
        hypothesis: str,
        log_details: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Word Error Rate (WER) for ASR evaluation
        
        WER = (Substitutions + Deletions + Insertions) / Total Words
        
        Args:
            reference: Ground truth transcription
            hypothesis: Model predicted transcription
            log_details: Whether to log detailed results
            
        Returns:
            Dict with WER, CER, and error breakdown
        """
        try:
            # Calculate WER (0-1, lower is better)
            wer_score = wer(reference, hypothesis)
            
            # Calculate CER (Character Error Rate)
            cer_score = cer(reference, hypothesis)
            
            result = {
                'wer': round(wer_score, 4),
                'cer': round(cer_score, 4),
                'reference': reference,
                'hypothesis': hypothesis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store for aggregate metrics
            self.asr_evaluations.append(result)
            self.metrics_summary['asr']['wer'].append(wer_score)
            self.metrics_summary['asr']['cer'].append(cer_score)
            self.metrics_summary['asr']['total_samples'] += 1
            
            # Update averages
            self.metrics_summary['asr']['avg_wer'] = np.mean(self.metrics_summary['asr']['wer'])
            self.metrics_summary['asr']['avg_cer'] = np.mean(self.metrics_summary['asr']['cer'])
            
            if log_details:
                logger.info(f"📊 ASR Evaluation - WER: {wer_score:.2%}, CER: {cer_score:.2%}")
                logger.info(f"   REF: {reference}")
                logger.info(f"   HYP: {hypothesis}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating WER: {e}")
            return {'wer': None, 'cer': None, 'error': str(e)}
    
    def calculate_bleu(
        self,
        reference: str,
        hypothesis: str,
        log_details: bool = True
    ) -> Dict[str, float]:
        """
        Calculate BLEU score for Machine Translation evaluation
        
        BLEU measures n-gram precision with brevity penalty (0-100, higher is better)
        
        Args:
            reference: Ground truth translation
            hypothesis: Model predicted translation
            log_details: Whether to log detailed results
            
        Returns:
            Dict with BLEU score and components
        """
        try:
            # SacreBLEU expects list of references and single hypothesis
            bleu_result = self.bleu.corpus_score([hypothesis], [[reference]])
            
            result = {
                'bleu': round(bleu_result.score, 2),
                'bleu_1': round(bleu_result.precisions[0], 2) if len(bleu_result.precisions) > 0 else 0,
                'bleu_2': round(bleu_result.precisions[1], 2) if len(bleu_result.precisions) > 1 else 0,
                'bleu_3': round(bleu_result.precisions[2], 2) if len(bleu_result.precisions) > 2 else 0,
                'bleu_4': round(bleu_result.precisions[3], 2) if len(bleu_result.precisions) > 3 else 0,
                'brevity_penalty': round(bleu_result.bp, 4),
                'reference': reference,
                'hypothesis': hypothesis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store for aggregate metrics
            self.mt_evaluations.append(result)
            self.metrics_summary['mt']['bleu'].append(bleu_result.score)
            self.metrics_summary['mt']['total_samples'] += 1
            
            # Update average
            self.metrics_summary['mt']['avg_bleu'] = np.mean(self.metrics_summary['mt']['bleu'])
            
            if log_details:
                logger.info(f"📊 MT Evaluation - BLEU: {bleu_result.score:.2f}")
                logger.info(f"   REF: {reference}")
                logger.info(f"   HYP: {hypothesis}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
            return {'bleu': None, 'error': str(e)}
    
    def calculate_mos(
        self,
        audio_samples: Optional[np.ndarray] = None,
        manual_score: Optional[float] = None,
        log_details: bool = True
    ) -> Dict[str, float]:
        """
        Calculate Mean Opinion Score (MOS) for TTS evaluation
        
        MOS is typically a human subjective score (1-5, higher is better)
        Since automated MOS prediction requires complex models, we support:
        1. Manual scoring input
        2. Placeholder for future automated MOS models
        
        Args:
            audio_samples: Audio waveform for automated evaluation (future)
            manual_score: Manual MOS score from human evaluation
            log_details: Whether to log detailed results
            
        Returns:
            Dict with MOS score
        """
        try:
            if manual_score is not None:
                # Use provided manual score
                if not (1.0 <= manual_score <= 5.0):
                    raise ValueError("MOS score must be between 1.0 and 5.0")
                
                mos_score = manual_score
                method = "manual"
                
            elif audio_samples is not None:
                # Placeholder for automated MOS prediction
                # In production, use models like MOSNet, DNSMOS, or NISQA
                logger.warning("⚠️  Automated MOS calculation not implemented. Using placeholder.")
                mos_score = 3.5  # Neutral placeholder
                method = "automated_placeholder"
                
            else:
                raise ValueError("Either audio_samples or manual_score must be provided")
            
            result = {
                'mos': round(mos_score, 2),
                'method': method,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store for aggregate metrics
            self.tts_evaluations.append(result)
            self.metrics_summary['tts']['mos'].append(mos_score)
            self.metrics_summary['tts']['total_samples'] += 1
            
            # Update average
            self.metrics_summary['tts']['avg_mos'] = np.mean(self.metrics_summary['tts']['mos'])
            
            if log_details:
                logger.info(f"📊 TTS Evaluation - MOS: {mos_score:.2f} ({method})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating MOS: {e}")
            return {'mos': None, 'error': str(e)}
    
    def evaluate_full_pipeline(
        self,
        ground_truth: Dict[str, str],
        predictions: Dict[str, any],
        manual_mos: Optional[float] = None
    ) -> Dict[str, Dict]:
        """
        Evaluate the entire translation pipeline
        
        Args:
            ground_truth: Dict with 'transcription' and 'translation'
            predictions: Dict with 'transcription', 'translation', and 'audio'
            manual_mos: Optional manual MOS score for TTS
            
        Returns:
            Dict with all evaluation metrics
        """
        results = {}
        
        # ASR Evaluation
        if 'transcription' in ground_truth and 'transcription' in predictions:
            results['asr'] = self.calculate_wer(
                reference=ground_truth['transcription'],
                hypothesis=predictions['transcription'],
                log_details=True
            )
        
        # MT Evaluation
        if 'translation' in ground_truth and 'translation' in predictions:
            results['mt'] = self.calculate_bleu(
                reference=ground_truth['translation'],
                hypothesis=predictions['translation'],
                log_details=True
            )
        
        # TTS Evaluation
        if manual_mos is not None or 'audio' in predictions:
            results['tts'] = self.calculate_mos(
                audio_samples=predictions.get('audio'),
                manual_score=manual_mos,
                log_details=True
            )
        
        return results
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of all evaluations"""
        summary = {
            'asr': {
                'avg_wer': round(self.metrics_summary['asr']['avg_wer'], 4),
                'avg_cer': round(self.metrics_summary['asr']['avg_cer'], 4),
                'total_samples': self.metrics_summary['asr']['total_samples'],
                'interpretation': self._interpret_wer(self.metrics_summary['asr']['avg_wer'])
            },
            'mt': {
                'avg_bleu': round(self.metrics_summary['mt']['avg_bleu'], 2),
                'total_samples': self.metrics_summary['mt']['total_samples'],
                'interpretation': self._interpret_bleu(self.metrics_summary['mt']['avg_bleu'])
            },
            'tts': {
                'avg_mos': round(self.metrics_summary['tts']['avg_mos'], 2),
                'total_samples': self.metrics_summary['tts']['total_samples'],
                'interpretation': self._interpret_mos(self.metrics_summary['tts']['avg_mos'])
            },
            'overall_quality': self._calculate_overall_quality()
        }
        
        return summary
    
    def _interpret_wer(self, wer: float) -> str:
        """Interpret WER score"""
        if wer < 0.05:
            return "Excellent (< 5%)"
        elif wer < 0.10:
            return "Very Good (5-10%)"
        elif wer < 0.20:
            return "Good (10-20%)"
        elif wer < 0.30:
            return "Fair (20-30%)"
        else:
            return "Poor (> 30%)"
    
    def _interpret_bleu(self, bleu: float) -> str:
        """Interpret BLEU score"""
        if bleu > 50:
            return "Excellent (> 50)"
        elif bleu > 40:
            return "Very Good (40-50)"
        elif bleu > 30:
            return "Good (30-40)"
        elif bleu > 20:
            return "Fair (20-30)"
        else:
            return "Poor (< 20)"
    
    def _interpret_mos(self, mos: float) -> str:
        """Interpret MOS score"""
        if mos >= 4.5:
            return "Excellent (4.5-5.0)"
        elif mos >= 4.0:
            return "Very Good (4.0-4.5)"
        elif mos >= 3.5:
            return "Good (3.5-4.0)"
        elif mos >= 3.0:
            return "Fair (3.0-3.5)"
        else:
            return "Poor (< 3.0)"
    
    def _calculate_overall_quality(self) -> Dict:
        """Calculate overall quality score"""
        # Weighted average: ASR (30%), MT (40%), TTS (30%)
        scores = []
        weights = []
        
        if self.metrics_summary['asr']['total_samples'] > 0:
            # WER: convert to quality score (1 - WER)
            asr_quality = (1 - self.metrics_summary['asr']['avg_wer']) * 100
            scores.append(asr_quality)
            weights.append(0.30)
        
        if self.metrics_summary['mt']['total_samples'] > 0:
            # BLEU: already 0-100 scale
            scores.append(self.metrics_summary['mt']['avg_bleu'])
            weights.append(0.40)
        
        if self.metrics_summary['tts']['total_samples'] > 0:
            # MOS: convert to 0-100 scale
            mos_quality = (self.metrics_summary['tts']['avg_mos'] / 5.0) * 100
            scores.append(mos_quality)
            weights.append(0.30)
        
        if scores:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            overall_score = sum(s * w for s, w in zip(scores, normalized_weights))
            
            return {
                'score': round(overall_score, 2),
                'interpretation': self._interpret_overall(overall_score)
            }
        
        return {'score': None, 'interpretation': 'No data'}
    
    def _interpret_overall(self, score: float) -> str:
        """Interpret overall quality score"""
        if score >= 90:
            return "Excellent System"
        elif score >= 80:
            return "Very Good System"
        elif score >= 70:
            return "Good System"
        elif score >= 60:
            return "Fair System"
        else:
            return "Needs Improvement"
    
    def export_results(self, filepath: str):
        """Export evaluation results to JSON file"""
        try:
            results = {
                'summary': self.get_summary(),
                'asr_evaluations': self.asr_evaluations,
                'mt_evaluations': self.mt_evaluations,
                'tts_evaluations': self.tts_evaluations,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Evaluation results exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
    
    def reset(self):
        """Reset all evaluation metrics"""
        self.asr_evaluations = []
        self.mt_evaluations = []
        self.tts_evaluations = []
        
        self.metrics_summary = {
            'asr': {
                'wer': [],
                'cer': [],
                'avg_wer': 0.0,
                'avg_cer': 0.0,
                'total_samples': 0
            },
            'mt': {
                'bleu': [],
                'avg_bleu': 0.0,
                'total_samples': 0
            },
            'tts': {
                'mos': [],
                'avg_mos': 0.0,
                'total_samples': 0
            }
        }
        
        logger.info("🔄 Evaluation metrics reset")
