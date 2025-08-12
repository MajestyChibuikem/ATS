"""
Async Processing Tasks for ML Operations
Handles batch processing and background tasks for scalability.
"""

from celery import shared_task
from typing import List, Dict, Any
import logging
from datetime import datetime

from core.apps.ml.models.career_recommender import CareerRecommender
from core.apps.ml.models.peer_analyzer import PeerAnalyzer
from core.apps.ml.models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


@shared_task
def batch_career_recommendations(student_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Process career recommendations in background for multiple students.
    
    Args:
        student_ids: List of student IDs to process
        
    Returns:
        List of career recommendation results
    """
    try:
        recommender = CareerRecommender()
        results = []
        
        for student_id in student_ids:
            try:
                result = recommender.recommend_careers(student_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process career recommendation for {student_id}: {e}")
                results.append({
                    'student_id': student_id,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch career recommendations failed: {e}")
        return [{'error': str(e), 'status': 'batch_failed'}]


@shared_task
def batch_peer_analysis(student_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Process peer analysis in background for multiple students.
    
    Args:
        student_ids: List of student IDs to process
        
    Returns:
        List of peer analysis results
    """
    try:
        analyzer = PeerAnalyzer()
        results = []
        
        for student_id in student_ids:
            try:
                result = analyzer.analyze_student_peers(student_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process peer analysis for {student_id}: {e}")
                results.append({
                    'student_id': student_id,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch peer analysis failed: {e}")
        return [{'error': str(e), 'status': 'batch_failed'}]


@shared_task
def batch_anomaly_detection(student_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Process anomaly detection in background for multiple students.
    
    Args:
        student_ids: List of student IDs to process
        
    Returns:
        List of anomaly detection results
    """
    try:
        # Initialize with privacy settings
        from django.conf import settings
        anomaly_settings = getattr(settings, 'ANOMALY_DETECTION_SETTINGS', {})
        detector = AnomalyDetector(
            contamination=anomaly_settings.get('CONTAMINATION', 0.1),
            sensitivity=anomaly_settings.get('SENSITIVITY', 0.8),
            epsilon=anomaly_settings.get('EPSILON', 1.0)
        )
        results = []
        
        for student_id in student_ids:
            try:
                result = detector.detect_anomalies(student_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process anomaly detection for {student_id}: {e}")
                results.append({
                    'student_id': student_id,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Batch anomaly detection failed: {e}")
        return [{'error': str(e), 'status': 'batch_failed'}]


@shared_task
def comprehensive_student_analysis(student_id: str) -> Dict[str, Any]:
    """
    Perform comprehensive analysis including all ML modules.
    
    Args:
        student_id: Student ID to analyze
        
    Returns:
        Comprehensive analysis results
    """
    try:
        start_time = datetime.now()
        
        # Initialize all ML modules with privacy settings
        career_recommender = CareerRecommender()
        peer_analyzer = PeerAnalyzer()
        
        # Initialize anomaly detector with privacy settings
        from django.conf import settings
        anomaly_settings = getattr(settings, 'ANOMALY_DETECTION_SETTINGS', {})
        anomaly_detector = AnomalyDetector(
            contamination=anomaly_settings.get('CONTAMINATION', 0.1),
            sensitivity=anomaly_settings.get('SENSITIVITY', 0.8),
            epsilon=anomaly_settings.get('EPSILON', 1.0)
        )
        
        # Initialize performance predictor with privacy settings
        prediction_settings = getattr(settings, 'PERFORMANCE_PREDICTION_SETTINGS', {})
        performance_predictor = PerformancePredictor(
            model_version=prediction_settings.get('MODEL_VERSION', 'v1.0'),
            epsilon=prediction_settings.get('EPSILON', 1.0)
        )
        
        # Perform all analyses
        career_results = career_recommender.recommend_careers(student_id)
        peer_results = peer_analyzer.analyze_student_peers(student_id)
        anomaly_results = anomaly_detector.detect_anomalies(student_id)
        performance_results = performance_predictor.predict(student_id)
        
        # Compile comprehensive results
        comprehensive_results = {
            'student_id': student_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000),
            'career_analysis': career_results,
            'peer_analysis': peer_results,
            'anomaly_analysis': anomaly_results,
            'performance_analysis': performance_results,
            'summary': {
                'career_recommendations_count': len(career_results.get('career_recommendations', [])),
                'peer_group_size': peer_results.get('peer_group_size', 0),
                'anomalies_detected': anomaly_results.get('anomalies_detected', 0),
                'performance_predictions_count': len(performance_results.get('predictions', {})),
                'overall_confidence': min(
                    career_results.get('recommendation_confidence', 0),
                    peer_results.get('analysis_confidence', 1),
                    anomaly_results.get('detection_confidence', 1),
                    performance_results.get('prediction_confidence', 1)
                )
            }
        }
        
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed for {student_id}: {e}")
        return {
            'student_id': student_id,
            'error': str(e),
            'status': 'failed',
            'analysis_timestamp': datetime.now().isoformat()
        }


@shared_task
def health_check_ml_modules() -> Dict[str, Any]:
    """
    Perform health check on all ML modules.
    
    Returns:
        Health status of all ML modules
    """
    try:
        health_results = {}
        
        # Check career recommender
        try:
            career_recommender = CareerRecommender()
            health_results['career_recommender'] = career_recommender.get_system_health()
        except Exception as e:
            health_results['career_recommender'] = {'status': 'error', 'error': str(e)}
        
        # Check peer analyzer
        try:
            peer_analyzer = PeerAnalyzer()
            health_results['peer_analyzer'] = peer_analyzer.get_analysis_health()
        except Exception as e:
            health_results['peer_analyzer'] = {'status': 'error', 'error': str(e)}
        
        # Check anomaly detector with privacy settings
        try:
            from django.conf import settings
            anomaly_settings = getattr(settings, 'ANOMALY_DETECTION_SETTINGS', {})
            anomaly_detector = AnomalyDetector(
                contamination=anomaly_settings.get('CONTAMINATION', 0.1),
                sensitivity=anomaly_settings.get('SENSITIVITY', 0.8),
                epsilon=anomaly_settings.get('EPSILON', 1.0)
            )
            health_results['anomaly_detector'] = anomaly_detector.get_detection_health()
        except Exception as e:
            health_results['anomaly_detector'] = {'status': 'error', 'error': str(e)}
        
        # Check performance predictor with privacy settings
        try:
            prediction_settings = getattr(settings, 'PERFORMANCE_PREDICTION_SETTINGS', {})
            performance_predictor = PerformancePredictor(
                model_version=prediction_settings.get('MODEL_VERSION', 'v1.0'),
                epsilon=prediction_settings.get('EPSILON', 1.0)
            )
            health_results['performance_predictor'] = performance_predictor.get_model_health()
        except Exception as e:
            health_results['performance_predictor'] = {'status': 'error', 'error': str(e)}
        
        # Overall health status
        overall_status = 'healthy'
        if any('error' in str(result) for result in health_results.values()):
            overall_status = 'degraded'
        
        health_results['overall_status'] = overall_status
        health_results['timestamp'] = datetime.now().isoformat()
        
        return health_results
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'overall_status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
