"""
RESTful API Endpoints for Advanced ML Modules
Production-ready API with authentication, rate limiting, and comprehensive error handling.
"""

from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle
from django.core.cache import cache
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_headers
import logging
import time
from typing import Dict, Any, List
from functools import wraps

from core.apps.ml.models.career_recommender import CareerRecommender
from core.apps.ml.models.peer_analyzer import PeerAnalyzer
from core.apps.ml.models.anomaly_detector import AnomalyDetector
from core.apps.api.tasks import (
    batch_career_recommendations,
    batch_peer_analysis,
    batch_anomaly_detection,
    comprehensive_student_analysis,
    health_check_ml_modules
)

logger = logging.getLogger(__name__)


# Custom Rate Limiting Classes
class MLAnalysisRateThrottle(UserRateThrottle):
    """Rate limiting for ML analysis endpoints."""
    scope = 'ml_analysis'


class BatchAnalysisRateThrottle(UserRateThrottle):
    """Rate limiting for batch analysis endpoints."""
    scope = 'batch_analysis'


class HealthCheckRateThrottle(UserRateThrottle):
    """Rate limiting for health check endpoints."""
    scope = 'health_check'


# Error Handling Decorator
def api_error_handler(func):
    """Decorator for consistent API error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            response = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            # Add performance metrics to response headers
            if hasattr(response, 'data') and isinstance(response.data, dict):
                response['X-Execution-Time'] = f"{execution_time:.2f}ms"
                response['X-API-Version'] = "v1.0"
            
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error in {func.__name__}: {e}")
            return Response(
                {
                    'error': 'Invalid input parameters',
                    'message': str(e),
                    'error_code': 'VALIDATION_ERROR'
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        except PermissionError as e:
            logger.warning(f"Permission error in {func.__name__}: {e}")
            return Response(
                {
                    'error': 'Insufficient permissions',
                    'message': str(e),
                    'error_code': 'PERMISSION_DENIED'
                },
                status=status.HTTP_403_FORBIDDEN
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return Response(
                {
                    'error': 'Internal server error',
                    'message': 'An unexpected error occurred. Please try again later.',
                    'error_code': 'INTERNAL_ERROR',
                    'execution_time_ms': f"{execution_time:.2f}"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    return wrapper


# Performance Monitoring Decorator
def monitor_performance(func):
    """Decorator to monitor API performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Execute the function
        response = func(*args, **kwargs)
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000
        
        # Log performance metrics
        logger.info(f"API {func.__name__} executed in {execution_time:.2f}ms")
        
        # Store metrics in cache for monitoring
        metrics_key = f"api_metrics_{func.__name__}"
        current_metrics = cache.get(metrics_key, {'total_calls': 0, 'total_time': 0})
        current_metrics['total_calls'] += 1
        current_metrics['total_time'] += execution_time
        current_metrics['avg_time'] = current_metrics['total_time'] / current_metrics['total_calls']
        cache.set(metrics_key, current_metrics, 3600)  # Store for 1 hour
        
        return response
    return wrapper


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([MLAnalysisRateThrottle])
@api_error_handler
@monitor_performance
def get_career_recommendations(request, student_id: str):
    """
    RESTful endpoint for career recommendations.
    
    Args:
        request: HTTP request object
        student_id: Student ID to analyze
        
    Returns:
        Career recommendation results
    """
    try:
        # Check cache first
        cache_key = f"career_rec_api_{student_id}"
        cached_result = cache.get(cache_key)
        
        if cached_result and not request.GET.get('force_refresh'):
            return Response(cached_result)
        
        # Generate recommendations
        recommender = CareerRecommender()
        result = recommender.recommend_careers(student_id)
        
        # Cache the result
        cache.set(cache_key, result, 3600)  # 1 hour cache
        
        return Response(result)
        
    except Exception as e:
        logger.error(f"Career recommendation API failed for {student_id}: {e}")
        return Response(
            {'error': str(e), 'student_id': student_id},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([MLAnalysisRateThrottle])
@api_error_handler
@monitor_performance
def get_peer_analysis(request, student_id: str):
    """
    RESTful endpoint for peer analysis.
    
    Args:
        request: HTTP request object
        student_id: Student ID to analyze
        
    Returns:
        Peer analysis results
    """
    try:
        # Check cache first
        cache_key = f"peer_analysis_api_{student_id}"
        cached_result = cache.get(cache_key)
        
        if cached_result and not request.GET.get('force_refresh'):
            return Response(cached_result)
        
        # Get subjects filter
        subjects = request.GET.get('subjects')
        subject_list = subjects.split(',') if subjects else None
        
        # Generate analysis
        analyzer = PeerAnalyzer()
        result = analyzer.analyze_student_peers(student_id, subject_list)
        
        # Cache the result
        cache.set(cache_key, result, 1800)  # 30 minutes cache
        
        return Response(result)
        
    except Exception as e:
        logger.error(f"Peer analysis API failed for {student_id}: {e}")
        return Response(
            {'error': str(e), 'student_id': student_id},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([MLAnalysisRateThrottle])
@api_error_handler
@monitor_performance
def get_anomaly_detection(request, student_id: str):
    """
    RESTful endpoint for anomaly detection.
    
    Args:
        request: HTTP request object
        student_id: Student ID to analyze
        
    Returns:
        Anomaly detection results
    """
    try:
        # Check cache first
        cache_key = f"anomaly_detection_api_{student_id}"
        cached_result = cache.get(cache_key)
        
        if cached_result and not request.GET.get('force_refresh'):
            return Response(cached_result)
        
        # Get time window
        time_window = int(request.GET.get('time_window', 30))
        
        # Generate analysis with privacy settings from configuration
        anomaly_settings = getattr(settings, 'ANOMALY_DETECTION_SETTINGS', {})
        detector = AnomalyDetector(
            contamination=anomaly_settings.get('CONTAMINATION', 0.1),
            sensitivity=anomaly_settings.get('SENSITIVITY', 0.8),
            epsilon=anomaly_settings.get('EPSILON', 1.0)
        )
        result = detector.detect_anomalies(student_id, time_window)
        
        # Cache the result
        cache.set(cache_key, result, 900)  # 15 minutes cache
        
        return Response(result)
        
    except Exception as e:
        logger.error(f"Anomaly detection API failed for {student_id}: {e}")
        return Response(
            {'error': str(e), 'student_id': student_id},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([MLAnalysisRateThrottle])
@api_error_handler
@monitor_performance
def get_comprehensive_analysis(request, student_id: str):
    """
    RESTful endpoint for comprehensive student analysis.
    
    Args:
        request: HTTP request object
        student_id: Student ID to analyze
        
    Returns:
        Comprehensive analysis results
    """
    try:
        # Check cache first
        cache_key = f"comprehensive_analysis_api_{student_id}"
        cached_result = cache.get(cache_key)
        
        if cached_result and not request.GET.get('force_refresh'):
            return Response(cached_result)
        
        # Check if async processing is requested
        if request.GET.get('async'):
            # Start async task
            task = comprehensive_student_analysis.delay(student_id)
            return Response({
                'task_id': task.id,
                'status': 'processing',
                'message': 'Analysis started in background'
            })
        
        # Perform synchronous analysis with privacy settings
        career_recommender = CareerRecommender()
        peer_analyzer = PeerAnalyzer()
        
        # Initialize anomaly detector with privacy settings
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
        
        career_results = career_recommender.recommend_careers(student_id)
        peer_results = peer_analyzer.analyze_student_peers(student_id)
        anomaly_results = anomaly_detector.detect_anomalies(student_id)
        performance_results = performance_predictor.predict(student_id)
        
        comprehensive_results = {
            'student_id': student_id,
            'career_analysis': career_results,
            'peer_analysis': peer_results,
            'anomaly_analysis': anomaly_results,
            'performance_analysis': performance_results,
            'summary': {
                'career_recommendations_count': len(career_results.get('career_recommendations', [])),
                'peer_group_size': peer_results.get('peer_group_size', 0),
                'anomalies_detected': anomaly_results.get('anomalies_detected', 0),
                'performance_predictions_count': len(performance_results.get('predictions', {}))
            }
        }
        
        # Cache the result
        cache.set(cache_key, comprehensive_results, 1800)  # 30 minutes cache
        
        return Response(comprehensive_results)
        
    except Exception as e:
        logger.error(f"Comprehensive analysis API failed for {student_id}: {e}")
        return Response(
            {'error': str(e), 'student_id': student_id},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([BatchAnalysisRateThrottle])
@api_error_handler
@monitor_performance
def batch_analysis(request):
    """
    RESTful endpoint for batch analysis of multiple students.
    
    Args:
        request: HTTP request object with student_ids list
        
    Returns:
        Batch analysis results
    """
    try:
        student_ids = request.data.get('student_ids', [])
        analysis_type = request.data.get('analysis_type', 'comprehensive')
        
        if not student_ids:
            return Response(
                {'error': 'student_ids is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Start appropriate async task
        if analysis_type == 'career':
            task = batch_career_recommendations.delay(student_ids)
        elif analysis_type == 'peer':
            task = batch_peer_analysis.delay(student_ids)
        elif analysis_type == 'anomaly':
            task = batch_anomaly_detection.delay(student_ids)
        else:
            return Response(
                {'error': 'Invalid analysis_type. Must be career, peer, anomaly, or comprehensive'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response({
            'task_id': task.id,
            'status': 'processing',
            'analysis_type': analysis_type,
            'student_count': len(student_ids),
            'message': f'Batch {analysis_type} analysis started in background'
        })
        
    except Exception as e:
        logger.error(f"Batch analysis API failed: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([HealthCheckRateThrottle])
@api_error_handler
@monitor_performance
def get_ml_health(request):
    """
    RESTful endpoint for ML system health check.
    
    Args:
        request: HTTP request object
        
    Returns:
        Health status of all ML modules
    """
    try:
        # Check cache first
        cache_key = "ml_health_check"
        cached_result = cache.get(cache_key)
        
        if cached_result and not request.GET.get('force_refresh'):
            return Response(cached_result)
        
        # Perform health check
        health_results = health_check_ml_modules.delay()
        result = health_results.get()  # Wait for result
        
        # Cache the result
        cache.set(cache_key, result, 300)  # 5 minutes cache
        
        return Response(result)
        
    except Exception as e:
        logger.error(f"ML health check API failed: {e}")
        return Response(
            {'error': str(e), 'overall_status': 'error'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([HealthCheckRateThrottle])
@api_error_handler
@monitor_performance
def get_task_status(request, task_id: str):
    """
    RESTful endpoint for checking async task status.
    
    Args:
        request: HTTP request object
        task_id: Celery task ID
        
    Returns:
        Task status and results
    """
    try:
        from celery.result import AsyncResult
        
        task_result = AsyncResult(task_id)
        
        response = {
            'task_id': task_id,
            'status': task_result.status,
            'ready': task_result.ready()
        }
        
        if task_result.ready():
            if task_result.successful():
                response['result'] = task_result.result
            else:
                response['error'] = str(task_result.info)
        
        return Response(response)
        
    except Exception as e:
        logger.error(f"Task status API failed for {task_id}: {e}")
        return Response(
            {'error': str(e), 'task_id': task_id},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([HealthCheckRateThrottle])
@api_error_handler
@monitor_performance
def get_system_metrics(request):
    """
    RESTful endpoint for system performance metrics.
    
    Args:
        request: HTTP request object
        
    Returns:
        System performance metrics
    """
    try:
        # Get metrics from all ML modules with privacy settings
        career_recommender = CareerRecommender()
        peer_analyzer = PeerAnalyzer()
        
        # Initialize anomaly detector with privacy settings
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
        
        metrics = {
            'career_recommender': career_recommender.get_system_health(),
            'peer_analyzer': peer_analyzer.get_analysis_health(),
            'anomaly_detector': anomaly_detector.get_detection_health(),
            'performance_predictor': performance_predictor.get_model_health(),
            'cache_stats': {
                'cache_hit_rate': cache.get('cache_hit_rate', 0),
                'total_requests': cache.get('total_requests', 0)
            }
        }
        
        return Response(metrics)
        
    except Exception as e:
        logger.error(f"System metrics API failed: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# New Enhanced Endpoints

@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([MLAnalysisRateThrottle])
@api_error_handler
@monitor_performance
def get_performance_prediction(request, student_id: str):
    """
    RESTful endpoint for performance prediction.
    
    Args:
        request: HTTP request object
        student_id: Student ID to analyze
        
    Returns:
        Performance prediction results
    """
    # Check cache first
    cache_key = f"performance_prediction_api_{student_id}"
    cached_result = cache.get(cache_key)
    
    if cached_result and not request.GET.get('force_refresh'):
        return Response(cached_result)
    
    # Get subjects filter
    subjects = request.GET.get('subjects')
    subject_list = subjects.split(',') if subjects else ['Mathematics']
    
    # Import performance predictor
    from core.apps.ml.models.performance_predictor import PerformancePredictor
    
    # Generate predictions with privacy settings
    prediction_settings = getattr(settings, 'PERFORMANCE_PREDICTION_SETTINGS', {})
    predictor = PerformancePredictor(
        model_version=prediction_settings.get('MODEL_VERSION', 'v1.0'),
        epsilon=prediction_settings.get('EPSILON', 1.0)
    )
    result = predictor.predict(student_id, subject_list)
    
    # Cache the result
    cache.set(cache_key, result, 1800)  # 30 minutes cache
    
    return Response(result)


@api_view(['GET'])
@api_error_handler
@monitor_performance
def api_root(request):
    """
    API root endpoint with available endpoints information.
    Public endpoint - no authentication required.
    """
    api_info = {
        'api_version': 'v1.0',
        'service': 'Smart Student Analytics System (SSAS)',
        'description': 'Production-ready ML-powered educational analytics API',
        'status': 'operational',
        'endpoints': {
            'authentication': {
                'login': '/api/auth/login/',
                'logout': '/api/auth/logout/',
                'logout_all': '/api/auth/logoutall/'
            },
            'student_analytics': {
                'career_recommendations': '/api/students/{student_id}/career-recommendations/',
                'peer_analysis': '/api/students/{student_id}/peer-analysis/',
                'anomaly_detection': '/api/students/{student_id}/anomalies/',
                'performance_prediction': '/api/students/{student_id}/performance-prediction/',
                'comprehensive_analysis': '/api/students/{student_id}/comprehensive-analysis/'
            },
            'batch_processing': {
                'batch_analysis': '/api/batch/analysis/',
                'task_status': '/api/tasks/{task_id}/status/'
            },
            'system_monitoring': {
                'health_check': '/api/system/health/',
                'metrics': '/api/system/metrics/',
                'api_documentation': '/api/docs/'
            }
        },
        'rate_limits': {
            'ml_analysis': '100 requests per hour',
            'batch_analysis': '10 requests per hour',
            'health_check': '200 requests per hour'
        },
        'features': [
            'Real-time ML predictions (<100ms)',
            'GDPR-compliant privacy protection',
            'Comprehensive error handling',
            'Performance monitoring',
            'Automatic caching',
            'Rate limiting',
            'Async batch processing'
        ]
    }
    
    return Response(api_info)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([HealthCheckRateThrottle])
@api_error_handler
@monitor_performance
def get_privacy_compliance(request):
    """
    RESTful endpoint for privacy compliance status.
    
    Args:
        request: HTTP request object
        
    Returns:
        Privacy compliance status and audit summary
    """
    try:
        # Import privacy audit logger
        from core.apps.ml.utils.privacy_audit_logger import get_privacy_compliance_status
        
        # Get compliance status
        compliance_data = get_privacy_compliance_status()
        
        return Response(compliance_data)
        
    except ImportError:
        return Response(
            {'error': 'Privacy audit logger not available'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    except Exception as e:
        logger.error(f"Privacy compliance API failed: {e}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@throttle_classes([HealthCheckRateThrottle])
@api_error_handler
@monitor_performance
def get_api_metrics(request):
    """
    RESTful endpoint for detailed API performance metrics.
    """
    # Get all API metrics from cache
    api_functions = [
        'get_career_recommendations',
        'get_peer_analysis', 
        'get_anomaly_detection',
        'get_performance_prediction',
        'get_comprehensive_analysis',
        'batch_analysis'
    ]
    
    metrics = {}
    total_calls = 0
    total_time = 0
    
    for func_name in api_functions:
        metrics_key = f"api_metrics_{func_name}"
        func_metrics = cache.get(metrics_key, {'total_calls': 0, 'total_time': 0, 'avg_time': 0})
        metrics[func_name] = func_metrics
        total_calls += func_metrics['total_calls']
        total_time += func_metrics['total_time']
    
    # Calculate overall metrics
    overall_avg = total_time / total_calls if total_calls > 0 else 0
    
    response_data = {
        'api_performance': {
            'total_api_calls': total_calls,
            'total_execution_time_ms': round(total_time, 2),
            'average_response_time_ms': round(overall_avg, 2),
            'uptime_status': 'healthy'
        },
        'endpoint_metrics': metrics,
        'performance_targets': {
            'career_recommendations': '< 500ms',
            'peer_analysis': '< 100ms',
            'anomaly_detection': '< 100ms',
            'performance_prediction': '< 200ms'
        },
        'timestamp': time.time()
    }
    
    return Response(response_data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@throttle_classes([MLAnalysisRateThrottle])
@api_error_handler
@monitor_performance
def validate_student_data(request):
    """
    RESTful endpoint for validating student data before analysis.
    """
    student_id = request.data.get('student_id')
    
    if not student_id:
        raise ValueError("student_id is required")
    
    # Import Student model and models for aggregation
    from core.apps.students.models import Student, StudentScore
    from django.db import models
    
    try:
        # Check if student exists
        student = Student.objects.get(student_id=student_id)
        
        # Get student scores
        scores = StudentScore.objects.filter(student=student)
        
        # Validate data quality
        validation_result = {
            'student_id': student_id,
            'student_exists': True,
            'data_quality': {
                'total_scores': scores.count(),
                'subjects_count': scores.values('subject').distinct().count(),
                'academic_years': scores.values('academic_year').distinct().count(),
                'average_score': scores.aggregate(avg_score=models.Avg('total_score'))['avg_score'],
                'data_completeness': 'good' if scores.count() > 10 else 'limited'
            },
            'ml_readiness': {
                'career_analysis': scores.count() >= 5,
                'peer_analysis': scores.count() >= 3,
                'anomaly_detection': scores.count() >= 10,
                'performance_prediction': scores.count() >= 8
            },
            'recommendations': []
        }
        
        # Add recommendations based on data quality
        if scores.count() < 5:
            validation_result['recommendations'].append("More historical data needed for accurate career recommendations")
        if scores.count() < 10:
            validation_result['recommendations'].append("Additional score data would improve anomaly detection accuracy")
        
        return Response(validation_result)
        
    except Student.DoesNotExist:
        return Response({
            'student_id': student_id,
            'student_exists': False,
            'error': 'Student not found in database',
            'error_code': 'STUDENT_NOT_FOUND'
        }, status=status.HTTP_404_NOT_FOUND)
