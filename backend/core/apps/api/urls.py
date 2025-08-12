"""
URL configuration for the API app.

This module defines the URL patterns for the SSAS API endpoints.
"""

from django.urls import path, include
from knox import views as knox_views

from . import main_views
from .views import (
    unified_predictor,
    unified_predictor_enhanced,
    batch_predictor,
    health_monitor
)

app_name = 'api'

urlpatterns = [
    # API Root
    path('', main_views.api_root, name='api-root'),
    
    # Authentication endpoints
    path('auth/', include('knox.urls')),
    path('auth/logout/', knox_views.LogoutView.as_view(), name='knox_logout'),
    path('auth/logoutall/', knox_views.LogoutAllView.as_view(), name='knox_logoutall'),
    
    # ML Module endpoints
    path('students/<str:student_id>/career-recommendations/', main_views.get_career_recommendations, name='career-recommendations'),
    path('students/<str:student_id>/peer-analysis/', main_views.get_peer_analysis, name='peer-analysis'),
    path('students/<str:student_id>/anomalies/', main_views.get_anomaly_detection, name='anomaly-detection'),
    path('students/<str:student_id>/performance-prediction/', main_views.get_performance_prediction, name='performance-prediction'),
    path('students/<str:student_id>/comprehensive-analysis/', main_views.get_comprehensive_analysis, name='comprehensive-analysis'),
    
    # Data validation endpoints
    path('students/validate/', main_views.validate_student_data, name='validate-student-data'),
    
    # Batch processing endpoints
    path('batch/analysis/', main_views.batch_analysis, name='batch-analysis'),
    path('tasks/<str:task_id>/status/', main_views.get_task_status, name='task-status'),
    
    # System endpoints
    path('system/health/', main_views.get_ml_health, name='ml-health'),
    path('system/metrics/', main_views.get_system_metrics, name='system-metrics'),
    path('system/api-metrics/', main_views.get_api_metrics, name='api-metrics'),
    path('system/privacy-compliance/', main_views.get_privacy_compliance, name='privacy-compliance'),
    
    # Unified Predictor API endpoints (Phase 5)
    path('v2/predict/', unified_predictor.UnifiedPredictorView.as_view(), name='unified-predictor'),
    path('v2/predict/enhanced/', unified_predictor_enhanced.EnhancedUnifiedPredictorView.as_view(), name='enhanced-unified-predictor'),
    path('v2/predict/batch/', batch_predictor.UnifiedBatchPredictorView.as_view(), name='batch-predictor'),
    path('v2/health/', health_monitor.UnifiedHealthCheckView.as_view(), name='unified-health-check'),
] 