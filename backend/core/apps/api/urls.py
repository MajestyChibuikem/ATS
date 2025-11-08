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
    # Authentication endpoints
    path('auth/login/', main_views.login_view, name='login'),
    path('auth/logout/', knox_views.LogoutView.as_view(), name='logout'),
    path('auth/', include('knox.urls')),
    
    # Health check endpoints
    path('test/', main_views.simple_health_check, name='test'),
    path('system/health/', main_views.get_ml_health, name='ml-health'),
    path('system/metrics/', main_views.get_system_metrics, name='system-metrics'),
    
    # Data endpoints
    path('students/', main_views.get_students_list, name='students-list'),
    path('students/paginated/', main_views.get_students_list_paginated, name='students-list-paginated'),
    path('teachers/', main_views.get_teachers_list, name='teachers-list'),
    path('teachers/paginated/', main_views.get_teachers_list_paginated, name='teachers-list-paginated'),
    
    # ML Analysis endpoints
    path('students/<str:student_id>/career-recommendations/', main_views.get_career_recommendations, name='career-recommendations'),
    path('students/<str:student_id>/peer-analysis/', main_views.get_peer_analysis, name='peer-analysis'),
    path('students/<str:student_id>/anomaly-detection/', main_views.get_anomaly_detection, name='anomaly-detection'),
    path('students/<str:student_id>/performance-prediction/', main_views.get_performance_prediction, name='performance-prediction'),
    path('students/<str:student_id>/comprehensive-analysis/', main_views.get_comprehensive_analysis, name='comprehensive-analysis'),
    
    # Batch processing endpoints
    path('batch/analysis/', main_views.batch_analysis, name='batch-analysis'),
    path('tasks/<str:task_id>/status/', main_views.get_task_status, name='task-status'),
    
    # Data validation and privacy
    path('validate/student-data/', main_views.validate_student_data, name='validate-student-data'),
    path('privacy/compliance/', main_views.get_privacy_compliance, name='privacy-compliance'),
    
    # API root and metrics
    path('', main_views.api_root, name='api-root'),
    path('metrics/', main_views.get_api_metrics, name='api-metrics'),
] 