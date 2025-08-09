"""
URL configuration for the API app.

This module defines the URL patterns for the SSAS API endpoints.
"""

from django.urls import path, include
from knox import views as knox_views

from . import views

app_name = 'api'

urlpatterns = [
    # API Root
    path('', views.api_root, name='api-root'),
    
    # Authentication endpoints
    path('auth/', include('knox.urls')),
    path('auth/logout/', knox_views.LogoutView.as_view(), name='knox_logout'),
    path('auth/logoutall/', knox_views.LogoutAllView.as_view(), name='knox_logoutall'),
    
    # ML Module endpoints
    path('students/<str:student_id>/career-recommendations/', views.get_career_recommendations, name='career-recommendations'),
    path('students/<str:student_id>/peer-analysis/', views.get_peer_analysis, name='peer-analysis'),
    path('students/<str:student_id>/anomalies/', views.get_anomaly_detection, name='anomaly-detection'),
    path('students/<str:student_id>/performance-prediction/', views.get_performance_prediction, name='performance-prediction'),
    path('students/<str:student_id>/comprehensive-analysis/', views.get_comprehensive_analysis, name='comprehensive-analysis'),
    
    # Data validation endpoints
    path('students/validate/', views.validate_student_data, name='validate-student-data'),
    
    # Batch processing endpoints
    path('batch/analysis/', views.batch_analysis, name='batch-analysis'),
    path('tasks/<str:task_id>/status/', views.get_task_status, name='task-status'),
    
    # System endpoints
    path('system/health/', views.get_ml_health, name='ml-health'),
    path('system/metrics/', views.get_system_metrics, name='system-metrics'),
    path('system/api-metrics/', views.get_api_metrics, name='api-metrics'),
] 