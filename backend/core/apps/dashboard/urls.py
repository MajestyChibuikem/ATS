"""
URL configuration for the Dashboard app.

This module defines the URL patterns for the SSAS dashboard.
"""

from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Dashboard home
    path('', views.DashboardHomeView.as_view(), name='home'),
    
    # Student management
    path('students/', views.StudentListView.as_view(), name='student-list'),
    path('students/<int:pk>/', views.StudentDetailView.as_view(), name='student-detail'),
    path('students/create/', views.StudentCreateView.as_view(), name='student-create'),
    path('students/<int:pk>/edit/', views.StudentUpdateView.as_view(), name='student-update'),
    path('students/<int:pk>/delete/', views.StudentDeleteView.as_view(), name='student-delete'),
    
    # Analytics dashboard
    path('analytics/', views.AnalyticsDashboardView.as_view(), name='analytics'),
    path('analytics/performance/', views.PerformanceDashboardView.as_view(), name='performance'),
    path('analytics/trends/', views.TrendsDashboardView.as_view(), name='trends'),
    path('analytics/anomalies/', views.AnomaliesDashboardView.as_view(), name='anomalies'),
    path('analytics/peer-comparison/', views.PeerComparisonDashboardView.as_view(), name='peer-comparison'),
    
    # Reports
    path('reports/', views.ReportListView.as_view(), name='report-list'),
    path('reports/create/', views.ReportCreateView.as_view(), name='report-create'),
    path('reports/<int:pk>/', views.ReportDetailView.as_view(), name='report-detail'),
    path('reports/<int:pk>/download/', views.ReportDownloadView.as_view(), name='report-download'),
    
    # Settings
    path('settings/', views.SettingsView.as_view(), name='settings'),
    path('settings/profile/', views.ProfileSettingsView.as_view(), name='profile-settings'),
    path('settings/system/', views.SystemSettingsView.as_view(), name='system-settings'),
    
    # Admin
    path('admin/', views.AdminDashboardView.as_view(), name='admin'),
    path('admin/users/', views.UserManagementView.as_view(), name='user-management'),
    path('admin/logs/', views.SystemLogsView.as_view(), name='system-logs'),
] 