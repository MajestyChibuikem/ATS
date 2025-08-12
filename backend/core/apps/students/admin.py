"""
Admin interface for student models.

This module provides the Django admin interface for managing student data.
"""

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import (
    Student, Subject, Teacher, TeacherPerformance, AcademicYear, 
    StudentScore, StudentAttendance, StudentBehavior
)


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    """Admin interface for Student model."""
    
    list_display = [
        'student_id', 'full_name', 'current_class', 'stream', 
        'gender', 'age', 'is_active', 'created_at'
    ]
    list_filter = [
        'current_class', 'stream', 'gender', 'is_active', 
        'admission_date', 'created_at'
    ]
    search_fields = [
        'student_id', 'first_name', 'last_name', 'guardian_name',
        'guardian_contact', 'guardian_email'
    ]
    readonly_fields = ['created_at', 'updated_at', 'age']
    fieldsets = (
        ('Personal Information', {
            'fields': ('student_id', 'first_name', 'last_name', 'date_of_birth', 'gender')
        }),
        ('Academic Information', {
            'fields': ('current_class', 'stream', 'admission_date')
        }),
        ('Contact Information', {
            'fields': ('guardian_name', 'guardian_contact', 'guardian_email', 'address')
        }),
        ('System Information', {
            'fields': ('is_active', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    ordering = ['student_id']
    list_per_page = 25
    
    def full_name(self, obj):
        """Display full name."""
        return obj.full_name
    full_name.short_description = 'Full Name'
    
    def age(self, obj):
        """Display age."""
        return obj.age
    age.short_description = 'Age'


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    """Admin interface for Subject model."""
    
    list_display = ['code', 'name', 'category', 'stream', 'is_active']
    list_filter = ['category', 'stream', 'is_active']
    search_fields = ['name', 'code', 'description']
    readonly_fields = []
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'code', 'description')
        }),
        ('Classification', {
            'fields': ('category', 'stream', 'is_active')
        }),
    )
    ordering = ['name']
    list_per_page = 25


@admin.register(Teacher)
class TeacherAdmin(admin.ModelAdmin):
    """Admin interface for Teacher model."""
    
    list_display = [
        'teacher_id', 'name', 'specialization', 'qualification_level',
        'years_experience', 'performance_rating', 'is_active'
    ]
    list_filter = [
        'specialization', 'qualification_level', 'is_active',
        'years_experience', 'performance_rating'
    ]
    search_fields = [
        'teacher_id', 'name', 'specialization'
    ]
    readonly_fields = ['created_at', 'updated_at', 'experience_level']
    fieldsets = (
        ('Basic Information', {
            'fields': ('teacher_id', 'name')
        }),
        ('Professional Information', {
            'fields': ('years_experience', 'qualification_level', 'specialization')
        }),
        ('Performance & Workload', {
            'fields': ('teaching_load', 'performance_rating', 'years_at_school')
        }),
        ('System Information', {
            'fields': ('is_active', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    ordering = ['teacher_id']
    list_per_page = 25
    
    def experience_level(self, obj):
        """Display experience level."""
        return obj.experience_level
    experience_level.short_description = 'Experience Level'


@admin.register(TeacherPerformance)
class TeacherPerformanceAdmin(admin.ModelAdmin):
    """Admin interface for TeacherPerformance model."""
    
    list_display = [
        'teacher', 'subject', 'academic_year', 'average_class_score',
        'pass_rate', 'student_satisfaction_rating', 'number_of_students'
    ]
    list_filter = [
        'subject', 'academic_year', 'teacher__specialization'
    ]
    search_fields = [
        'teacher__teacher_id', 'teacher__name', 'subject__name', 'subject__code'
    ]
    readonly_fields = ['created_at', 'updated_at']
    fieldsets = (
        ('Teacher & Subject', {
            'fields': ('teacher', 'subject', 'academic_year')
        }),
        ('Performance Metrics', {
            'fields': ('average_class_score', 'number_of_students', 'pass_rate')
        }),
        ('Quality Indicators', {
            'fields': ('student_satisfaction_rating', 'professional_development_hours', 'class_attendance_rate')
        }),
        ('System Information', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    ordering = ['teacher', 'subject', 'academic_year']
    list_per_page = 25
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        return super().get_queryset(request).select_related('teacher', 'subject')


@admin.register(AcademicYear)
class AcademicYearAdmin(admin.ModelAdmin):
    """Admin interface for AcademicYear model."""
    
    list_display = ['year', 'term', 'start_date', 'end_date', 'is_current']
    list_filter = ['year', 'term', 'is_current']
    search_fields = ['year', 'term']
    readonly_fields = []
    fieldsets = (
        ('Academic Period', {
            'fields': ('year', 'term')
        }),
        ('Dates', {
            'fields': ('start_date', 'end_date')
        }),
        ('Status', {
            'fields': ('is_current',)
        }),
    )
    ordering = ['-year', 'term']
    list_per_page = 25


@admin.register(StudentScore)
class StudentScoreAdmin(admin.ModelAdmin):
    """Admin interface for StudentScore model."""
    
    list_display = [
        'student', 'subject', 'teacher', 'academic_year', 'continuous_assessment',
        'examination_score', 'total_score', 'grade', 'created_at'
    ]
    list_filter = [
        'subject', 'academic_year', 'grade', 'teacher__specialization', 'created_at'
    ]
    search_fields = [
        'student__student_id', 'student__first_name', 'student__last_name',
        'subject__name', 'subject__code', 'teacher__name'
    ]
    readonly_fields = ['total_score', 'grade', 'created_at', 'updated_at']
    fieldsets = (
        ('Student Information', {
            'fields': ('student', 'subject', 'academic_year', 'teacher')
        }),
        ('Scores', {
            'fields': ('continuous_assessment', 'examination_score', 'total_score', 'grade')
        }),
        ('Additional Information', {
            'fields': ('remarks', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    ordering = ['student', 'subject', 'academic_year']
    list_per_page = 25
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        return super().get_queryset(request).select_related('student', 'subject', 'academic_year', 'teacher')


@admin.register(StudentAttendance)
class StudentAttendanceAdmin(admin.ModelAdmin):
    """Admin interface for StudentAttendance model."""
    
    list_display = [
        'student', 'teacher', 'date', 'status', 'reason_short', 'recorded_by', 'created_at'
    ]
    list_filter = [
        'status', 'date', 'teacher__specialization', 'created_at'
    ]
    search_fields = [
        'student__student_id', 'student__first_name', 'student__last_name',
        'teacher__name', 'reason'
    ]
    readonly_fields = ['created_at']
    fieldsets = (
        ('Attendance Information', {
            'fields': ('student', 'teacher', 'date', 'status')
        }),
        ('Details', {
            'fields': ('reason', 'recorded_by')
        }),
        ('System Information', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    ordering = ['-date', 'student']
    list_per_page = 25
    
    def reason_short(self, obj):
        """Display shortened reason."""
        return obj.reason[:50] + '...' if len(obj.reason) > 50 else obj.reason
    reason_short.short_description = 'Reason'
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        return super().get_queryset(request).select_related('student', 'teacher', 'recorded_by')


@admin.register(StudentBehavior)
class StudentBehaviorAdmin(admin.ModelAdmin):
    """Admin interface for StudentBehavior model."""
    
    list_display = [
        'student', 'teacher', 'date', 'category', 'severity', 'description_short',
        'recorded_by', 'created_at'
    ]
    list_filter = [
        'category', 'severity', 'date', 'teacher__specialization', 'created_at'
    ]
    search_fields = [
        'student__student_id', 'student__first_name', 'student__last_name',
        'teacher__name', 'description', 'action_taken'
    ]
    readonly_fields = ['created_at']
    fieldsets = (
        ('Behavior Information', {
            'fields': ('student', 'teacher', 'date', 'category', 'severity')
        }),
        ('Details', {
            'fields': ('description', 'action_taken', 'recorded_by')
        }),
        ('System Information', {
            'fields': ('created_at',),
            'classes': ('collapse',)
        }),
    )
    ordering = ['-date', 'student']
    list_per_page = 25
    
    def description_short(self, obj):
        """Display shortened description."""
        return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description
    description_short.short_description = 'Description'
    
    def get_queryset(self, request):
        """Optimize queryset with select_related."""
        return super().get_queryset(request).select_related('student', 'teacher', 'recorded_by')


# Customize admin site
admin.site.site_header = "SSAS Administration"
admin.site.site_title = "SSAS Admin"
admin.site.index_title = "Welcome to SSAS Administration"
