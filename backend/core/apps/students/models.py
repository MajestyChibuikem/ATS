"""
Student models for SSAS (Smart Student Analytics System).

This module contains all the models related to student data management.
"""

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import uuid


class Student(models.Model):
    """
    Student model for storing basic student information.
    
    This model contains all the essential information about a student
    including personal details, academic information, and contact details.
    """
    
    # Personal Information
    student_id = models.CharField(max_length=20, unique=True, help_text="Unique student identifier")
    first_name = models.CharField(max_length=100, help_text="Student's first name")
    last_name = models.CharField(max_length=100, help_text="Student's last name")
    date_of_birth = models.DateField(help_text="Student's date of birth")
    gender = models.CharField(
        max_length=10,
        choices=[('Male', 'Male'), ('Female', 'Female')],
        help_text="Student's gender"
    )
    
    # Academic Information
    current_class = models.CharField(
        max_length=10,
        choices=[
            ('SS1', 'Senior Secondary 1'),
            ('SS2', 'Senior Secondary 2'),
            ('SS3', 'Senior Secondary 3'),
        ],
        help_text="Current academic class"
    )
    stream = models.CharField(
        max_length=20,
        choices=[
            ('Science', 'Science'),
            ('Arts', 'Arts'),
            ('Commercial', 'Commercial'),
        ],
        help_text="Academic stream"
    )
    admission_date = models.DateField(default=timezone.now, help_text="Date of admission")
    
    # Contact Information
    guardian_name = models.CharField(max_length=200, help_text="Parent/Guardian name")
    guardian_contact = models.CharField(max_length=20, help_text="Guardian's phone number")
    guardian_email = models.EmailField(blank=True, null=True, help_text="Guardian's email address")
    address = models.TextField(help_text="Student's residential address")
    
    # System Information
    created_at = models.DateTimeField(auto_now_add=True, help_text="Record creation timestamp")
    updated_at = models.DateTimeField(auto_now=True, help_text="Record last update timestamp")
    is_active = models.BooleanField(default=True, help_text="Whether the student is currently active")
    
    class Meta:
        db_table = 'students'
        ordering = ['student_id']
        verbose_name = 'Student'
        verbose_name_plural = 'Students'
        indexes = [
            models.Index(fields=['student_id']),
            models.Index(fields=['current_class']),
            models.Index(fields=['stream']),
            models.Index(fields=['is_active']),
        ]
    
    def __str__(self):
        """String representation of the student."""
        return f"{self.student_id} - {self.first_name} {self.last_name}"
    
    @property
    def full_name(self):
        """Get the student's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age(self):
        """Calculate the student's age."""
        today = timezone.now().date()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )


class Subject(models.Model):
    """
    Subject model for storing academic subjects.
    
    This model contains information about different academic subjects
    that students can study.
    """
    
    name = models.CharField(max_length=100, unique=True, help_text="Subject name")
    code = models.CharField(max_length=10, unique=True, help_text="Subject code")
    category = models.CharField(
        max_length=20,
        choices=[
            ('Core', 'Core Subject'),
            ('Elective', 'Elective Subject'),
            ('Optional', 'Optional Subject'),
        ],
        help_text="Subject category"
    )
    stream = models.CharField(
        max_length=20,
        choices=[
            ('All', 'All Streams'),
            ('Science', 'Science'),
            ('Arts', 'Arts'),
            ('Commercial', 'Commercial'),
        ],
        help_text="Applicable stream"
    )
    description = models.TextField(blank=True, help_text="Subject description")
    is_active = models.BooleanField(default=True, help_text="Whether the subject is currently active")
    
    class Meta:
        db_table = 'subjects'
        ordering = ['name']
        verbose_name = 'Subject'
        verbose_name_plural = 'Subjects'
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['code']),
            models.Index(fields=['category']),
            models.Index(fields=['stream']),
        ]
    
    def __str__(self):
        """String representation of the subject."""
        return f"{self.code} - {self.name}"


class AcademicYear(models.Model):
    """
    Academic year model for organizing academic periods.
    
    This model helps organize academic data by year and term.
    """
    
    year = models.CharField(max_length=20, help_text="Academic year (e.g., 2023/2024)")
    term = models.CharField(
        max_length=20,
        choices=[
            ('First Term', 'First Term'),
            ('Second Term', 'Second Term'),
            ('Third Term', 'Third Term'),
        ],
        help_text="Academic term"
    )
    start_date = models.DateField(help_text="Term start date")
    end_date = models.DateField(help_text="Term end date")
    is_current = models.BooleanField(default=False, help_text="Whether this is the current academic period")
    
    class Meta:
        db_table = 'academic_years'
        ordering = ['-year', 'term']
        verbose_name = 'Academic Year'
        verbose_name_plural = 'Academic Years'
        unique_together = ['year', 'term']
        indexes = [
            models.Index(fields=['year']),
            models.Index(fields=['term']),
            models.Index(fields=['is_current']),
        ]
    
    def __str__(self):
        """String representation of the academic year."""
        return f"{self.year} - {self.term}"


class StudentScore(models.Model):
    """
    Student score model for storing academic performance data.
    
    This model stores individual subject scores for students
    across different academic periods.
    """
    
    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name='scores',
        help_text="Student who received this score"
    )
    subject = models.ForeignKey(
        Subject,
        on_delete=models.CASCADE,
        related_name='student_scores',
        help_text="Subject for this score"
    )
    academic_year = models.ForeignKey(
        AcademicYear,
        on_delete=models.CASCADE,
        related_name='student_scores',
        help_text="Academic year for this score"
    )
    
    # Score Information
    continuous_assessment = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Continuous assessment score (0-100)"
    )
    examination_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Examination score (0-100)"
    )
    total_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Total score (0-100)"
    )
    class_average = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
        help_text="Class average score (0-100)",
        default=0
    )
    
    @property
    def term(self):
        """Get the term from the academic year."""
        return self.academic_year.term if self.academic_year else None
    grade = models.CharField(
        max_length=2,
        choices=[
            ('A1', 'A1 (80-100)'),
            ('B2', 'B2 (70-79)'),
            ('B3', 'B3 (65-69)'),
            ('C4', 'C4 (60-64)'),
            ('C5', 'C5 (55-59)'),
            ('C6', 'C6 (50-54)'),
            ('D7', 'D7 (45-49)'),
            ('E8', 'E8 (40-44)'),
            ('F9', 'F9 (0-39)'),
        ],
        help_text="Grade based on total score"
    )
    
    # Additional Information
    remarks = models.TextField(blank=True, help_text="Additional remarks about the score")
    created_at = models.DateTimeField(auto_now_add=True, help_text="Record creation timestamp")
    updated_at = models.DateTimeField(auto_now=True, help_text="Record last update timestamp")
    
    class Meta:
        db_table = 'student_scores'
        ordering = ['student', 'subject', 'academic_year']
        verbose_name = 'Student Score'
        verbose_name_plural = 'Student Scores'
        unique_together = ['student', 'subject', 'academic_year']
        indexes = [
            models.Index(fields=['student', 'subject']),
            models.Index(fields=['academic_year']),
            models.Index(fields=['created_at']),
            models.Index(fields=['grade']),
            # Optimization indexes for peer analysis
            models.Index(fields=['total_score']),
            models.Index(fields=['student', 'total_score']),
            models.Index(fields=['subject', 'total_score']),
        ]
    
    def __str__(self):
        """String representation of the student score."""
        return f"{self.student.student_id} - {self.subject.name} - {self.total_score}"
    
    def save(self, *args, **kwargs):
        """Override save to calculate total score and grade."""
        # Calculate total score (70% exam + 30% CA)
        self.total_score = (self.examination_score * 0.7) + (self.continuous_assessment * 0.3)
        
        # Calculate grade based on total score
        if self.total_score >= 80:
            self.grade = 'A1'
        elif self.total_score >= 70:
            self.grade = 'B2'
        elif self.total_score >= 65:
            self.grade = 'B3'
        elif self.total_score >= 60:
            self.grade = 'C4'
        elif self.total_score >= 55:
            self.grade = 'C5'
        elif self.total_score >= 50:
            self.grade = 'C6'
        elif self.total_score >= 45:
            self.grade = 'D7'
        elif self.total_score >= 40:
            self.grade = 'E8'
        else:
            self.grade = 'F9'
        
        super().save(*args, **kwargs)


class StudentAttendance(models.Model):
    """
    Student attendance model for tracking attendance records.
    
    This model stores daily attendance records for students.
    """
    
    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name='attendance_records',
        help_text="Student for this attendance record"
    )
    date = models.DateField(help_text="Attendance date")
    status = models.CharField(
        max_length=20,
        choices=[
            ('Present', 'Present'),
            ('Absent', 'Absent'),
            ('Late', 'Late'),
            ('Excused', 'Excused'),
        ],
        help_text="Attendance status"
    )
    reason = models.TextField(blank=True, help_text="Reason for absence (if applicable)")
    recorded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="User who recorded this attendance"
    )
    created_at = models.DateTimeField(auto_now_add=True, help_text="Record creation timestamp")
    
    class Meta:
        db_table = 'student_attendance'
        ordering = ['-date', 'student']
        verbose_name = 'Student Attendance'
        verbose_name_plural = 'Student Attendance'
        unique_together = ['student', 'date']
        indexes = [
            models.Index(fields=['student', 'date']),
            models.Index(fields=['date']),
            models.Index(fields=['status']),
        ]
    
    def __str__(self):
        """String representation of the attendance record."""
        return f"{self.student.student_id} - {self.date} - {self.status}"


class StudentBehavior(models.Model):
    """
    Student behavior model for tracking behavioral records.
    
    This model stores behavioral observations and incidents.
    """
    
    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name='behavior_records',
        help_text="Student for this behavior record"
    )
    date = models.DateField(help_text="Behavior record date")
    category = models.CharField(
        max_length=50,
        choices=[
            ('Academic', 'Academic'),
            ('Social', 'Social'),
            ('Disciplinary', 'Disciplinary'),
            ('Health', 'Health'),
            ('Other', 'Other'),
        ],
        help_text="Behavior category"
    )
    description = models.TextField(help_text="Description of the behavior")
    severity = models.CharField(
        max_length=20,
        choices=[
            ('Low', 'Low'),
            ('Medium', 'Medium'),
            ('High', 'High'),
            ('Critical', 'Critical'),
        ],
        help_text="Severity level of the behavior"
    )
    action_taken = models.TextField(blank=True, help_text="Action taken in response")
    recorded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="User who recorded this behavior"
    )
    created_at = models.DateTimeField(auto_now_add=True, help_text="Record creation timestamp")
    
    class Meta:
        db_table = 'student_behavior'
        ordering = ['-date', 'student']
        verbose_name = 'Student Behavior'
        verbose_name_plural = 'Student Behavior'
        indexes = [
            models.Index(fields=['student', 'date']),
            models.Index(fields=['category']),
            models.Index(fields=['severity']),
        ]
    
    def __str__(self):
        """String representation of the behavior record."""
        return f"{self.student.student_id} - {self.date} - {self.category}"
