"""
Django management command for exporting student data to Excel files.

This command exports student data from the database to Excel format.
"""

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
import logging
import os
from pathlib import Path

from core.apps.students.models import (
    Student, Subject, Teacher, TeacherPerformance, AcademicYear, 
    StudentScore, StudentAttendance, StudentBehavior
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Management command for exporting student data."""
    
    help = 'Export student data to Excel files'
    
    def add_arguments(self, parser):
        """Add command arguments."""
        parser.add_argument(
            'file_path',
            type=str,
            help='Path to the Excel file for export'
        )
        parser.add_argument(
            '--include-scores',
            action='store_true',
            help='Include student scores in export'
        )
        parser.add_argument(
            '--include-attendance',
            action='store_true',
            help='Include attendance records in export'
        )
        parser.add_argument(
            '--include-behavior',
            action='store_true',
            help='Include behavioral records in export'
        )
        parser.add_argument(
            '--include-teachers',
            action='store_true',
            help='Include teacher data in export'
        )
        parser.add_argument(
            '--academic-year',
            type=str,
            help='Filter by specific academic year'
        )
        parser.add_argument(
            '--class-level',
            type=str,
            choices=['SS1', 'SS2', 'SS3'],
            help='Filter by class level'
        )
    
    def handle(self, *args, **options):
        """Handle the command execution."""
        file_path = options['file_path']
        include_scores = options['include_scores']
        include_attendance = options['include_attendance']
        include_behavior = options['include_behavior']
        include_teachers = options['include_teachers']
        academic_year = options['academic_year']
        class_level = options['class_level']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting export to: {file_path}')
        )
        
        try:
            # Export data
            exported_count = self.export_data(
                file_path, include_scores, include_attendance, 
                include_behavior, include_teachers, academic_year, class_level
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully exported {exported_count} records to {file_path}'
                )
            )
            
        except Exception as e:
            logger.error(f'Export failed: {str(e)}')
            raise CommandError(f'Export failed: {str(e)}')
    
    def export_data(self, file_path, include_scores, include_attendance, 
                   include_behavior, include_teachers, academic_year, class_level):
        """Export data to Excel file."""
        exported_count = 0
        
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                
                # Export students
                self.stdout.write('Exporting students...')
                students_df = self.export_students(academic_year, class_level)
                students_df.to_excel(writer, sheet_name='Students', index=False)
                exported_count += len(students_df)
                self.stdout.write(f'Exported {len(students_df)} students')
                
                # Export subjects
                self.stdout.write('Exporting subjects...')
                subjects_df = self.export_subjects()
                subjects_df.to_excel(writer, sheet_name='Subjects', index=False)
                exported_count += len(subjects_df)
                self.stdout.write(f'Exported {len(subjects_df)} subjects')
                
                # Export teachers if requested
                if include_teachers:
                    self.stdout.write('Exporting teachers...')
                    teachers_df = self.export_teachers()
                    teachers_df.to_excel(writer, sheet_name='Teachers', index=False)
                    exported_count += len(teachers_df)
                    self.stdout.write(f'Exported {len(teachers_df)} teachers')
                    
                    # Export teacher performance
                    self.stdout.write('Exporting teacher performance...')
                    teacher_performance_df = self.export_teacher_performance(academic_year)
                    teacher_performance_df.to_excel(writer, sheet_name='TeacherPerformance', index=False)
                    exported_count += len(teacher_performance_df)
                    self.stdout.write(f'Exported {len(teacher_performance_df)} teacher performance records')
                
                # Export scores if requested
                if include_scores:
                    self.stdout.write('Exporting subject scores...')
                    scores_df = self.export_scores(academic_year, class_level)
                    scores_df.to_excel(writer, sheet_name='SubjectScores', index=False)
                    exported_count += len(scores_df)
                    self.stdout.write(f'Exported {len(scores_df)} subject scores')
                
                # Export attendance if requested
                if include_attendance:
                    self.stdout.write('Exporting attendance records...')
                    attendance_df = self.export_attendance(academic_year, class_level)
                    attendance_df.to_excel(writer, sheet_name='Attendance', index=False)
                    exported_count += len(attendance_df)
                    self.stdout.write(f'Exported {len(attendance_df)} attendance records')
                
                # Export behavior if requested
                if include_behavior:
                    self.stdout.write('Exporting behavioral records...')
                    behavior_df = self.export_behavior(academic_year, class_level)
                    behavior_df.to_excel(writer, sheet_name='BehavioralRecords', index=False)
                    exported_count += len(behavior_df)
                    self.stdout.write(f'Exported {len(behavior_df)} behavioral records')
            
            return exported_count
            
        except Exception as e:
            logger.error(f'Error during export: {str(e)}')
            raise CommandError(f'Export error: {str(e)}')
    
    def export_students(self, academic_year=None, class_level=None):
        """Export student data."""
        queryset = Student.objects.filter(is_active=True)
        
        if class_level:
            queryset = queryset.filter(current_class=class_level)
        
        data = []
        for student in queryset:
            data.append({
                'StudentID': student.student_id,
                'FirstName': student.first_name,
                'LastName': student.last_name,
                'DateOfBirth': student.date_of_birth,
                'Gender': student.gender,
                'CurrentClass': student.current_class,
                'Stream': student.stream,
                'EnrollmentDate': student.admission_date,
                'GuardianName': student.guardian_name,
                'GuardianContact': student.guardian_contact,
                'GuardianEmail': student.guardian_email,
                'Address': student.address,
                'CreatedAt': student.created_at,
                'UpdatedAt': student.updated_at
            })
        
        return pd.DataFrame(data)
    
    def export_subjects(self):
        """Export subject data."""
        queryset = Subject.objects.filter(is_active=True)
        
        data = []
        for subject in queryset:
            data.append({
                'Name': subject.name,
                'Code': subject.code,
                'Category': subject.category,
                'Stream': subject.stream,
                'Description': subject.description,
                'IsActive': subject.is_active,
                'CreatedAt': subject.created_at,
                'UpdatedAt': subject.updated_at
            })
        
        return pd.DataFrame(data)
    
    def export_teachers(self):
        """Export teacher data."""
        queryset = Teacher.objects.filter(is_active=True)
        
        data = []
        for teacher in queryset:
            data.append({
                'TeacherID': teacher.teacher_id,
                'Name': teacher.name,
                'YearsExperience': teacher.years_experience,
                'QualificationLevel': teacher.qualification_level,
                'Specialization': teacher.specialization,
                'TeachingLoad': teacher.teaching_load,
                'PerformanceRating': teacher.performance_rating,
                'YearsAtSchool': teacher.years_at_school,
                'ExperienceLevel': teacher.experience_level,
                'CreatedAt': teacher.created_at,
                'UpdatedAt': teacher.updated_at
            })
        
        return pd.DataFrame(data)
    
    def export_teacher_performance(self, academic_year=None):
        """Export teacher performance data."""
        queryset = TeacherPerformance.objects.all()
        
        if academic_year:
            queryset = queryset.filter(academic_year=academic_year)
        
        data = []
        for performance in queryset:
            data.append({
                'TeacherID': performance.teacher.teacher_id,
                'TeacherName': performance.teacher.name,
                'SubjectName': performance.subject.name,
                'AcademicYear': performance.academic_year,
                'AverageClassScore': performance.average_class_score,
                'NumberOfStudents': performance.number_of_students,
                'PassRate': performance.pass_rate,
                'StudentSatisfactionRating': performance.student_satisfaction_rating,
                'ProfessionalDevelopmentHours': performance.professional_development_hours,
                'ClassAttendanceRate': performance.class_attendance_rate,
                'CreatedAt': performance.created_at,
                'UpdatedAt': performance.updated_at
            })
        
        return pd.DataFrame(data)
    
    def export_scores(self, academic_year=None, class_level=None):
        """Export subject scores with teacher information."""
        queryset = StudentScore.objects.select_related(
            'student', 'subject', 'academic_year', 'teacher'
        )
        
        if academic_year:
            queryset = queryset.filter(academic_year__year=academic_year)
        
        if class_level:
            queryset = queryset.filter(student__current_class=class_level)
        
        data = []
        for score in queryset:
            data.append({
                'StudentID': score.student.student_id,
                'SubjectName': score.subject.name,
                'TeacherID': score.teacher.teacher_id if score.teacher else None,
                'TeacherName': score.teacher.name if score.teacher else None,
                'AssessmentType': 'Examination',  # Default type
                'Score': score.total_score,
                'AcademicYear': score.academic_year.year,
                'Term': score.academic_year.term,
                'Class': score.student.current_class,
                'Stream': score.student.stream,
                'AssessmentDate': score.created_at,
                'TeacherRemarks': score.remarks,
                'ContinuousAssessment': score.continuous_assessment,
                'ExaminationScore': score.examination_score,
                'ClassAverage': score.class_average,
                'Grade': score.grade
            })
        
        return pd.DataFrame(data)
    
    def export_attendance(self, academic_year=None, class_level=None):
        """Export attendance records with teacher information."""
        queryset = StudentAttendance.objects.select_related('student', 'teacher')
        
        if class_level:
            queryset = queryset.filter(student__current_class=class_level)
        
        data = []
        for attendance in queryset:
            data.append({
                'StudentID': attendance.student.student_id,
                'TeacherID': attendance.teacher.teacher_id if attendance.teacher else None,
                'TeacherName': attendance.teacher.name if attendance.teacher else None,
                'Date': attendance.date,
                'Status': attendance.status,
                'Remarks': attendance.reason,
                'RecordedBy': attendance.recorded_by.username if attendance.recorded_by else None,
                'CreatedAt': attendance.created_at
            })
        
        return pd.DataFrame(data)
    
    def export_behavior(self, academic_year=None, class_level=None):
        """Export behavioral records with teacher information."""
        queryset = StudentBehavior.objects.select_related('student', 'teacher')
        
        if class_level:
            queryset = queryset.filter(student__current_class=class_level)
        
        data = []
        for behavior in queryset:
            data.append({
                'StudentID': behavior.student.student_id,
                'TeacherID': behavior.teacher.teacher_id if behavior.teacher else None,
                'TeacherName': behavior.teacher.name if behavior.teacher else None,
                'RecordDate': behavior.date,
                'Category': behavior.category,
                'Description': behavior.description,
                'Severity': behavior.severity,
                'ActionTaken': behavior.action_taken,
                'RecordedBy': behavior.recorded_by.username if behavior.recorded_by else None,
                'CreatedAt': behavior.created_at
            })
        
        return pd.DataFrame(data) 