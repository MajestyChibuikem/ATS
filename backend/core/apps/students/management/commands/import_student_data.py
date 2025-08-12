"""
Django management command for importing student data from Excel files.

This command processes Excel files containing student data and imports them into the database.
"""

import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone
from datetime import datetime
import logging
import os
from pathlib import Path

from core.apps.students.models import (
    Student, Subject, Teacher, TeacherPerformance, AcademicYear, 
    StudentScore, StudentAttendance, StudentBehavior
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Management command for importing student data."""
    
    help = 'Import student data from Excel files'
    
    def add_arguments(self, parser):
        """Add command arguments."""
        parser.add_argument(
            'file_path',
            type=str,
            help='Path to the Excel file containing student data'
        )
        parser.add_argument(
            '--academic-year',
            type=str,
            default='2023/2024',
            help='Academic year for the data (default: 2023/2024)'
        )
        parser.add_argument(
            '--term',
            type=str,
            default='First Term',
            choices=['First Term', 'Second Term', 'Third Term'],
            help='Academic term (default: First Term)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without actually importing data (for testing)'
        )
        parser.add_argument(
            '--clear-existing',
            action='store_true',
            help='Clear existing data before importing'
        )
        parser.add_argument(
            '--import-teachers',
            action='store_true',
            help='Import teacher data from Teachers and TeacherPerformance sheets'
        )
    
    def handle(self, *args, **options):
        """Handle the command execution."""
        file_path = options['file_path']
        academic_year = options['academic_year']
        term = options['term']
        dry_run = options['dry_run']
        clear_existing = options['clear_existing']
        import_teachers = options['import_teachers']
        
        # Validate file path
        if not os.path.exists(file_path):
            raise CommandError(f'File not found: {file_path}')
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting import from: {file_path}')
        )
        
        try:
            # Process the import
            if dry_run:
                self.stdout.write(
                    self.style.WARNING('DRY RUN MODE - No data will be imported')
                )
            
            # Import data
            imported_count = self.import_data(
                file_path, academic_year, term, dry_run, clear_existing, import_teachers
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully imported {imported_count} records'
                )
            )
            
        except Exception as e:
            logger.error(f'Import failed: {str(e)}')
            raise CommandError(f'Import failed: {str(e)}')
    
    def import_data(self, file_path, academic_year, term, dry_run, clear_existing, import_teachers):
        """Import data from Excel file."""
        imported_count = 0
        
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            self.stdout.write(f'Available sheets: {excel_file.sheet_names}')
            
            # Import teachers first if requested
            if import_teachers and 'Teachers' in excel_file.sheet_names:
                self.stdout.write('Importing teachers...')
                df_teachers = pd.read_excel(file_path, sheet_name='Teachers')
                teacher_count = self.import_teachers(df_teachers, dry_run, clear_existing)
                imported_count += teacher_count
                self.stdout.write(f'Imported {teacher_count} teachers')
            
            # Import teacher performance if available
            if import_teachers and 'TeacherPerformance' in excel_file.sheet_names:
                self.stdout.write('Importing teacher performance...')
                df_teacher_performance = pd.read_excel(file_path, sheet_name='TeacherPerformance')
                performance_count = self.import_teacher_performance(df_teacher_performance, dry_run, clear_existing)
                imported_count += performance_count
                self.stdout.write(f'Imported {performance_count} teacher performance records')
            
            # Import subjects
            if 'Subjects' in excel_file.sheet_names:
                self.stdout.write('Importing subjects...')
                df_subjects = pd.read_excel(file_path, sheet_name='Subjects')
                subject_count = self.import_subjects(df_subjects, dry_run, clear_existing)
                imported_count += subject_count
                self.stdout.write(f'Imported {subject_count} subjects')
            
            # Import students
            if 'Students' in excel_file.sheet_names:
                self.stdout.write('Importing students...')
                df_students = pd.read_excel(file_path, sheet_name='Students')
                student_count = self.import_students(df_students, dry_run, clear_existing)
                imported_count += student_count
                self.stdout.write(f'Imported {student_count} students')
            
            # Import subject scores (with teacher relationships)
            if 'SubjectScores' in excel_file.sheet_names:
                self.stdout.write('Importing subject scores...')
                df_scores = pd.read_excel(file_path, sheet_name='SubjectScores')
                score_count = self.import_scores(df_scores, academic_year, dry_run, clear_existing)
                imported_count += score_count
                self.stdout.write(f'Imported {score_count} subject scores')
            
            # Import attendance
            if 'Attendance' in excel_file.sheet_names:
                self.stdout.write('Importing attendance...')
                df_attendance = pd.read_excel(file_path, sheet_name='Attendance')
                attendance_count = self.import_attendance(df_attendance, dry_run, clear_existing)
                imported_count += attendance_count
                self.stdout.write(f'Imported {attendance_count} attendance records')
            
            # Import behavioral records
            if 'BehavioralRecords' in excel_file.sheet_names:
                self.stdout.write('Importing behavioral records...')
                df_behavior = pd.read_excel(file_path, sheet_name='BehavioralRecords')
                behavior_count = self.import_behavior(df_behavior, dry_run, clear_existing)
                imported_count += behavior_count
                self.stdout.write(f'Imported {behavior_count} behavioral records')
            
            return imported_count
            
        except Exception as e:
            logger.error(f'Error during import: {str(e)}')
            raise CommandError(f'Import error: {str(e)}')
    
    def import_teachers(self, df, dry_run, clear_existing):
        """Import teacher data."""
        if clear_existing and not dry_run:
            Teacher.objects.all().delete()
            self.stdout.write('Cleared existing teachers')
        
        count = 0
        for _, row in df.iterrows():
            try:
                if dry_run:
                    count += 1
                    continue
                
                teacher, created = Teacher.objects.get_or_create(
                    teacher_id=row['TeacherID'],
                    defaults={
                        'name': row['Name'],
                        'years_experience': row['YearsExperience'],
                        'qualification_level': row['QualificationLevel'],
                        'specialization': row['Specialization'],
                        'teaching_load': row['TeachingLoad'],
                        'performance_rating': row['PerformanceRating'],
                        'years_at_school': row['YearsAtSchool'],
                        'is_active': True
                    }
                )
                
                if created:
                    count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing teacher {row.get("TeacherID", "Unknown")}: {str(e)}')
                )
        
        return count
    
    def import_teacher_performance(self, df, dry_run, clear_existing):
        """Import teacher performance data."""
        if clear_existing and not dry_run:
            TeacherPerformance.objects.all().delete()
            self.stdout.write('Cleared existing teacher performance records')
        
        count = 0
        for _, row in df.iterrows():
            try:
                if dry_run:
                    count += 1
                    continue
                
                # Get teacher and subject
                teacher = Teacher.objects.get(teacher_id=row['TeacherID'])
                subject = Subject.objects.get(name=row['SubjectName'])
                
                performance, created = TeacherPerformance.objects.get_or_create(
                    teacher=teacher,
                    subject=subject,
                    academic_year=row['AcademicYear'],
                    defaults={
                        'average_class_score': row['AverageClassScore'],
                        'number_of_students': row['NumberOfStudents'],
                        'pass_rate': row['PassRate'],
                        'student_satisfaction_rating': row['StudentSatisfactionRating'],
                        'professional_development_hours': row['ProfessionalDevelopmentHours'],
                        'class_attendance_rate': row['ClassAttendanceRate']
                    }
                )
                
                if created:
                    count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing teacher performance for {row.get("TeacherID", "Unknown")}: {str(e)}')
                )
        
        return count

    def import_students(self, df, dry_run, clear_existing):
        """Import student data."""
        if clear_existing and not dry_run:
            Student.objects.all().delete()
            self.stdout.write('Cleared existing students')
        
        count = 0
        for _, row in df.iterrows():
            try:
                if dry_run:
                    count += 1
                    continue
                
                # Convert date strings to datetime objects
                date_of_birth = pd.to_datetime(row['DateOfBirth']).date()
                admission_date = pd.to_datetime(row['EnrollmentDate']).date()
                
                student, created = Student.objects.get_or_create(
                    student_id=row['StudentID'],
                    defaults={
                        'first_name': row['FirstName'],
                        'last_name': row['LastName'],
                        'date_of_birth': date_of_birth,
                        'gender': row['Gender'],
                        'current_class': row['CurrentClass'],
                        'stream': row['Stream'],
                        'admission_date': admission_date,
                        'guardian_name': row.get('GuardianName', ''),
                        'guardian_contact': row.get('GuardianContact', ''),
                        'guardian_email': row.get('GuardianEmail', ''),
                        'address': row.get('Address', ''),
                        'is_active': True
                    }
                )
                
                if created:
                    count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing student {row.get("StudentID", "Unknown")}: {str(e)}')
                )
        
        return count

    def import_subjects(self, df, dry_run, clear_existing):
        """Import subject data."""
        if clear_existing and not dry_run:
            Subject.objects.all().delete()
            self.stdout.write('Cleared existing subjects')
        
        count = 0
        for _, row in df.iterrows():
            try:
                if dry_run:
                    count += 1
                    continue
                
                subject, created = Subject.objects.get_or_create(
                    name=row['Name'],
                    defaults={
                        'code': row['Code'],
                        'category': row['Category'],
                        'stream': row['Stream'],
                        'description': row.get('Description', ''),
                        'is_active': True
                    }
                )
                
                if created:
                    count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing subject {row.get("Name", "Unknown")}: {str(e)}')
                )
        
        return count

    def import_scores(self, df, academic_year, dry_run, clear_existing):
        """Import subject scores with teacher relationships."""
        if clear_existing and not dry_run:
            StudentScore.objects.all().delete()
            self.stdout.write('Cleared existing scores')
        
        count = 0
        for _, row in df.iterrows():
            try:
                if dry_run:
                    count += 1
                    continue
                
                # Get related objects
                student = Student.objects.get(student_id=row['StudentID'])
                subject = Subject.objects.get(name=row['SubjectName'])
                
                # Get or create academic year
                academic_year_obj, _ = AcademicYear.objects.get_or_create(
                    year=academic_year,
                    term=row['Term'],
                    defaults={
                        'start_date': datetime.now().date(),
                        'end_date': datetime.now().date(),
                        'is_current': False
                    }
                )
                
                # Get teacher if available
                teacher = None
                if 'TeacherID' in row and pd.notna(row['TeacherID']):
                    try:
                        teacher = Teacher.objects.get(teacher_id=row['TeacherID'])
                    except Teacher.DoesNotExist:
                        self.stdout.write(
                            self.style.WARNING(f'Teacher {row["TeacherID"]} not found for score')
                        )
                
                # Calculate scores based on assessment type
                if row['AssessmentType'] == 'Quiz':
                    continuous_assessment = row['Score']
                    examination_score = 0
                elif row['AssessmentType'] == 'Assignment':
                    continuous_assessment = row['Score']
                    examination_score = 0
                elif row['AssessmentType'] == 'Project':
                    continuous_assessment = row['Score']
                    examination_score = 0
                elif row['AssessmentType'] == 'Examination':
                    continuous_assessment = 0
                    examination_score = row['Score']
                else:
                    # Default split
                    continuous_assessment = row['Score'] * 0.3
                    examination_score = row['Score'] * 0.7
                
                score, created = StudentScore.objects.get_or_create(
                    student=student,
                    subject=subject,
                    academic_year=academic_year_obj,
                    defaults={
                        'continuous_assessment': continuous_assessment,
                        'examination_score': examination_score,
                        'teacher': teacher,
                        'remarks': row.get('TeacherRemarks', ''),
                        'class_average': row.get('ClassAverage', 0)
                    }
                )
                
                if created:
                    count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing score for {row.get("StudentID", "Unknown")}: {str(e)}')
                )
        
        return count

    def import_attendance(self, df, dry_run, clear_existing):
        """Import attendance data with teacher relationships."""
        if clear_existing and not dry_run:
            StudentAttendance.objects.all().delete()
            self.stdout.write('Cleared existing attendance records')
        
        count = 0
        for _, row in df.iterrows():
            try:
                if dry_run:
                    count += 1
                    continue
                
                # Get student
                student = Student.objects.get(student_id=row['StudentID'])
                
                # Convert date
                date = pd.to_datetime(row['Date']).date()
                
                # Get teacher if available
                teacher = None
                if 'TeacherID' in row and pd.notna(row['TeacherID']):
                    try:
                        teacher = Teacher.objects.get(teacher_id=row['TeacherID'])
                    except Teacher.DoesNotExist:
                        self.stdout.write(
                            self.style.WARNING(f'Teacher {row["TeacherID"]} not found for attendance')
                        )
                
                attendance, created = StudentAttendance.objects.get_or_create(
                    student=student,
                    date=date,
                    defaults={
                        'status': row['Status'],
                        'reason': row.get('Remarks', ''),
                        'teacher': teacher
                    }
                )
                
                if created:
                    count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing attendance for {row.get("StudentID", "Unknown")}: {str(e)}')
                )
        
        return count

    def import_behavior(self, df, dry_run, clear_existing):
        """Import behavioral records with teacher relationships."""
        if clear_existing and not dry_run:
            StudentBehavior.objects.all().delete()
            self.stdout.write('Cleared existing behavioral records')
        
        count = 0
        for _, row in df.iterrows():
            try:
                if dry_run:
                    count += 1
                    continue
                
                # Get student
                student = Student.objects.get(student_id=row['StudentID'])
                
                # Convert date
                date = pd.to_datetime(row['RecordDate']).date()
                
                # Get teacher if available
                teacher = None
                if 'RecordedBy' in row and pd.notna(row['RecordedBy']):
                    try:
                        # Try to find teacher by name
                        teacher = Teacher.objects.filter(name__icontains=row['RecordedBy']).first()
                    except Exception:
                        pass
                
                behavior, created = StudentBehavior.objects.get_or_create(
                    student=student,
                    date=date,
                    category=row['Category'],
                    description=row['Description'],
                    defaults={
                        'severity': 'Medium',  # Default severity
                        'action_taken': row.get('ActionTaken', ''),
                        'teacher': teacher
                    }
                )
                
                if created:
                    count += 1
                    
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing behavior for {row.get("StudentID", "Unknown")}: {str(e)}')
                )
        
        return count 