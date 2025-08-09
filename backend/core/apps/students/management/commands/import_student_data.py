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
    Student, Subject, AcademicYear, StudentScore, StudentAttendance, StudentBehavior
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
    
    def handle(self, *args, **options):
        """Handle the command execution."""
        file_path = options['file_path']
        academic_year = options['academic_year']
        term = options['term']
        dry_run = options['dry_run']
        clear_existing = options['clear_existing']
        
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
                file_path, academic_year, term, dry_run, clear_existing
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully imported {imported_count} records'
                )
            )
            
        except Exception as e:
            logger.error(f'Import failed: {str(e)}')
            raise CommandError(f'Import failed: {str(e)}')
    
    def import_data(self, file_path, academic_year, term, dry_run, clear_existing):
        """Import data from Excel file."""
        imported_count = 0
        
        try:
            # Read Excel file
            self.stdout.write('Reading Excel file...')
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            # Get or create academic year
            academic_year_obj, created = AcademicYear.objects.get_or_create(
                year=academic_year,
                term=term,
                defaults={
                    'start_date': datetime(2023, 9, 1).date(),
                    'end_date': datetime(2024, 6, 30).date(),
                    'is_current': True
                }
            )
            
            if created:
                self.stdout.write(
                    f'Created academic year: {academic_year} - {term}'
                )
            
            # Process sheets in specific order to ensure dependencies are met
            sheet_order = ['Students', 'Subjects', 'SubjectScores', 'Attendance', 'BehavioralRecords']
            
            for sheet_name in sheet_order:
                if sheet_name in excel_data:
                    self.stdout.write(f'Processing sheet: {sheet_name}')
                    df = excel_data[sheet_name]
                    
                    if sheet_name == 'Students':
                        imported_count += self.import_students(df, dry_run, clear_existing)
                    elif sheet_name == 'Subjects':
                        imported_count += self.import_subjects(df, dry_run, clear_existing)
                    elif sheet_name == 'SubjectScores':
                        imported_count += self.import_scores(
                            df, academic_year_obj, dry_run, clear_existing
                        )
                    elif sheet_name == 'Attendance':
                        imported_count += self.import_attendance(df, dry_run, clear_existing)
                    elif sheet_name == 'BehavioralRecords':
                        imported_count += self.import_behavior(df, dry_run, clear_existing)
                else:
                    self.stdout.write(
                        self.style.WARNING(f'Sheet not found: {sheet_name}')
                    )
            
            # Process any remaining sheets
            for sheet_name, df in excel_data.items():
                if sheet_name not in sheet_order:
                    self.stdout.write(
                        self.style.WARNING(f'Skipping unknown sheet: {sheet_name}')
                    )
            
            return imported_count
            
        except Exception as e:
            logger.error(f'Error reading Excel file: {str(e)}')
            raise CommandError(f'Error reading Excel file: {str(e)}')
    
    def import_students(self, df, dry_run, clear_existing):
        """Import student data."""
        if clear_existing and not dry_run:
            Student.objects.all().delete()
            self.stdout.write('Cleared existing student data')
        
        imported_count = 0
        
        for _, row in df.iterrows():
            try:
                # Create student data with actual column names
                student_data = {
                    'student_id': str(row.get('StudentID', '')),
                    'first_name': str(row.get('FirstName', '')),
                    'last_name': str(row.get('LastName', '')),
                    'date_of_birth': pd.to_datetime(row.get('DateOfBirth')).date(),
                    'gender': str(row.get('Gender', '')),
                    'current_class': str(row.get('CurrentClass', '')),
                    'stream': str(row.get('Stream', '')),
                    'guardian_contact': str(row.get('GuardianContact', '')),
                    'guardian_name': str(row.get('GuardianName', '')) if 'GuardianName' in row else 'Guardian',
                    'guardian_email': str(row.get('GuardianEmail', '')) if 'GuardianEmail' in row and pd.notna(row.get('GuardianEmail')) else '',
                    'address': str(row.get('Address', '')) if 'Address' in row else 'Address not provided',
                    'admission_date': pd.to_datetime(row.get('EnrollmentDate', row.get('AdmissionDate', timezone.now()))).date(),
                }
                
                if not dry_run:
                    # Create or update student
                    student, created = Student.objects.update_or_create(
                        student_id=student_data['student_id'],
                        defaults=student_data
                    )
                    
                    if created:
                        imported_count += 1
                        self.stdout.write(f'Created student: {student.student_id}')
                    else:
                        self.stdout.write(f'Updated student: {student.student_id}')
                else:
                    imported_count += 1
                    self.stdout.write(f'Would create student: {student_data["student_id"]}')
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing student: {str(e)}')
                )
        
        return imported_count
    
    def import_subjects(self, df, dry_run, clear_existing):
        """Import subject data."""
        if clear_existing and not dry_run:
            Subject.objects.all().delete()
            self.stdout.write('Cleared existing subject data')
        
        imported_count = 0
        
        for _, row in df.iterrows():
            try:
                subject_data = {
                    'name': str(row.get('SubjectName', '')),
                    'code': str(row.get('SubjectCode', '')),
                    'category': str(row.get('Category', 'Core')),
                    'stream': str(row.get('Stream', 'All')),
                    'description': str(row.get('Description', '')),
                    'is_active': True,
                }
                
                if not dry_run:
                    subject, created = Subject.objects.update_or_create(
                        code=subject_data['code'],
                        defaults=subject_data
                    )
                    
                    if created:
                        imported_count += 1
                        self.stdout.write(f'Created subject: {subject.code}')
                    else:
                        self.stdout.write(f'Updated subject: {subject.code}')
                else:
                    imported_count += 1
                    self.stdout.write(f'Would create subject: {subject_data["code"]}')
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing subject: {str(e)}')
                )
        
        return imported_count
    
    def import_scores(self, df, academic_year, dry_run, clear_existing):
        """Import student scores."""
        if clear_existing and not dry_run:
            StudentScore.objects.filter(academic_year=academic_year).delete()
            self.stdout.write('Cleared existing score data')
        
        imported_count = 0
        
        # Group scores by student and subject
        grouped_scores = df.groupby(['StudentID', 'SubjectName'])
        
        for (student_id, subject_name), group in grouped_scores:
            try:
                # Get student and subject
                student_id = str(student_id)
                subject_name = str(subject_name)
                
                try:
                    student = Student.objects.get(student_id=student_id)
                except Student.DoesNotExist:
                    self.stdout.write(
                        self.style.WARNING(f'Student not found: {student_id}')
                    )
                    continue
                
                # Create or get subject
                subject, created = Subject.objects.get_or_create(
                    name=subject_name,
                    defaults={
                        'code': subject_name[:3].upper(),
                        'category': 'Core',
                        'stream': 'All',
                        'description': f'Subject: {subject_name}',
                        'is_active': True,
                    }
                )
                
                # Calculate scores from assessment types
                continuous_assessment = 0
                examination_score = 0
                
                for _, row in group.iterrows():
                    assessment_type = str(row.get('AssessmentType', '')).lower()
                    score = float(row.get('Score', 0))
                    
                    if 'continuous' in assessment_type or 'ca' in assessment_type:
                        continuous_assessment = score
                    elif 'exam' in assessment_type or 'examination' in assessment_type:
                        examination_score = score
                    else:
                        # Default to continuous assessment if type is unclear
                        continuous_assessment = score
                
                # Create score data
                score_data = {
                    'student': student,
                    'subject': subject,
                    'academic_year': academic_year,
                    'continuous_assessment': continuous_assessment,
                    'examination_score': examination_score,
                    'remarks': str(group.iloc[0].get('TeacherRemarks', '')),
                }
                
                if not dry_run:
                    score, created = StudentScore.objects.update_or_create(
                        student=student,
                        subject=subject,
                        academic_year=academic_year,
                        defaults=score_data
                    )
                    
                    if created:
                        imported_count += 1
                        self.stdout.write(
                            f'Created score: {student.student_id} - {subject.name}'
                        )
                    else:
                        self.stdout.write(
                            f'Updated score: {student.student_id} - {subject.name}'
                        )
                else:
                    imported_count += 1
                    self.stdout.write(
                        f'Would create score: {student_id} - {subject_name}'
                    )
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing score: {str(e)}')
                )
        
        return imported_count
    
    def import_attendance(self, df, dry_run, clear_existing):
        """Import attendance records."""
        if clear_existing and not dry_run:
            StudentAttendance.objects.all().delete()
            self.stdout.write('Cleared existing attendance data')
        
        imported_count = 0
        
        for _, row in df.iterrows():
            try:
                student_id = str(row.get('StudentID', ''))
                
                try:
                    student = Student.objects.get(student_id=student_id)
                except Student.DoesNotExist:
                    self.stdout.write(
                        self.style.WARNING(f'Student not found: {student_id}')
                    )
                    continue
                
                attendance_data = {
                    'student': student,
                    'date': pd.to_datetime(row.get('Date', row.get('AttendanceDate'))).date(),
                    'status': str(row.get('Status', 'Present')),
                    'reason': str(row.get('Reason', '')),
                }
                
                if not dry_run:
                    attendance, created = StudentAttendance.objects.update_or_create(
                        student=student,
                        date=attendance_data['date'],
                        defaults=attendance_data
                    )
                    
                    if created:
                        imported_count += 1
                        self.stdout.write(
                            f'Created attendance: {student.student_id} - {attendance_data["date"]}'
                        )
                    else:
                        self.stdout.write(
                            f'Updated attendance: {student.student_id} - {attendance_data["date"]}'
                        )
                else:
                    imported_count += 1
                    self.stdout.write(
                        f'Would create attendance: {student_id} - {attendance_data["date"]}'
                    )
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing attendance: {str(e)}')
                )
        
        return imported_count
    
    def import_behavior(self, df, dry_run, clear_existing):
        """Import behavior records."""
        if clear_existing and not dry_run:
            StudentBehavior.objects.all().delete()
            self.stdout.write('Cleared existing behavior data')
        
        imported_count = 0
        
        for _, row in df.iterrows():
            try:
                student_id = str(row.get('StudentID', ''))
                
                try:
                    student = Student.objects.get(student_id=student_id)
                except Student.DoesNotExist:
                    self.stdout.write(
                        self.style.WARNING(f'Student not found: {student_id}')
                    )
                    continue
                
                behavior_data = {
                    'student': student,
                    'date': pd.to_datetime(row.get('Date', row.get('BehaviorDate'))).date(),
                    'category': str(row.get('Category', 'Other')),
                    'description': str(row.get('Description', '')),
                    'severity': str(row.get('Severity', 'Medium')),
                    'action_taken': str(row.get('ActionTaken', '')),
                }
                
                if not dry_run:
                    behavior, created = StudentBehavior.objects.update_or_create(
                        student=student,
                        date=behavior_data['date'],
                        category=behavior_data['category'],
                        defaults=behavior_data
                    )
                    
                    if created:
                        imported_count += 1
                        self.stdout.write(
                            f'Created behavior: {student.student_id} - {behavior_data["date"]}'
                        )
                    else:
                        self.stdout.write(
                            f'Updated behavior: {student.student_id} - {behavior_data["date"]}'
                        )
                else:
                    imported_count += 1
                    self.stdout.write(
                        f'Would create behavior: {student_id} - {behavior_data["date"]}'
                    )
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error importing behavior: {str(e)}')
                )
        
        return imported_count 