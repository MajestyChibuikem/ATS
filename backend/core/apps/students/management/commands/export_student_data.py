"""
Django management command for exporting student data to Excel files.
"""

import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
import logging
from pathlib import Path

from core.apps.students.models import Student, Subject, StudentScore

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Management command for exporting student data."""
    
    help = 'Export student data to Excel files'
    
    def add_arguments(self, parser):
        """Add command arguments."""
        parser.add_argument(
            'output_path',
            type=str,
            help='Path for the output Excel file'
        )
        parser.add_argument(
            '--include-scores',
            action='store_true',
            help='Include student scores in export'
        )
    
    def handle(self, *args, **options):
        """Handle the command execution."""
        output_path = options['output_path']
        include_scores = options['include_scores']
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting export to: {output_path}')
        )
        
        try:
            # Export data
            self.export_data(output_path, include_scores)
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully exported data to: {output_path}')
            )
            
        except Exception as e:
            logger.error(f'Export failed: {str(e)}')
            raise CommandError(f'Export failed: {str(e)}')
    
    def export_data(self, output_path, include_scores):
        """Export data to Excel file."""
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Export students
            self.export_students(writer)
            
            # Export subjects
            self.export_subjects(writer)
            
            # Export scores if requested
            if include_scores:
                self.export_scores(writer)
    
    def export_students(self, writer):
        """Export student data."""
        self.stdout.write('Exporting students...')
        
        # Convert to DataFrame
        students_data = []
        for student in Student.objects.all():
            students_data.append({
                'StudentID': student.student_id,
                'FirstName': student.first_name,
                'LastName': student.last_name,
                'DateOfBirth': student.date_of_birth,
                'Gender': student.gender,
                'CurrentClass': student.current_class,
                'Stream': student.stream,
                'GuardianName': student.guardian_name,
                'GuardianContact': student.guardian_contact,
                'GuardianEmail': student.guardian_email,
                'Address': student.address,
                'AdmissionDate': student.admission_date,
                'IsActive': student.is_active,
            })
        
        df = pd.DataFrame(students_data)
        df.to_excel(writer, sheet_name='Students', index=False)
        
        self.stdout.write(f'Exported {len(students_data)} students')
    
    def export_subjects(self, writer):
        """Export subject data."""
        self.stdout.write('Exporting subjects...')
        
        subjects_data = []
        for subject in Subject.objects.all():
            subjects_data.append({
                'SubjectCode': subject.code,
                'SubjectName': subject.name,
                'Category': subject.category,
                'Stream': subject.stream,
                'Description': subject.description,
                'IsActive': subject.is_active,
            })
        
        df = pd.DataFrame(subjects_data)
        df.to_excel(writer, sheet_name='Subjects', index=False)
        
        self.stdout.write(f'Exported {len(subjects_data)} subjects')
    
    def export_scores(self, writer):
        """Export student scores."""
        self.stdout.write('Exporting scores...')
        
        # Convert to DataFrame
        scores_data = []
        for score in StudentScore.objects.select_related('student', 'subject', 'academic_year'):
            scores_data.append({
                'StudentID': score.student.student_id,
                'SubjectName': score.subject.name,
                'SubjectCode': score.subject.code,
                'AcademicYear': score.academic_year.year,
                'Term': score.academic_year.term,
                'ContinuousAssessment': score.continuous_assessment,
                'ExaminationScore': score.examination_score,
                'TotalScore': score.total_score,
                'Grade': score.grade,
                'Remarks': score.remarks,
            })
        
        df = pd.DataFrame(scores_data)
        df.to_excel(writer, sheet_name='SubjectScores', index=False)
        
        self.stdout.write(f'Exported {len(scores_data)} scores') 