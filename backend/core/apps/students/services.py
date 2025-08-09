"""
Data validation and processing services for student data.
"""

import pandas as pd
import numpy as np
from django.core.exceptions import ValidationError
from django.db import transaction
from typing import Dict, List, Tuple, Optional
import logging

from .models import Student, Subject, StudentScore, AcademicYear

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation service for student data."""
    
    @staticmethod
    def validate_student_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate student data DataFrame."""
        errors = []
        
        # Check required columns
        required_columns = ['StudentID', 'FirstName', 'LastName', 'DateOfBirth', 'Gender']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate student IDs
        if 'StudentID' in df.columns:
            duplicates = df[df['StudentID'].duplicated()]['StudentID'].tolist()
            if duplicates:
                errors.append(f"Duplicate student IDs found: {duplicates[:5]}...")
        
        # Check for valid gender values
        if 'Gender' in df.columns:
            valid_genders = ['Male', 'Female']
            invalid_genders = df[~df['Gender'].isin(valid_genders)]['Gender'].unique()
            if len(invalid_genders) > 0:
                errors.append(f"Invalid gender values: {invalid_genders}")
        
        # Check for valid class levels
        if 'CurrentClass' in df.columns:
            valid_classes = ['SS1', 'SS2', 'SS3']
            invalid_classes = df[~df['CurrentClass'].isin(valid_classes)]['CurrentClass'].unique()
            if len(invalid_classes) > 0:
                errors.append(f"Invalid class levels: {invalid_classes}")
        
        # Check for valid streams
        if 'Stream' in df.columns:
            valid_streams = ['Science', 'Arts', 'Commercial']
            invalid_streams = df[~df['Stream'].isin(valid_streams)]['Stream'].unique()
            if len(invalid_streams) > 0:
                errors.append(f"Invalid streams: {invalid_streams}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_score_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate score data DataFrame."""
        errors = []
        
        # Check required columns
        required_columns = ['StudentID', 'SubjectName', 'Score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for valid score ranges
        if 'Score' in df.columns:
            invalid_scores = df[(df['Score'] < 0) | (df['Score'] > 100)]['Score'].tolist()
            if invalid_scores:
                errors.append(f"Invalid scores (not between 0-100): {invalid_scores[:5]}...")
        
        # Check for missing scores
        if 'Score' in df.columns:
            missing_scores = df[df['Score'].isna()].shape[0]
            if missing_scores > 0:
                errors.append(f"Missing scores: {missing_scores} records")
        
        return len(errors) == 0, errors


class DataProcessor:
    """Data processing service for student data."""
    
    @staticmethod
    def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize student data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['StudentID'])
        
        # Standardize column names
        df.columns = df.columns.str.strip()
        
        # Clean string columns
        string_columns = ['FirstName', 'LastName', 'Gender', 'CurrentClass', 'Stream']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Standardize gender values
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].str.title()
            df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
        
        # Standardize class levels
        if 'CurrentClass' in df.columns:
            df['CurrentClass'] = df['CurrentClass'].str.upper()
        
        # Standardize streams
        if 'Stream' in df.columns:
            df['Stream'] = df['Stream'].str.title()
        
        return df
    
    @staticmethod
    def clean_score_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize score data."""
        # Remove rows with missing essential data
        df = df.dropna(subset=['StudentID', 'SubjectName', 'Score'])
        
        # Ensure score is numeric
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        
        # Remove invalid scores
        df = df[(df['Score'] >= 0) & (df['Score'] <= 100)]
        
        # Standardize subject names
        if 'SubjectName' in df.columns:
            df['SubjectName'] = df['SubjectName'].str.strip().str.title()
        
        return df
    
    @staticmethod
    def calculate_statistics(df: pd.DataFrame) -> Dict:
        """Calculate basic statistics for the dataset."""
        stats = {
            'total_students': len(df) if 'StudentID' in df.columns else 0,
            'unique_students': df['StudentID'].nunique() if 'StudentID' in df.columns else 0,
            'gender_distribution': df['Gender'].value_counts().to_dict() if 'Gender' in df.columns else {},
            'class_distribution': df['CurrentClass'].value_counts().to_dict() if 'CurrentClass' in df.columns else {},
            'stream_distribution': df['Stream'].value_counts().to_dict() if 'Stream' in df.columns else {},
        }
        
        if 'Score' in df.columns:
            stats.update({
                'score_statistics': {
                    'mean': df['Score'].mean(),
                    'median': df['Score'].median(),
                    'std': df['Score'].std(),
                    'min': df['Score'].min(),
                    'max': df['Score'].max(),
                }
            })
        
        return stats


class DataAnalyzer:
    """Data analysis service for student performance."""
    
    @staticmethod
    def analyze_student_performance(student_id: str) -> Dict:
        """Analyze performance for a specific student."""
        try:
            student = Student.objects.get(student_id=student_id)
            scores = StudentScore.objects.filter(student=student).select_related('subject')
            
            if not scores.exists():
                return {'error': 'No scores found for this student'}
            
            # Calculate performance metrics
            total_scores = list(scores.values_list('total_score', flat=True))
            continuous_scores = list(scores.values_list('continuous_assessment', flat=True))
            exam_scores = list(scores.values_list('examination_score', flat=True))
            
            analysis = {
                'student_id': student_id,
                'student_name': f"{student.first_name} {student.last_name}",
                'class': student.current_class,
                'stream': student.stream,
                'total_subjects': len(scores),
                'average_total_score': np.mean(total_scores),
                'average_continuous_assessment': np.mean(continuous_scores),
                'average_examination_score': np.mean(exam_scores),
                'best_subject': None,
                'worst_subject': None,
                'grade_distribution': {},
            }
            
            # Find best and worst subjects
            subject_performance = []
            for score in scores:
                subject_performance.append({
                    'subject': score.subject.name,
                    'total_score': score.total_score,
                    'grade': score.grade,
                })
            
            if subject_performance:
                subject_performance.sort(key=lambda x: x['total_score'], reverse=True)
                analysis['best_subject'] = subject_performance[0]
                analysis['worst_subject'] = subject_performance[-1]
                
                # Grade distribution
                grades = [s['grade'] for s in subject_performance]
                analysis['grade_distribution'] = {grade: grades.count(grade) for grade in set(grades)}
            
            return analysis
            
        except Student.DoesNotExist:
            return {'error': 'Student not found'}
        except Exception as e:
            logger.error(f"Error analyzing student performance: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    @staticmethod
    def analyze_class_performance(class_level: str, stream: Optional[str] = None) -> Dict:
        """Analyze performance for a class level."""
        try:
            # Get students in the class
            students_query = Student.objects.filter(current_class=class_level)
            if stream:
                students_query = students_query.filter(stream=stream)
            
            students = students_query.values_list('student_id', flat=True)
            
            if not students.exists():
                return {'error': f'No students found in {class_level}'}
            
            # Get all scores for these students
            scores = StudentScore.objects.filter(
                student__student_id__in=students
            ).select_related('student', 'subject')
            
            if not scores.exists():
                return {'error': 'No scores found for this class'}
            
            # Calculate class statistics
            total_scores = list(scores.values_list('total_score', flat=True))
            continuous_scores = list(scores.values_list('continuous_assessment', flat=True))
            exam_scores = list(scores.values_list('examination_score', flat=True))
            
            analysis = {
                'class_level': class_level,
                'stream': stream,
                'total_students': len(students),
                'total_scores': len(scores),
                'average_total_score': np.mean(total_scores),
                'average_continuous_assessment': np.mean(continuous_scores),
                'average_examination_score': np.mean(exam_scores),
                'score_distribution': {
                    'excellent': len([s for s in total_scores if s >= 80]),
                    'good': len([s for s in total_scores if 70 <= s < 80]),
                    'average': len([s for s in total_scores if 60 <= s < 70]),
                    'below_average': len([s for s in total_scores if s < 60]),
                },
                'subject_performance': {},
            }
            
            # Analyze performance by subject
            subjects = scores.values_list('subject__name', flat=True).distinct()
            for subject_name in subjects:
                subject_scores = scores.filter(subject__name=subject_name)
                subject_total_scores = list(subject_scores.values_list('total_score', flat=True))
                
                analysis['subject_performance'][subject_name] = {
                    'average_score': np.mean(subject_total_scores),
                    'total_students': len(subject_scores),
                    'pass_rate': len([s for s in subject_total_scores if s >= 50]) / len(subject_total_scores) * 100,
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing class performance: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'} 