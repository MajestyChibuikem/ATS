#!/usr/bin/env python
"""
Test script for the data pipeline.
"""

import os
import sys
import django

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

import pandas as pd
from core.apps.students.services.validation import DataValidator
from core.apps.students.models import Student, Subject, StudentScore

def test_data_validation():
    """Test data validation functionality."""
    print("Testing data validation...")
    
    # Test with valid data
    valid_data = pd.DataFrame({
        'StudentID': ['STD0001', 'STD0002'],
        'FirstName': ['John', 'Jane'],
        'LastName': ['Doe', 'Smith'],
        'DateOfBirth': ['2006-01-01', '2006-02-01'],
        'Gender': ['Male', 'Female']
    })
    
    is_valid, errors = DataValidator.validate_student_data(valid_data)
    print(f"Valid data test: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test with invalid data
    invalid_data = pd.DataFrame({
        'StudentID': ['STD0001', 'STD0001'],  # Duplicate
        'FirstName': ['John', 'Jane'],
        'LastName': ['Doe', 'Smith'],
        'DateOfBirth': ['2006-01-01', '2006-02-01'],
        'Gender': ['Male', 'Invalid']  # Invalid gender
    })
    
    is_valid, errors = DataValidator.validate_student_data(invalid_data)
    print(f"Invalid data test: {'PASS' if not is_valid else 'FAIL'}")
    if errors:
        print(f"Expected errors: {errors}")

def test_database_queries():
    """Test database queries and statistics."""
    print("\nTesting database queries...")
    
    # Get basic statistics
    total_students = Student.objects.count()
    total_subjects = Subject.objects.count()
    total_scores = StudentScore.objects.count()
    
    print(f"Total students: {total_students}")
    print(f"Total subjects: {total_subjects}")
    print(f"Total scores: {total_scores}")
    
    # Get gender distribution
    gender_dist = Student.objects.values('gender').annotate(
        count=django.db.models.Count('id')
    )
    print(f"Gender distribution: {list(gender_dist)}")
    
    # Get class distribution
    class_dist = Student.objects.values('current_class').annotate(
        count=django.db.models.Count('id')
    )
    print(f"Class distribution: {list(class_dist)}")
    
    # Get stream distribution
    stream_dist = Student.objects.values('stream').annotate(
        count=django.db.models.Count('id')
    )
    print(f"Stream distribution: {list(stream_dist)}")
    
    # Get average scores
    avg_scores = StudentScore.objects.aggregate(
        avg_total=django.db.models.Avg('total_score'),
        avg_ca=django.db.models.Avg('continuous_assessment'),
        avg_exam=django.db.models.Avg('examination_score')
    )
    print(f"Average scores: {avg_scores}")

def test_subject_performance():
    """Test subject performance analysis."""
    print("\nTesting subject performance analysis...")
    
    # Get performance by subject
    subject_performance = StudentScore.objects.values('subject__name').annotate(
        avg_score=django.db.models.Avg('total_score'),
        total_students=django.db.models.Count('student', distinct=True),
        pass_count=django.db.models.Count('id', filter=django.db.models.Q(total_score__gte=50))
    )
    
    for subject in subject_performance:
        pass_rate = (subject['pass_count'] / subject['total_students']) * 100 if subject['total_students'] > 0 else 0
        print(f"{subject['subject__name']}: "
              f"Avg={subject['avg_score']:.1f}, "
              f"Students={subject['total_students']}, "
              f"Pass Rate={pass_rate:.1f}%")

if __name__ == '__main__':
    print("=== Data Pipeline Test ===")
    
    try:
        test_data_validation()
        test_database_queries()
        test_subject_performance()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc() 