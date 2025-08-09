#!/usr/bin/env python
"""
Debug script to check data pipeline.
"""

import os
import sys
import django

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from core.apps.students.models import Student, StudentScore, Subject

def main():
    """Debug the data pipeline."""
    print("=== Data Pipeline Debug ===")
    
    # Check students
    total_students = Student.objects.count()
    print(f"Total students: {total_students}")
    
    # Check scores
    total_scores = StudentScore.objects.count()
    print(f"Total scores: {total_scores}")
    
    # Check subjects
    total_subjects = Subject.objects.count()
    print(f"Total subjects: {total_subjects}")
    
    # Check sample data
    if total_scores > 0:
        sample_score = StudentScore.objects.first()
        print(f"Sample score: {sample_score.student.student_id} - {sample_score.subject.name} - {sample_score.total_score}")
    
    # Check data distribution
    print("\n=== Data Distribution ===")
    
    # Students by class
    class_dist = Student.objects.values('current_class').annotate(
        count=django.db.models.Count('id')
    )
    print("Students by class:")
    for item in class_dist:
        print(f"  {item['current_class']}: {item['count']}")
    
    # Scores by subject
    subject_dist = StudentScore.objects.values('subject__name').annotate(
        count=django.db.models.Count('id')
    )
    print("\nScores by subject:")
    for item in subject_dist:
        print(f"  {item['subject__name']}: {item['count']}")
    
    # Check if we have enough data for training
    print("\n=== Training Data Check ===")
    
    # Get unique students with scores
    students_with_scores = StudentScore.objects.values('student_id').distinct().count()
    print(f"Students with scores: {students_with_scores}")
    
    # Get subjects with sufficient data
    subjects_with_data = StudentScore.objects.values('subject__name').annotate(
        student_count=django.db.models.Count('student', distinct=True)
    ).filter(student_count__gte=10)
    
    print(f"Subjects with >=10 students: {subjects_with_data.count()}")
    for item in subjects_with_data:
        print(f"  {item['subject__name']}: {item['student_count']} students")

if __name__ == '__main__':
    main()
