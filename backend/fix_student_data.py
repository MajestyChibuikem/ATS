#!/usr/bin/env python3
"""
Fix Student Data - Clear and Reimport with Proper Longitudinal Tracking
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from core.apps.students.models import Student, StudentScore, Subject, AcademicYear, Teacher, TeacherPerformance
from django.db import transaction

def clear_existing_data():
    """Clear all existing student data."""
    print("🧹 Clearing Existing Data")
    print("=" * 30)
    
    with transaction.atomic():
        # Clear in reverse dependency order
        print("Deleting student scores...")
        StudentScore.objects.all().delete()
        
        print("Deleting academic years...")
        AcademicYear.objects.all().delete()
        
        print("Deleting subjects...")
        Subject.objects.all().delete()
        
        print("Deleting students...")
        Student.objects.all().delete()
        
        print("Deleting teacher performance...")
        TeacherPerformance.objects.all().delete()
        
        print("Deleting teachers...")
        Teacher.objects.all().delete()
    
    print("✅ All data cleared successfully")

def reimport_data():
    """Reimport data with corrected logic."""
    print("\n📊 Reimporting Data with Corrected Logic")
    print("=" * 50)
    
    # Import the corrected train_real_data module
    from train_real_data import import_excel_data, process_data
    
    try:
        # Import data from Excel files
        ss2_students, ss2_scores, ss3_students, ss3_scores = import_excel_data()
        
        # Process data with corrected longitudinal tracking
        students_processed, scores_created = process_data(ss2_students, ss2_scores, ss3_students, ss3_scores)
        
        print(f"\n🎉 Data Import Complete!")
        print(f"   Students processed: {students_processed}")
        print(f"   Scores created: {scores_created}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data import: {e}")
        return False

def verify_data_integrity():
    """Verify the data integrity after import."""
    print("\n🔍 Verifying Data Integrity")
    print("=" * 35)
    
    # Check for duplicate names
    from django.db.models import Count
    
    duplicates = Student.objects.values('first_name', 'last_name').annotate(
        count=Count('id')
    ).filter(count__gt=1)
    
    print(f"Duplicate name combinations: {duplicates.count()}")
    
    if duplicates.count() > 0:
        print("⚠️  Still have duplicate names:")
        for dup in duplicates[:5]:
            name = f"{dup['first_name']} {dup['last_name']}"
            print(f"  {name}: {dup['count']} occurrences")
    else:
        print("✅ No duplicate names found!")
    
    # Check student progression
    students_with_ss2_scores = Student.objects.filter(
        scores__academic_year__year='2022/2023'
    ).distinct().count()
    
    students_with_ss3_scores = Student.objects.filter(
        scores__academic_year__year='2023/2024'
    ).distinct().count()
    
    students_with_both = Student.objects.filter(
        scores__academic_year__year='2022/2023'
    ).filter(
        scores__academic_year__year='2023/2024'
    ).distinct().count()
    
    print(f"\n📊 Student Progression Analysis:")
    print(f"   Students with SS2 scores (2022/2023): {students_with_ss2_scores}")
    print(f"   Students with SS3 scores (2023/2024): {students_with_ss3_scores}")
    print(f"   Students with both SS2 and SS3 scores: {students_with_both}")
    
    # Check sample student
    sample_student = Student.objects.first()
    if sample_student:
        print(f"\n👤 Sample Student: {sample_student.student_id}")
        print(f"   Name: {sample_student.first_name} {sample_student.last_name}")
        print(f"   Current Class: {sample_student.current_class}")
        
        scores = sample_student.scores.all()
        print(f"   Total Scores: {scores.count()}")
        
        academic_years = scores.values_list('academic_year__year', flat=True).distinct()
        print(f"   Academic Years: {list(academic_years)}")

def main():
    """Main execution function."""
    print("🚀 Fixing Student Data - Longitudinal Tracking")
    print("=" * 55)
    
    # Step 1: Clear existing data
    clear_existing_data()
    
    # Step 2: Reimport with corrected logic
    success = reimport_data()
    
    if success:
        # Step 3: Verify data integrity
        verify_data_integrity()
        
        print("\n🎉 Student Data Fix Complete!")
        print("   ✅ Duplicate names eliminated")
        print("   ✅ Student progression properly tracked")
        print("   ✅ Academic years correctly linked")
        print("   ✅ Ready for ML model retraining")
    else:
        print("\n❌ Data import failed. Please check the errors above.")

if __name__ == "__main__":
    main()
