#!/usr/bin/env python3
"""
Import Teacher Data from Excel Files
"""

import os
import sys
import django
import pandas as pd
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from core.apps.students.models import Teacher, Subject, TeacherPerformance

def check_excel_structure():
    """Check the structure of Excel files."""
    print("📊 Checking Excel File Structure")
    print("=" * 40)
    
    # Check SS3 file
    try:
        ss3_file = pd.ExcelFile('student_records_SS3.xlsx')
        print(f"SS3 Sheets: {ss3_file.sheet_names}")
        
        if 'Teachers' in ss3_file.sheet_names:
            df_teachers = pd.read_excel('student_records_SS3.xlsx', sheet_name='Teachers')
            print(f"SS3 Teachers columns: {list(df_teachers.columns)}")
            print(f"SS3 Teachers count: {len(df_teachers)}")
            print("Sample teacher data:")
            print(df_teachers.head(3))
        else:
            print("❌ No 'Teachers' sheet found in SS3 file")
            
        if 'TeacherPerformance' in ss3_file.sheet_names:
            df_performance = pd.read_excel('student_records_SS3.xlsx', sheet_name='TeacherPerformance')
            print(f"SS3 TeacherPerformance columns: {list(df_performance.columns)}")
            print(f"SS3 TeacherPerformance count: {len(df_performance)}")
        else:
            print("❌ No 'TeacherPerformance' sheet found in SS3 file")
            
    except Exception as e:
        print(f"❌ Error reading SS3 file: {e}")
    
    # Check SS2 file
    try:
        ss2_file = pd.ExcelFile('student_records_SS2.xlsx')
        print(f"\nSS2 Sheets: {ss2_file.sheet_names}")
        
        if 'Teachers' in ss2_file.sheet_names:
            df_teachers = pd.read_excel('student_records_SS2.xlsx', sheet_name='Teachers')
            print(f"SS2 Teachers columns: {list(df_teachers.columns)}")
            print(f"SS2 Teachers count: {len(df_teachers)}")
        else:
            print("❌ No 'Teachers' sheet found in SS2 file")
            
    except Exception as e:
        print(f"❌ Error reading SS2 file: {e}")

def import_teachers_from_excel():
    """Import teachers from Excel files."""
    print("\n👨‍🏫 Importing Teachers")
    print("=" * 25)
    
    teachers_created = 0
    
    # Import from SS3
    try:
        if 'Teachers' in pd.ExcelFile('student_records_SS3.xlsx').sheet_names:
            df_teachers = pd.read_excel('student_records_SS3.xlsx', sheet_name='Teachers')
            print(f"Importing {len(df_teachers)} teachers from SS3...")
            
            for _, row in df_teachers.iterrows():
                try:
                    teacher, created = Teacher.objects.get_or_create(
                        teacher_id=str(row.get('TeacherID', f"TCH_{teachers_created+1}")),
                        defaults={
                            'name': str(row.get('Name', f'Teacher {teachers_created+1}')),
                            'years_experience': int(row.get('YearsExperience', 5)),
                            'qualification_level': str(row.get('QualificationLevel', 'B.Ed')),
                            'specialization': str(row.get('Specialization', 'General')),
                            'teaching_load': int(row.get('TeachingLoad', 25)),
                            'performance_rating': float(row.get('PerformanceRating', 4.0)),
                            'years_at_school': int(row.get('YearsAtSchool', 3)),
                            'is_active': True
                        }
                    )
                    
                    if created:
                        teachers_created += 1
                        print(f"✅ Created teacher: {teacher.name} ({teacher.teacher_id})")
                        
                except Exception as e:
                    print(f"⚠️  Error creating teacher: {e}")
                    
    except Exception as e:
        print(f"❌ Error importing SS3 teachers: {e}")
    
    # Import from SS2
    try:
        if 'Teachers' in pd.ExcelFile('student_records_SS2.xlsx').sheet_names:
            df_teachers = pd.read_excel('student_records_SS2.xlsx', sheet_name='Teachers')
            print(f"Importing {len(df_teachers)} teachers from SS2...")
            
            for _, row in df_teachers.iterrows():
                try:
                    teacher, created = Teacher.objects.get_or_create(
                        teacher_id=str(row.get('TeacherID', f"TCH_{teachers_created+1}")),
                        defaults={
                            'name': str(row.get('Name', f'Teacher {teachers_created+1}')),
                            'years_experience': int(row.get('YearsExperience', 5)),
                            'qualification_level': str(row.get('QualificationLevel', 'B.Ed')),
                            'specialization': str(row.get('Specialization', 'General')),
                            'teaching_load': int(row.get('TeachingLoad', 25)),
                            'performance_rating': float(row.get('PerformanceRating', 4.0)),
                            'years_at_school': int(row.get('YearsAtSchool', 3)),
                            'is_active': True
                        }
                    )
                    
                    if created:
                        teachers_created += 1
                        print(f"✅ Created teacher: {teacher.name} ({teacher.teacher_id})")
                        
                except Exception as e:
                    print(f"⚠️  Error creating teacher: {e}")
                    
    except Exception as e:
        print(f"❌ Error importing SS2 teachers: {e}")
    
    print(f"\n✅ Total teachers created: {teachers_created}")
    return teachers_created

def create_sample_teachers():
    """Create sample teachers if no teachers exist."""
    print("\n🎭 Creating Sample Teachers")
    print("=" * 30)
    
    if Teacher.objects.count() > 0:
        print(f"✅ Teachers already exist: {Teacher.objects.count()}")
        return
    
    sample_teachers = [
        {
            'teacher_id': 'TCH001',
            'name': 'Dr. Sarah Johnson',
            'years_experience': 15,
            'qualification_level': 'PhD',
            'specialization': 'Mathematics',
            'teaching_load': 25,
            'performance_rating': 4.5,
            'years_at_school': 8
        },
        {
            'teacher_id': 'TCH002',
            'name': 'Mr. Michael Chen',
            'years_experience': 8,
            'qualification_level': 'M.Ed',
            'specialization': 'Sciences',
            'teaching_load': 30,
            'performance_rating': 4.2,
            'years_at_school': 5
        },
        {
            'teacher_id': 'TCH003',
            'name': 'Mrs. Patricia Williams',
            'years_experience': 12,
            'qualification_level': 'B.Ed',
            'specialization': 'Languages',
            'teaching_load': 20,
            'performance_rating': 4.8,
            'years_at_school': 10
        },
        {
            'teacher_id': 'TCH004',
            'name': 'Mr. David Rodriguez',
            'years_experience': 6,
            'qualification_level': 'B.Sc + PGDE',
            'specialization': 'Arts',
            'teaching_load': 28,
            'performance_rating': 3.9,
            'years_at_school': 4
        },
        {
            'teacher_id': 'TCH005',
            'name': 'Dr. Emily Brown',
            'years_experience': 20,
            'qualification_level': 'PhD',
            'specialization': 'Sciences',
            'teaching_load': 22,
            'performance_rating': 4.9,
            'years_at_school': 15
        }
    ]
    
    for teacher_data in sample_teachers:
        teacher, created = Teacher.objects.get_or_create(
            teacher_id=teacher_data['teacher_id'],
            defaults={
                'name': teacher_data['name'],
                'years_experience': teacher_data['years_experience'],
                'qualification_level': teacher_data['qualification_level'],
                'specialization': teacher_data['specialization'],
                'teaching_load': teacher_data['teaching_load'],
                'performance_rating': teacher_data['performance_rating'],
                'years_at_school': teacher_data['years_at_school'],
                'is_active': True
            }
        )
        
        if created:
            print(f"✅ Created sample teacher: {teacher.name}")
    
    print(f"✅ Created {len(sample_teachers)} sample teachers")

if __name__ == "__main__":
    print("🚀 Teacher Import Script")
    print("=" * 30)
    
    # Check Excel structure
    check_excel_structure()
    
    # Try to import from Excel
    teachers_imported = import_teachers_from_excel()
    
    # If no teachers imported, create sample teachers
    if teachers_imported == 0:
        create_sample_teachers()
    
    # Final count
    total_teachers = Teacher.objects.count()
    print(f"\n🎉 Total teachers in database: {total_teachers}")
    
    # Show some teachers
    print("\n📋 Sample Teachers:")
    for teacher in Teacher.objects.all()[:5]:
        print(f"  - {teacher.name} ({teacher.teacher_id}) - {teacher.specialization}")
