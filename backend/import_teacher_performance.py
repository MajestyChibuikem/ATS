#!/usr/bin/env python3
"""
Import Teacher Performance Data from Excel Files
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

def import_teacher_performance():
    """Import teacher performance data from Excel files."""
    print("📊 Importing Teacher Performance Data")
    print("=" * 40)
    
    performance_created = 0
    
    # Import from SS3
    try:
        if 'TeacherPerformance' in pd.ExcelFile('student_records_SS3.xlsx').sheet_names:
            df_performance = pd.read_excel('student_records_SS3.xlsx', sheet_name='TeacherPerformance')
            print(f"Importing {len(df_performance)} performance records from SS3...")
            
            for _, row in df_performance.iterrows():
                try:
                    teacher_id = str(row.get('TeacherID'))
                    subject_name = str(row.get('SubjectName'))
                    
                    # Get teacher and subject
                    teacher = Teacher.objects.filter(teacher_id=teacher_id).first()
                    subject, _ = Subject.objects.get_or_create(name=subject_name)
                    
                    if teacher:
                        performance, created = TeacherPerformance.objects.get_or_create(
                            teacher=teacher,
                            subject=subject,
                            academic_year=str(row.get('AcademicYear', '2024/2025')),
                            defaults={
                                'average_class_score': float(row.get('AverageClassScore', 75.0)),
                                'number_of_students': int(row.get('NumberOfStudents', 30)),
                                'pass_rate': float(row.get('PassRate', 80.0)),
                                'student_satisfaction_rating': float(row.get('StudentSatisfactionRating', 4.0)),
                                'professional_development_hours': int(row.get('ProfessionalDevelopmentHours', 20)),
                                'class_attendance_rate': float(row.get('ClassAttendanceRate', 85.0))
                            }
                        )
                        
                        if created:
                            performance_created += 1
                            print(f"✅ Created performance record for {teacher.name} - {subject.name}")
                    else:
                        print(f"⚠️  Teacher {teacher_id} not found for performance record")
                        
                except Exception as e:
                    print(f"⚠️  Error creating performance record: {e}")
                    
    except Exception as e:
        print(f"❌ Error importing SS3 performance: {e}")
    
    # Import from SS2
    try:
        if 'TeacherPerformance' in pd.ExcelFile('student_records_SS2.xlsx').sheet_names:
            df_performance = pd.read_excel('student_records_SS2.xlsx', sheet_name='TeacherPerformance')
            print(f"Importing {len(df_performance)} performance records from SS2...")
            
            for _, row in df_performance.iterrows():
                try:
                    teacher_id = str(row.get('TeacherID'))
                    subject_name = str(row.get('SubjectName'))
                    
                    # Get teacher and subject
                    teacher = Teacher.objects.filter(teacher_id=teacher_id).first()
                    subject, _ = Subject.objects.get_or_create(name=subject_name)
                    
                    if teacher:
                        performance, created = TeacherPerformance.objects.get_or_create(
                            teacher=teacher,
                            subject=subject,
                            academic_year=str(row.get('AcademicYear', '2024/2025')),
                            defaults={
                                'average_class_score': float(row.get('AverageClassScore', 75.0)),
                                'number_of_students': int(row.get('NumberOfStudents', 30)),
                                'pass_rate': float(row.get('PassRate', 80.0)),
                                'student_satisfaction_rating': float(row.get('StudentSatisfactionRating', 4.0)),
                                'professional_development_hours': int(row.get('ProfessionalDevelopmentHours', 20)),
                                'class_attendance_rate': float(row.get('ClassAttendanceRate', 85.0))
                            }
                        )
                        
                        if created:
                            performance_created += 1
                            print(f"✅ Created performance record for {teacher.name} - {subject.name}")
                    else:
                        print(f"⚠️  Teacher {teacher_id} not found for performance record")
                        
                except Exception as e:
                    print(f"⚠️  Error creating performance record: {e}")
                    
    except Exception as e:
        print(f"❌ Error importing SS2 performance: {e}")
    
    print(f"\n✅ Total performance records created: {performance_created}")
    return performance_created

if __name__ == "__main__":
    print("🚀 Teacher Performance Import Script")
    print("=" * 40)
    
    # Import performance data
    performance_imported = import_teacher_performance()
    
    # Final count
    total_performance = TeacherPerformance.objects.count()
    print(f"\n🎉 Total performance records in database: {total_performance}")
    
    # Show some performance records
    print("\n📋 Sample Performance Records:")
    for performance in TeacherPerformance.objects.all()[:5]:
        print(f"  - {performance.teacher.name} - {performance.subject.name}: {performance.average_class_score}%")
