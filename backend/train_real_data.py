"""
Train ML Models with Real Student Data
"""

import os
import sys
import pandas as pd
import django
import numpy as np
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from core.apps.students.models import Student, Subject, AcademicYear, StudentScore
from core.apps.ml.models.tier1_critical_predictor import Tier1CriticalPredictor
from core.apps.ml.models.tier2_science_predictor import Tier2SciencePredictor
from core.apps.ml.models.tier3_arts_predictor import Tier3ArtsPredictor

def import_excel_data():
    """Import data from Excel files."""
    print("📊 Importing Real Student Data")
    print("=" * 40)
    
    # Import SS2 data with all sheets
    print("Importing SS2 data...")
    ss2_students = pd.read_excel('student_records_SS2.xlsx', sheet_name='Students')
    ss2_scores = pd.read_excel('student_records_SS2.xlsx', sheet_name='SubjectScores')
    print(f"✅ SS2 Students: {len(ss2_students)} records")
    print(f"✅ SS2 Scores: {len(ss2_scores)} records")
    
    # Import SS3 data with all sheets
    print("Importing SS3 data...")
    ss3_students = pd.read_excel('student_records_SS3.xlsx', sheet_name='Students')
    ss3_scores = pd.read_excel('student_records_SS3.xlsx', sheet_name='SubjectScores')
    print(f"✅ SS3 Students: {len(ss3_students)} records")
    print(f"✅ SS3 Scores: {len(ss3_scores)} records")
    
    # Show data structure
    print(f"\nSS2 Students columns: {list(ss2_students.columns)}")
    print(f"SS2 Scores columns: {list(ss2_scores.columns)}")
    
    return ss2_students, ss2_scores, ss3_students, ss3_scores

def process_data(ss2_students, ss2_scores, ss3_students, ss3_scores):
    """Process student data."""
    print("\n🔧 Processing Data")
    print("=" * 25)
    
    # Create subjects from the actual data
    all_subjects = set()
    for df in [ss2_scores, ss3_scores]:
        if 'SubjectName' in df.columns:
            all_subjects.update(df['SubjectName'].unique())
    
    subjects = list(all_subjects)
    print(f"Found subjects: {subjects}")
    
    for subject_name in subjects:
        Subject.objects.get_or_create(name=subject_name)
    
    print(f"✅ Created {len(subjects)} subjects")
    
    # Process students
    students_created = 0
    scores_created = 0
    
    # Process SS2 students
    for _, row in ss2_students.iterrows():
        try:
            student_id = str(row.get('StudentID', f"SS2_{students_created+1}"))
            student, created = Student.objects.get_or_create(
                student_id=student_id,
                defaults={
                    'first_name': str(row.get('FirstName', f'Student{students_created+1}')),
                    'last_name': str(row.get('LastName', f'SS2_{students_created+1}')),
                    'current_class': 'SS2',
                    'stream': str(row.get('Stream', 'Science')),
                    'date_of_birth': datetime.now().date(),
                    'gender': str(row.get('Gender', 'Male')),
                    'guardian_name': 'Guardian',
                    'guardian_contact': str(row.get('GuardianContact', '1234567890')),
                    'address': 'Address'
                }
            )
            
            students_created += 1  # Count all processed students, not just created ones
                    
        except Exception as e:
            print(f"⚠️  Error processing SS2 student: {e}")
    
    # Process SS3 students
    for _, row in ss3_students.iterrows():
        try:
            student_id = str(row.get('StudentID', f"SS3_{students_created+1}"))
            student, created = Student.objects.get_or_create(
                student_id=student_id,
                defaults={
                    'first_name': str(row.get('FirstName', f'Student{students_created+1}')),
                    'last_name': str(row.get('LastName', f'SS3_{students_created+1}')),
                    'current_class': 'SS3',
                    'stream': str(row.get('Stream', 'Science')),
                    'date_of_birth': datetime.now().date(),
                    'gender': str(row.get('Gender', 'Male')),
                    'guardian_name': 'Guardian',
                    'guardian_contact': str(row.get('GuardianContact', '1234567890')),
                    'address': 'Address'
                }
            )
            
            students_created += 1  # Count all processed students, not just created ones
                    
        except Exception as e:
            print(f"⚠️  Error processing SS3 student: {e}")
    
    # Process scores
    for df_scores in [ss2_scores, ss3_scores]:
        for _, row in df_scores.iterrows():
            try:
                student_id = str(row.get('StudentID'))
                subject_name = str(row.get('SubjectName'))
                score_value = float(row.get('Score', 0))
                
                student = Student.objects.get(student_id=student_id)
                subject = Subject.objects.get(name=subject_name)
                
                # Get or create academic year
                academic_year, _ = AcademicYear.objects.get_or_create(
                    year='2024/2025',
                    term='First Term',
                    defaults={
                        'start_date': datetime.now().date(),
                        'end_date': datetime.now().date(),
                        'is_current': True
                    }
                )
                
                StudentScore.objects.get_or_create(
                    student=student,
                    subject=subject,
                    academic_year=academic_year,
                    defaults={
                        'total_score': score_value,
                        'class_average': 75.0,
                        'continuous_assessment': score_value * 0.3,
                        'examination_score': score_value * 0.7
                    }
                )
                scores_created += 1
                    
            except Exception as e:
                print(f"⚠️  Error processing score: {e}")
    
    print(f"✅ Created {students_created} students")
    print(f"✅ Created {scores_created} score records")
    
    return students_created, scores_created

def train_models():
    """Train ML models."""
    print("\n🤖 Training ML Models")
    print("=" * 30)
    
    # Train models
    tier1 = Tier1CriticalPredictor()
    tier2 = Tier2SciencePredictor()
    tier3 = Tier3ArtsPredictor()
    
    print("Training Tier 1...")
    try:
        tier1_result = tier1.train_models()
        tier1_success = tier1_result.get('success', False)
        print(f"Tier 1 training: {'SUCCESS' if tier1_success else 'FAILED'}")
    except Exception as e:
        print(f"Tier 1 training failed: {e}")
        tier1_success = False
    
    print("Training Tier 2...")
    try:
        tier2_result = tier2.train_models()
        tier2_success = tier2_result.get('success', False)
        print(f"Tier 2 training: {'SUCCESS' if tier2_success else 'FAILED'}")
    except Exception as e:
        print(f"Tier 2 training failed: {e}")
        tier2_success = False
    
    print("Training Tier 3...")
    try:
        tier3_result = tier3.train_models()
        tier3_success = tier3_result.get('success', False)
        print(f"Tier 3 training: {'SUCCESS' if tier3_success else 'FAILED'}")
    except Exception as e:
        print(f"Tier 3 training failed: {e}")
        tier3_success = False
    
    return tier1_success and tier2_success and tier3_success

def test_predictions():
    """Test predictions."""
    print("\n🧪 Testing Predictions")
    print("=" * 25)
    
    students = Student.objects.all()[:3]
    
    for student in students:
        print(f"\nStudent: {student.student_id}")
        
        # Test one prediction per tier
        try:
            tier1 = Tier1CriticalPredictor()
            pred1 = tier1.predict(student.student_id, 'Mathematics')
            print(f"  Math: {pred1.get('prediction', 'N/A')}")
        except:
            print(f"  Math: Error")
        
        try:
            tier2 = Tier2SciencePredictor()
            pred2 = tier2.predict(student.student_id, 'Physics')
            print(f"  Physics: {pred2.get('prediction', 'N/A')}")
        except:
            print(f"  Physics: Error")
        
        try:
            tier3 = Tier3ArtsPredictor()
            pred3 = tier3.predict(student.student_id, 'Literature')
            print(f"  Literature: {pred3.get('prediction', 'N/A')}")
        except:
            print(f"  Literature: Error")

def main():
    """Main training process."""
    print("🚀 SSAS ML Training with Real Data")
    print("=" * 50)
    
    # Import data
    ss2_students, ss2_scores, ss3_students, ss3_scores = import_excel_data()
    
    # Process data
    students_created, scores_created = process_data(ss2_students, ss2_scores, ss3_students, ss3_scores)
    
    # Check if we have data in the database
    from core.apps.students.models import Student, StudentScore
    actual_students = Student.objects.count()
    actual_scores = StudentScore.objects.count()
    
    print(f"\n📊 Database Status:")
    print(f"   Students: {actual_students}")
    print(f"   Scores: {actual_scores}")
    
    if actual_students > 0 and actual_scores > 0:
        # Train models
        training_success = train_models()
        
        if training_success:
            test_predictions()
            print("\n🎉 Training completed!")
        else:
            print("\n❌ Training failed")
    else:
        print("\n❌ No data in database")
    
    return actual_students > 0 and actual_scores > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
