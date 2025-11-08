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
    """Process student data with proper longitudinal tracking."""
    print("\n🔧 Processing Data with Student Progression")
    print("=" * 45)
    
    # Create subjects from the actual data
    all_subjects = set()
    for df in [ss2_scores, ss3_scores]:
        if 'SubjectName' in df.columns:
            all_subjects.update(df['SubjectName'].unique())
    
    subjects = list(all_subjects)
    print(f"Found subjects: {subjects}")
    
    # Create subjects with proper codes
    subject_codes = {
        'Mathematics': 'MATH',
        'English Language': 'ENG',
        'Further Mathematics': 'FUR_MATH',
        'Physics': 'PHY',
        'Chemistry': 'CHEM',
        'Biology': 'BIO',
        'Agricultural Science': 'AGRI_SCI',
        'Government': 'GOV',
        'Economics': 'ECON',
        'History': 'HIST',
        'Literature': 'LIT',
        'Geography': 'GEO',
        'Christian Religious Studies': 'CRS',
        'Civic Education': 'CIVIC',
        'Oral English': 'ORAL_ENG',
        'Igbo Language': 'IGBO'
    }
    
    for subject_name in subjects:
        code = subject_codes.get(subject_name, subject_name[:10].replace(' ', '_').upper())
        Subject.objects.get_or_create(
            name=subject_name,
            defaults={
                'code': code,
                'category': 'Core' if subject_name in ['Mathematics', 'English Language'] else 'Elective',
                'stream': 'All',
                'description': f'{subject_name} subject'
            }
        )
    
    print(f"✅ Created {len(subjects)} subjects")
    
    # Create academic years
    academic_years = {
        'SS2': AcademicYear.objects.get_or_create(
            year='2022/2023',
            term='First Term',
            defaults={
                'start_date': datetime(2022, 9, 1).date(),
                'end_date': datetime(2023, 6, 30).date(),
                'is_current': False
            }
        )[0],
        'SS3': AcademicYear.objects.get_or_create(
            year='2023/2024',
            term='First Term',
            defaults={
                'start_date': datetime(2023, 9, 1).date(),
                'end_date': datetime(2024, 6, 30).date(),
                'is_current': True
            }
        )[0]
    }
    
    print("✅ Created academic years")
    
    # Process students with longitudinal tracking
    students_processed = 0
    scores_created = 0
    
    # First, process all unique students (combine SS2 and SS3)
    all_students_data = {}
    
    # Collect SS2 students
    for _, row in ss2_students.iterrows():
        student_id = str(row.get('StudentID', ''))
        if student_id:
            all_students_data[student_id] = {
                'ss2_data': row,
                'ss3_data': None
            }
    
    # Collect SS3 students and match with SS2
    for _, row in ss3_students.iterrows():
        student_id = str(row.get('StudentID', ''))
        if student_id:
            if student_id in all_students_data:
                # Student exists in SS2, add SS3 data
                all_students_data[student_id]['ss3_data'] = row
            else:
                # New student in SS3 only
                all_students_data[student_id] = {
                    'ss2_data': None,
                    'ss3_data': row
                }
    
    print(f"📊 Found {len(all_students_data)} unique students")
    
    # Process each unique student
    for student_id, data in all_students_data.items():
        try:
            # Determine current class (prioritize SS3 if available)
            if data['ss3_data'] is not None:
                current_class = 'SS3'
                source_data = data['ss3_data']
                print(f"🔄 Processing {student_id}: SS2 → SS3 progression")
            else:
                current_class = 'SS2'
                source_data = data['ss2_data']
                print(f"📚 Processing {student_id}: SS2 only")
            
            # Create or update student
            student, created = Student.objects.get_or_create(
                student_id=student_id,
                defaults={
                    'first_name': str(source_data.get('FirstName', 'Unknown')),
                    'last_name': str(source_data.get('LastName', 'Student')),
                    'current_class': current_class,
                    'stream': str(source_data.get('Stream', 'Science')),
                    'date_of_birth': datetime.now().date(),
                    'gender': str(source_data.get('Gender', 'Male')),
                    'guardian_name': 'Guardian',
                    'guardian_contact': str(source_data.get('GuardianContact', '1234567890')),
                    'address': 'Address'
                }
            )
            
            # If student already exists, update current class
            if not created:
                student.current_class = current_class
                student.save()
                print(f"  ✅ Updated {student_id} to {current_class}")
            
            students_processed += 1
                    
        except Exception as e:
            print(f"⚠️  Error processing student {student_id}: {e}")
    
    # Process scores with proper aggregation per subject per academic year
    print("\n📊 Processing Student Scores by Academic Year (with aggregation)")
    
    # Function to aggregate and create scores
    def aggregate_and_create_scores(scores_df, academic_year_obj, year_name):
        """Aggregate multiple assessments into final subject scores."""
        print(f"Processing {year_name} scores...")
        
        # Group by student and subject
        grouped = scores_df.groupby(['StudentID', 'SubjectName'])
        
        for (student_id, subject_name), group in grouped:
            try:
                student = Student.objects.get(student_id=student_id)
                subject = Subject.objects.get(name=subject_name)
                
                # Calculate final scores from all assessments
                # Get examination scores (final exams are most important)
                exam_scores = group[group['AssessmentType'] == 'Examination']['Score']
                
                # Get continuous assessment scores (Quiz, Assignment, Project)
                ca_scores = group[group['AssessmentType'].isin(['Quiz', 'Assignment', 'Project'])]['Score']
                
                # Calculate final CA and Exam scores
                if len(exam_scores) > 0:
                    final_exam_score = float(exam_scores.mean())
                else:
                    final_exam_score = float(group['Score'].mean())
                
                if len(ca_scores) > 0:
                    final_ca_score = float(ca_scores.mean())
                else:
                    final_ca_score = float(group['Score'].mean())
                
                # Calculate total score (70% exam + 30% CA)
                total_score = (final_exam_score * 0.7) + (final_ca_score * 0.3)
                
                # Calculate class average for this subject
                all_subject_scores = scores_df[scores_df['SubjectName'] == subject_name]
                class_average = float(all_subject_scores['Score'].mean())
                
                # Create or update the student score
                StudentScore.objects.update_or_create(
                    student=student,
                    subject=subject,
                    academic_year=academic_year_obj,
                    defaults={
                        'total_score': total_score,
                        'class_average': class_average,
                        'continuous_assessment': final_ca_score,
                        'examination_score': final_exam_score
                    }
                )
                
            except Exception as e:
                print(f"⚠️  Error processing {student_id} - {subject_name}: {e}")
    
    # Process SS2 scores (2022/2023)
    aggregate_and_create_scores(ss2_scores, academic_years['SS2'], 'SS2 (2022/2023)')
    
    # Process SS3 scores (2023/2024)
    aggregate_and_create_scores(ss3_scores, academic_years['SS3'], 'SS3 (2023/2024)')
    
    # Count final scores created
    final_score_count = StudentScore.objects.count()
    scores_created = final_score_count
    
    print(f"✅ Processed {students_processed} unique students")
    print(f"✅ Created {scores_created} score records")
    
    return students_processed, scores_created

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
        tier1_success = 'ensemble_performance' in tier1_result
        print(f"Tier 1 training: {'SUCCESS' if tier1_success else 'FAILED'}")
        if tier1_success:
            ensemble_r2 = tier1_result['ensemble_performance']['r2']
            print(f"  Ensemble R²: {ensemble_r2:.3f}")
            # Save models to disk
            tier1.save_models()
            print(f"  ✅ Models saved to disk")
    except Exception as e:
        print(f"Tier 1 training failed: {e}")
        tier1_success = False
    
    print("Training Tier 2...")
    try:
        tier2_result = tier2.train_models()
        tier2_success = len(tier2_result) > 0  # Check if results dictionary has subjects
        print(f"Tier 2 training: {'SUCCESS' if tier2_success else 'FAILED'}")
        if tier2_success:
            subjects_trained = len(tier2_result)
            print(f"  Subjects trained: {subjects_trained}")
            # Show some performance metrics
            for subject, result in list(tier2_result.items())[:2]:  # Show first 2 subjects
                r2 = result['performance']['r2']
                print(f"    {subject}: R² = {r2:.3f}")
            # Save models to disk
            tier2.save_models()
            print(f"  ✅ Models saved to disk")
    except Exception as e:
        print(f"Tier 2 training failed: {e}")
        tier2_success = False
    
    print("Training Tier 3...")
    try:
        tier3_result = tier3.train_models()
        tier3_success = len(tier3_result) > 0  # Check if results dictionary has subjects
        print(f"Tier 3 training: {'SUCCESS' if tier3_success else 'FAILED'}")
        if tier3_success:
            subjects_trained = len(tier3_result)
            print(f"  Subjects trained: {subjects_trained}")
            # Show some performance metrics
            for subject, result in list(tier3_result.items())[:2]:  # Show first 2 subjects
                r2 = result['performance']['r2']
                print(f"    {subject}: R² = {r2:.3f}")
            # Save models to disk
            tier3.save_models()
            print(f"  ✅ Models saved to disk")
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
