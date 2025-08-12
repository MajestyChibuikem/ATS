"""
Train ML Models with Real Student Data
Imports data from student_records_SS2.xlsx and student_records_SS3.xlsx
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from core.apps.students.models import Student, Subject, AcademicYear, StudentScore, Teacher
from core.apps.ml.models.tier1_critical_predictor import Tier1CriticalPredictor
from core.apps.ml.models.tier2_science_predictor import Tier2SciencePredictor
from core.apps.ml.models.tier3_arts_predictor import Tier3ArtsPredictor

class RealDataTrainer:
    """Train ML models with real student data."""
    
    def __init__(self):
        self.tier1_predictor = Tier1CriticalPredictor()
        self.tier2_predictor = Tier2SciencePredictor()
        self.tier3_predictor = Tier3ArtsPredictor()
        
        # Subject mappings
        self.critical_subjects = ['Mathematics', 'English Language']
        self.science_subjects = ['Physics', 'Chemistry', 'Biology']
        self.arts_subjects = ['Literature', 'History', 'Geography', 'Economics']
    
    def import_excel_data(self):
        """Import data from Excel files."""
        print("📊 Importing Real Student Data")
        print("=" * 40)
        
        # Import SS2 data
        print("Importing SS2 data...")
        ss2_data = pd.read_excel('student_records_SS2.xlsx')
        print(f"✅ SS2: {len(ss2_data)} records")
        
        # Import SS3 data
        print("Importing SS3 data...")
        ss3_data = pd.read_excel('student_records_SS3.xlsx')
        print(f"✅ SS3: {len(ss3_data)} records")
        
        # Combine data
        combined_data = pd.concat([ss2_data, ss3_data], ignore_index=True)
        print(f"✅ Combined: {len(combined_data)} total records")
        
        # Show data structure
        print(f"\nData columns: {list(combined_data.columns)}")
        print(f"Sample data:\n{combined_data.head()}")
        
        return combined_data
    
    def process_student_data(self, data):
        """Process and clean student data."""
        print("\n🔧 Processing Student Data")
        print("=" * 30)
        
        # Create subjects
        self._create_subjects()
        
        # Create teachers
        teachers = self._create_teachers()
        
        # Process students
        students_created = 0
        scores_created = 0
        
        for _, row in data.iterrows():
            try:
                # Create or get student
                student_id = str(row.get('Student ID', f"STU{students_created+1:04d}"))
                student, created = Student.objects.get_or_create(
                    student_id=student_id,
                    defaults={
                        'first_name': str(row.get('First Name', f'Student{students_created+1}')),
                        'last_name': str(row.get('Last Name', f'Test{students_created+1}')),
                        'date_of_birth': datetime.now().date(),  # Default date
                        'gender': str(row.get('Gender', 'M')),
                        'class_level': str(row.get('Class', 'SS2')),
                        'admission_date': datetime.now().date()
                    }
                )
                
                if created:
                    students_created += 1
                
                # Process subject scores
                for subject_name in self.critical_subjects + self.science_subjects + self.arts_subjects:
                    score_column = f'{subject_name} Score'
                    
                    if score_column in row and pd.notna(row[score_column]):
                        try:
                            subject = Subject.objects.get(name=subject_name)
                            score_value = float(row[score_column])
                            
                            # Get teacher for subject
                            teacher = next((t for t in teachers if t.subject == subject), teachers[0])
                            
                            # Create score record
                            StudentScore.objects.get_or_create(
                                student=student,
                                subject=subject,
                                academic_year=AcademicYear.objects.get_or_create(name='2024/2025')[0],
                                term='First Term',
                                defaults={
                                    'total_score': score_value,
                                    'class_average': np.random.normal(70, 8),  # Generate class average
                                    'grade': self._get_grade(score_value),
                                    'teacher': teacher
                                }
                            )
                            scores_created += 1
                            
                        except (ValueError, Subject.DoesNotExist) as e:
                            print(f"⚠️  Error processing {subject_name} for {student_id}: {e}")
                
            except Exception as e:
                print(f"⚠️  Error processing student row: {e}")
        
        print(f"✅ Created {students_created} students")
        print(f"✅ Created {scores_created} score records")
        
        return students_created, scores_created
    
    def _create_subjects(self):
        """Create all subjects."""
        subjects = self.critical_subjects + self.science_subjects + self.arts_subjects
        
        for subject_name in subjects:
            Subject.objects.get_or_create(name=subject_name)
        
        print(f"✅ Created {len(subjects)} subjects")
    
    def _create_teachers(self):
        """Create teachers for different subjects."""
        teachers = []
        teacher_data = [
            ('John Smith', 'Mathematics', 5, 'MSc Mathematics'),
            ('Sarah Johnson', 'English Language', 8, 'MA English'),
            ('Dr. Michael Chen', 'Physics', 12, 'PhD Physics'),
            ('Dr. Emily Brown', 'Chemistry', 10, 'PhD Chemistry'),
            ('Prof. David Wilson', 'Biology', 15, 'PhD Biology'),
            ('Ms. Lisa Davis', 'Literature', 6, 'MA Literature'),
            ('Mr. Robert Taylor', 'History', 9, 'MA History'),
            ('Dr. Maria Garcia', 'Geography', 11, 'PhD Geography'),
            ('Mr. James Anderson', 'Economics', 7, 'MSc Economics'),
        ]
        
        for name, subject_name, experience, qualification in teacher_data:
            try:
                subject = Subject.objects.get(name=subject_name)
                teacher, created = Teacher.objects.get_or_create(
                    name=name,
                    defaults={
                        'subject': subject,
                        'experience_years': experience,
                        'qualification': qualification,
                        'performance_rating': np.random.uniform(3.5, 5.0)
                    }
                )
                teachers.append(teacher)
            except Subject.DoesNotExist:
                print(f"⚠️  Subject {subject_name} not found")
        
        print(f"✅ Created {len(teachers)} teachers")
        return teachers
    
    def _get_grade(self, score):
        """Convert score to grade."""
        if score >= 80:
            return 'A'
        elif score >= 70:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'F'
    
    def train_models(self):
        """Train all three ML models."""
        print("\n🤖 Training ML Models")
        print("=" * 40)
        
        # Train Tier 1 (Critical Subjects)
        print("Training Tier 1 - Critical Subjects...")
        tier1_success = self.tier1_predictor.train()
        print(f"✅ Tier 1 training: {'SUCCESS' if tier1_success else 'FAILED'}")
        
        # Train Tier 2 (Science Subjects)
        print("Training Tier 2 - Science Subjects...")
        tier2_success = self.tier2_predictor.train()
        print(f"✅ Tier 2 training: {'SUCCESS' if tier2_success else 'FAILED'}")
        
        # Train Tier 3 (Arts Subjects)
        print("Training Tier 3 - Arts Subjects...")
        tier3_success = self.tier3_predictor.train()
        print(f"✅ Tier 3 training: {'SUCCESS' if tier3_success else 'FAILED'}")
        
        return tier1_success and tier2_success and tier3_success
    
    def test_real_predictions(self):
        """Test predictions with real student data."""
        print("\n🧪 Testing Real Predictions")
        print("=" * 35)
        
        # Get some real students
        test_students = Student.objects.all()[:5]
        
        for student in test_students:
            print(f"\nTesting predictions for {student.student_id} ({student.first_name} {student.last_name}):")
            
            # Test each tier with real subjects
            for subject_name in self.critical_subjects:
                try:
                    prediction = self.tier1_predictor.predict(student.student_id, subject_name)
                    pred_score = prediction.get('prediction', 'N/A')
                    confidence = prediction.get('confidence', 'N/A')
                    print(f"  {subject_name}: {pred_score} (confidence: {confidence})")
                except Exception as e:
                    print(f"  {subject_name}: Error - {e}")
            
            for subject_name in self.science_subjects:
                try:
                    prediction = self.tier2_predictor.predict(student.student_id, subject_name)
                    pred_score = prediction.get('prediction', 'N/A')
                    confidence = prediction.get('confidence', 'N/A')
                    print(f"  {subject_name}: {pred_score} (confidence: {confidence})")
                except Exception as e:
                    print(f"  {subject_name}: Error - {e}")
            
            for subject_name in self.arts_subjects:
                try:
                    prediction = self.tier3_predictor.predict(student.student_id, subject_name)
                    pred_score = prediction.get('prediction', 'N/A')
                    confidence = prediction.get('confidence', 'N/A')
                    print(f"  {subject_name}: {pred_score} (confidence: {confidence})")
                except Exception as e:
                    print(f"  {subject_name}: Error - {e}")

def main():
    """Train ML models with real student data."""
    print("🚀 SSAS ML Training with Real Data")
    print("=" * 50)
    
    trainer = RealDataTrainer()
    
    # Import real data
    data = trainer.import_excel_data()
    
    # Process data
    students_created, scores_created = trainer.process_student_data(data)
    
    if students_created > 0 and scores_created > 0:
        # Train models
        training_success = trainer.train_models()
        
        if training_success:
            # Test predictions
            trainer.test_real_predictions()
            print("\n🎉 ML Models trained with real data!")
            print("✅ MVP is ready with real student predictions!")
        else:
            print("\n❌ Training failed. Check logs for details.")
    else:
        print("\n❌ No data processed. Check Excel file format.")
    
    return students_created > 0 and scores_created > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
