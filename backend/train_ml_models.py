"""
ML Model Training for SSAS MVP
Trains the 3-tier ML models with realistic student data.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from core.apps.students.models import Student, Subject, AcademicYear, StudentScore, Teacher
from core.apps.ml.models.tier1_critical_predictor import Tier1CriticalPredictor
from core.apps.ml.models.tier2_science_predictor import Tier2SciencePredictor
from core.apps.ml.models.tier3_arts_predictor import Tier3ArtsPredictor

class MLModelTrainer:
    """Train ML models for SSAS MVP."""
    
    def __init__(self):
        self.tier1_predictor = Tier1CriticalPredictor()
        self.tier2_predictor = Tier2SciencePredictor()
        self.tier3_predictor = Tier3ArtsPredictor()
        
        # Subject mappings
        self.critical_subjects = ['Mathematics', 'English Language']
        self.science_subjects = ['Physics', 'Chemistry', 'Biology']
        self.arts_subjects = ['Literature', 'History', 'Geography', 'Economics']
    
    def generate_training_data(self, num_students=1000):
        """Generate realistic training data for MVP."""
        print("📊 Generating Training Data")
        print("=" * 40)
        
        # Create subjects if they don't exist
        self._create_subjects()
        
        # Create teachers
        teachers = self._create_teachers()
        
        # Create students and their performance data
        students = self._create_students(num_students)
        
        # Generate performance data
        self._generate_performance_data(students, teachers)
        
        print(f"✅ Generated data for {num_students} students")
        return students
    
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
        
        print(f"✅ Created {len(teachers)} teachers")
        return teachers
    
    def _create_students(self, num_students):
        """Create students with realistic profiles."""
        students = []
        
        for i in range(num_students):
            # Create student with realistic attributes
            student = Student.objects.create(
                student_id=f"STU{i+1:04d}",
                first_name=f"Student{i+1}",
                last_name=f"Test{i+1}",
                date_of_birth=datetime.now().date() - timedelta(days=365*16),  # 16 years old
                gender=np.random.choice(['M', 'F']),
                class_level=np.random.choice(['SS1', 'SS2', 'SS3']),
                admission_date=datetime.now().date() - timedelta(days=365*2)
            )
            students.append(student)
        
        print(f"✅ Created {len(students)} students")
        return students
    
    def _generate_performance_data(self, students, teachers):
        """Generate realistic performance data."""
        academic_years = [
            AcademicYear.objects.get_or_create(name='2022/2023')[0],
            AcademicYear.objects.get_or_create(name='2023/2024')[0],
            AcademicYear.objects.get_or_create(name='2024/2025')[0]
        ]
        
        terms = ['First Term', 'Second Term', 'Third Term']
        subjects = Subject.objects.all()
        
        scores_created = 0
        
        for student in students:
            # Determine student's academic level (affects performance)
            student_level = np.random.choice(['high', 'medium', 'low'], p=[0.2, 0.6, 0.2])
            
            for academic_year in academic_years:
                for term in terms:
                    for subject in subjects:
                        # Get teacher for this subject
                        teacher = next((t for t in teachers if t.subject == subject), teachers[0])
                        
                        # Generate realistic scores based on student level
                        if student_level == 'high':
                            base_score = np.random.normal(85, 8)
                        elif student_level == 'medium':
                            base_score = np.random.normal(70, 10)
                        else:  # low
                            base_score = np.random.normal(55, 12)
                        
                        # Add some variation based on subject difficulty
                        if subject.name in self.critical_subjects:
                            base_score += np.random.normal(0, 5)
                        elif subject.name in self.science_subjects:
                            base_score += np.random.normal(-2, 6)
                        else:  # arts
                            base_score += np.random.normal(1, 4)
                        
                        # Ensure score is within valid range
                        total_score = max(0, min(100, base_score))
                        class_average = np.random.normal(70, 8)
                        
                        # Determine grade
                        if total_score >= 80:
                            grade = 'A'
                        elif total_score >= 70:
                            grade = 'B'
                        elif total_score >= 60:
                            grade = 'C'
                        elif total_score >= 50:
                            grade = 'D'
                        else:
                            grade = 'F'
                        
                        # Create score record
                        StudentScore.objects.create(
                            student=student,
                            subject=subject,
                            academic_year=academic_year,
                            term=term,
                            total_score=total_score,
                            class_average=class_average,
                            grade=grade,
                            teacher=teacher
                        )
                        scores_created += 1
        
        print(f"✅ Created {scores_created} score records")
    
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
    
    def test_predictions(self):
        """Test predictions with sample data."""
        print("\n🧪 Testing Predictions")
        print("=" * 30)
        
        # Get some test students
        test_students = Student.objects.all()[:5]
        
        for student in test_students:
            print(f"\nTesting predictions for {student.student_id}:")
            
            # Test each tier
            for subject_name in self.critical_subjects[:1]:  # Test one subject per tier
                try:
                    prediction = self.tier1_predictor.predict(student.student_id, subject_name)
                    print(f"  {subject_name}: {prediction.get('prediction', 'N/A')}")
                except Exception as e:
                    print(f"  {subject_name}: Error - {e}")
            
            for subject_name in self.science_subjects[:1]:
                try:
                    prediction = self.tier2_predictor.predict(student.student_id, subject_name)
                    print(f"  {subject_name}: {prediction.get('prediction', 'N/A')}")
                except Exception as e:
                    print(f"  {subject_name}: Error - {e}")
            
            for subject_name in self.arts_subjects[:1]:
                try:
                    prediction = self.tier3_predictor.predict(student.student_id, subject_name)
                    print(f"  {subject_name}: {prediction.get('prediction', 'N/A')}")
                except Exception as e:
                    print(f"  {subject_name}: Error - {e}")

def main():
    """Run ML model training for MVP."""
    print("🚀 SSAS ML Model Training for MVP")
    print("=" * 50)
    
    trainer = MLModelTrainer()
    
    # Generate training data
    students = trainer.generate_training_data(500)  # Start with 500 students for MVP
    
    # Train models
    training_success = trainer.train_models()
    
    if training_success:
        # Test predictions
        trainer.test_predictions()
        print("\n🎉 ML Models trained successfully!")
        print("✅ MVP is ready for testing!")
    else:
        print("\n❌ Training failed. Check logs for details.")
    
    return training_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
