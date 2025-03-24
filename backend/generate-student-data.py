import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_cohort_data(num_students=2000):
    # Helper functions
    def generate_student_id(index):
        return f'STD{str(index+1).zfill(4)}'
    
    def generate_date_of_birth():
        start_date = datetime(2005, 1, 1)
        end_date = datetime(2007, 12, 31)
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        return start_date + timedelta(days=random_days)
    
    def generate_phone():
        return f'080{random.randint(10000000, 99999999)}'

    # Define subject groups
    common_subjects = ['English Language', 'Oral English', 'Igbo Language', 
                      'Mathematics', 'Civic Education']
    
    science_subjects = ['Physics', 'Chemistry', 'Biology', 'Agricultural Science', 
                       'Further Mathematics']
    
    arts_subjects = ['Government', 'Economics', 'History', 'Literature', 
                    'Geography', 'Christian Religious Studies']

    # Nigerian names
    first_names = [
        'Oluwaseun', 'Chioma', 'Adebayo', 'Ngozi', 'Emmanuel', 'Aisha', 'Chidi',
        'Olayinka', 'Bukola', 'Yusuf', 'Chinua', 'Folake', 'Ibrahim', 'Amina',
        'Obinna', 'Fatima', 'Oluwadamilola', 'Chiamaka', 'Babatunde', 'Zainab'
    ]
    
    last_names = [
        'Okonkwo', 'Adeyemi', 'Okafor', 'Mohammed', 'Olawale', 'Eze', 'Adebisi',
        'Uzoma', 'Ibrahim', 'Oludare', 'Abubakar', 'Ogunleye', 'Nnamdi', 'Afolabi',
        'Oluwaseyi', 'Ademola', 'Okoro', 'Bankole', 'Ogunbiyi', 'Nwachukwu'
    ]

    # Generate base student data
    print("Generating base student data...")
    student_data = {
        'StudentID': [generate_student_id(i) for i in range(num_students)],
        'FirstName': [random.choice(first_names) for _ in range(num_students)],
        'LastName': [random.choice(last_names) for _ in range(num_students)],
        'DateOfBirth': [generate_date_of_birth() for _ in range(num_students)],
        'Gender': [random.choice(['Male', 'Female']) for _ in range(num_students)],
        'GuardianContact': [generate_phone() for _ in range(num_students)],
        'Stream': random.choices(['Science', 'Arts'], weights=[3, 2], k=num_students)
    }

    # Academic years setup
    academic_years = {
        'SS2': {'year': '2022/2023', 'start_date': datetime(2022, 9, 1)},
        'SS3': {'year': '2023/2024', 'start_date': datetime(2023, 9, 1)}
    }

    # Define subject difficulty and distribution parameters
    subject_parameters = {
        # Common Subjects
        'English Language': {'difficulty': 0.7, 'mean': 72, 'std': 12},
        'Oral English': {'difficulty': 0.65, 'mean': 75, 'std': 10},
        'Igbo Language': {'difficulty': 0.6, 'mean': 78, 'std': 11},
        'Mathematics': {'difficulty': 0.8, 'mean': 68, 'std': 15},
        'Civic Education': {'difficulty': 0.5, 'mean': 80, 'std': 8},
        
        # Science Subjects
        'Physics': {'difficulty': 0.85, 'mean': 65, 'std': 14},
        'Chemistry': {'difficulty': 0.82, 'mean': 67, 'std': 13},
        'Biology': {'difficulty': 0.75, 'mean': 70, 'std': 12},
        'Agricultural Science': {'difficulty': 0.65, 'mean': 75, 'std': 10},
        'Further Mathematics': {'difficulty': 0.9, 'mean': 62, 'std': 16},
        
        # Arts Subjects
        'Government': {'difficulty': 0.7, 'mean': 73, 'std': 11},
        'Economics': {'difficulty': 0.75, 'mean': 71, 'std': 12},
        'History': {'difficulty': 0.7, 'mean': 74, 'std': 10},
        'Literature': {'difficulty': 0.75, 'mean': 72, 'std': 11},
        'Geography': {'difficulty': 0.72, 'mean': 73, 'std': 11},
        'Christian Religious Studies': {'difficulty': 0.65, 'mean': 77, 'std': 9}
    }

    all_data = {}
    
    def generate_subject_score(subject_params, student_ability):
        base = np.random.normal(subject_params['mean'], subject_params['std'])
        ability_factor = student_ability * (1 - subject_params['difficulty'])
        score = base + ability_factor
        return max(min(score, 100), 0)

    for class_level, year_info in academic_years.items():
        print(f"\nGenerating data for {class_level} ({year_info['year']})...")
        
        # Create year-specific student data
        year_student_data = student_data.copy()
        year_student_data['CurrentClass'] = class_level
        year_student_data['AcademicYear'] = year_info['year']
        year_student_data['EnrollmentDate'] = year_info['start_date']
        
        df_students = pd.DataFrame(year_student_data)

        # Generate Subject Scores
        assessment_types = ['Quiz', 'Assignment', 'Project', 'Examination']

        # Store base scores for each student-subject combination if it's SS2
        if class_level == 'SS2':
            student_subject_bases = {}
            for student_id, stream in zip(df_students['StudentID'], df_students['Stream']):
                # Generate general student ability
                student_ability = np.random.normal(20, 5)
                
                student_subjects = common_subjects.copy()
                if stream == 'Science':
                    student_subjects.extend(science_subjects)
                else:  # Arts
                    student_subjects.extend(arts_subjects)
                
                student_subject_bases[student_id] = {
                    subject: generate_subject_score(subject_parameters[subject], student_ability)
                    for subject in student_subjects
                }
        
        score_data = []
        for idx, student in df_students.iterrows():
            student_id = student['StudentID']
            student_subjects = common_subjects.copy()
            
            if student['Stream'] == 'Science':
                student_subjects.extend(science_subjects)
            else:  # Arts
                student_subjects.extend(arts_subjects)

            for subject in student_subjects:
                # Calculate progression for SS3
                if class_level == 'SS3':
                    # Get previous score and subject parameters
                    prev_score = student_subject_bases[student_id][subject]
                    subject_params = subject_parameters[subject]
                    
                    # Calculate potential for improvement
                    improvement_potential = (100 - prev_score) * 0.3
                    
                    # Higher difficulty subjects show more improvement with mastery
                    mastery_factor = subject_params['difficulty'] * 0.7
                    
                    # Calculate improvement based on starting point
                    if prev_score < 50:
                        improvement = random.uniform(5, 12) * mastery_factor
                    elif prev_score < 70:
                        improvement = random.uniform(3, 8) * mastery_factor
                    elif prev_score < 85:
                        improvement = random.uniform(2, 5) * mastery_factor
                    else:
                        improvement = random.uniform(0, 3) * mastery_factor
                    
                    # Add subject-specific variation
                    subject_variation = random.gauss(0, subject_params['std'] * 0.1)
                    
                    # Calculate final SS3 base score
                    base_score = min(100, prev_score + (improvement * improvement_potential / 100) + subject_variation)
                else:
                    base_score = student_subject_bases[student_id][subject]

                for assessment in assessment_types:
                    for term in [1, 2, 3]:
                        variation = random.uniform(-8, 8)
                        final_score = min(max(base_score + variation, 0), 100)
                        
                        score_data.append({
                            'StudentID': student_id,
                            'SubjectName': subject,
                            'AssessmentType': assessment,
                            'Score': round(final_score, 2),
                            'AcademicYear': year_info['year'],
                            'Term': f'Term {term}',
                            'Class': class_level,
                            'Stream': student['Stream'],
                            'AssessmentDate': year_info['start_date'] + timedelta(days=random.randint(0, 180)),
                            'TeacherRemarks': random.choice([
                                'Excellent performance', 'Good effort', 'Needs improvement',
                                'Outstanding work', 'Average performance', 'Shows promise',
                                'Requires additional support', 'Consistent performance'
                            ])
                        })

        df_scores = pd.DataFrame(score_data)

        # Generate Attendance Data
        attendance_data = []
        for student_id in df_students['StudentID']:
            base_attendance_rate = 0.9 if class_level == 'SS2' else 0.93
            
            for day in range(180):
                if random.random() < base_attendance_rate:
                    status = random.choice(['Present'] * 9 + ['Late'])
                else:
                    status = 'Absent'
                
                attendance_data.append({
                    'StudentID': student_id,
                    'Date': year_info['start_date'] + timedelta(days=day),
                    'Status': status,
                    'Remarks': '' if status == 'Present' else random.choice([
                        'Sick leave', 'Family emergency', 'Medical appointment',
                        'Religious holiday', 'Transport issues'
                    ]) if status == 'Absent' else 'Transport delay'
                })
        df_attendance = pd.DataFrame(attendance_data)

        # Generate Behavioral Records
        behavior_data = []
        behavior_categories = ['Academic', 'Discipline', 'Extra-curricular', 'Leadership']
        behavior_descriptions = {
            'Academic': [
                'Completed extra credit work',
                'Helped peers with studies',
                'Showed improvement in mathematics',
                'Active participation in class',
                'Submitted all assignments on time'
            ],
            'Discipline': [
                'Exemplary behavior',
                'Followed all school rules',
                'Respectful to teachers and peers',
                'Maintains proper uniform',
                'Demonstrates good conduct'
            ],
            'Extra-curricular': [
                'Participated in science fair',
                'Led sports team',
                'Organized school event',
                'Active in debate club',
                'Volunteers for community service'
            ],
            'Leadership': [
                'Class representative',
                'Mentored junior students',
                'Led group project successfully',
                'Organized study groups',
                'Assists teachers with class activities'
            ]
        }

        for student_id in df_students['StudentID']:
            num_records = random.randint(5, 8) if class_level == 'SS2' else random.randint(7, 10)
            
            for _ in range(num_records):
                category = random.choice(behavior_categories)
                behavior_data.append({
                    'StudentID': student_id,
                    'RecordDate': year_info['start_date'] + timedelta(days=random.randint(0, 180)),
                    'Category': category,
                    'Description': random.choice(behavior_descriptions[category]),
                    'ActionTaken': random.choice([
                        'Positive reinforcement',
                        'Verbal commendation',
                        'Merit points awarded',
                        'Parent notification',
                        'Certificate of achievement'
                    ])
                })
        df_behavioral = pd.DataFrame(behavior_data)

        # Store dataframes for this academic year
        all_data[class_level] = {
            'students': df_students,
            'scores': df_scores,
            'attendance': df_attendance,
            'behavioral': df_behavioral
        }

    # Save to Excel files
    for class_level, dfs in all_data.items():
        filename = f'student_records_{class_level}.xlsx'
        print(f"\nSaving {class_level} data to {filename}...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            dfs['students'].to_excel(writer, sheet_name='Students', index=False)
            dfs['scores'].to_excel(writer, sheet_name='SubjectScores', index=False)
            dfs['attendance'].to_excel(writer, sheet_name='Attendance', index=False)
            dfs['behavioral'].to_excel(writer, sheet_name='BehavioralRecords', index=False)

    print("\nData generation complete!")
    
    return all_data

if __name__ == "__main__":
    data = generate_cohort_data(2000)  # Generate data for 2000 students