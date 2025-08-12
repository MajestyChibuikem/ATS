import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_teacher_data():
    """Generate realistic teacher data with Nigerian names and qualifications."""
    teacher_names = [
        'Mr. Adebayo Johnson', 'Mrs. Chinwe Okafor', 'Dr. Fatima Abubakar',
        'Mr. Emmanuel Eze', 'Mrs. Folake Adeyemi', 'Mr. Chidi Okonkwo',
        'Mrs. Ngozi Uzoma', 'Mr. Yusuf Ibrahim', 'Mrs. Bukola Olawale',
        'Dr. Olumide Bankole', 'Mrs. Aisha Mohammed', 'Mr. Obinna Nwachukwu',
        'Mrs. Adunni Ogundimu', 'Mr. Kelechi Okwu', 'Mrs. Hauwa Garba',
        'Mr. Tunde Adeleke', 'Mrs. Chidinma Anyanwu', 'Dr. Rasheed Lawal',
        'Mrs. Blessing Etim', 'Mr. Segun Oyeleke', 'Mrs. Funmilayo Adebola',
        'Mr. Emeka Nwosu', 'Mrs. Zainab Yusuf', 'Dr. Godwin Okechukwu'
    ]
    
    specializations = [
        'Mathematics', 'Mathematics', 'Sciences', 'Sciences', 'Languages', 
        'Languages', 'Arts', 'Arts', 'General', 'General', 'Mathematics',
        'Sciences', 'Languages', 'Arts', 'General', 'Mathematics', 'Sciences',
        'Languages', 'Arts', 'General', 'Sciences', 'Languages', 'Arts', 'General'
    ]
    
    teachers = []
    for i, (name, specialization) in enumerate(zip(teacher_names, specializations)):
        teachers.append({
            'TeacherID': f'TCH{str(i+1).zfill(3)}',
            'Name': name,
            'YearsExperience': random.randint(3, 25),
            'QualificationLevel': random.choices(
                ['B.Ed', 'M.Ed', 'PhD', 'B.Sc + PGDE', 'HND + PGDE'],
                weights=[3, 4, 1, 5, 2]
            )[0],
            'Specialization': specialization,
            'TeachingLoad': random.randint(18, 25),  # hours per week
            'PerformanceRating': random.uniform(3.2, 4.8),
            'YearsAtSchool': random.randint(1, min(15, max(1, random.randint(3, 25))))
        })
    return teachers

def assign_teachers_to_subjects(teachers, subjects_list, class_level):
    """Assign teachers to subjects based on specialization and workload."""
    
    # Define how many teachers each subject typically needs
    teacher_requirements = {
        'Mathematics': 2,  # Usually needs 2 teachers for different sections
        'English Language': 2,
        'Physics': 1,
        'Chemistry': 1,
        'Biology': 1,
        'Further Mathematics': 1,
        'Government': 1,
        'Economics': 1,
        'History': 1,
        'Literature': 1,
        'Geography': 1,
        'Civic Education': 1,
        'Igbo Language': 1,
        'Oral English': 1,
        'Agricultural Science': 1,
        'Christian Religious Studies': 1
    }
    
    subject_teacher_assignments = {}
    available_teachers = [t.copy() for t in teachers]  # Create copies to track workload
    
    # Sort subjects by difficulty to assign best teachers first
    high_priority_subjects = ['Mathematics', 'English Language', 'Physics', 'Chemistry', 'Further Mathematics']
    other_subjects = [s for s in subjects_list if s not in high_priority_subjects]
    ordered_subjects = high_priority_subjects + other_subjects
    
    for subject in ordered_subjects:
        if subject not in subjects_list:
            continue
            
        num_teachers_needed = teacher_requirements.get(subject, 1)
        
        # Filter teachers by specialization preference
        preferred_teachers = []
        
        if subject in ['Mathematics', 'Further Mathematics']:
            preferred_teachers = [t for t in available_teachers 
                                if 'Mathematics' in t['Specialization'] or t['Specialization'] == 'Sciences']
        elif subject in ['Physics', 'Chemistry', 'Biology', 'Agricultural Science']:
            preferred_teachers = [t for t in available_teachers 
                                if t['Specialization'] in ['Sciences', 'General']]
        elif subject in ['English Language', 'Oral English', 'Literature']:
            preferred_teachers = [t for t in available_teachers 
                                if t['Specialization'] in ['Languages', 'General']]
        elif subject in ['Government', 'Economics', 'History', 'Geography', 'Civic Education', 'Christian Religious Studies']:
            preferred_teachers = [t for t in available_teachers 
                                if t['Specialization'] in ['Arts', 'General']]
        elif subject == 'Igbo Language':
            preferred_teachers = [t for t in available_teachers 
                                if t['Specialization'] in ['Languages', 'General', 'Arts']]
        
        # Sort by performance rating and experience
        preferred_teachers.sort(key=lambda x: (x['PerformanceRating'], x['YearsExperience']), reverse=True)
        
        # Fallback to any available teachers if not enough specialists
        if len(preferred_teachers) < num_teachers_needed:
            remaining_teachers = [t for t in available_teachers if t not in preferred_teachers]
            remaining_teachers.sort(key=lambda x: (x['PerformanceRating'], x['YearsExperience']), reverse=True)
            preferred_teachers.extend(remaining_teachers)
        
        # Select teachers based on workload capacity
        selected_teachers = []
        for teacher in preferred_teachers:
            if len(selected_teachers) >= num_teachers_needed:
                break
            if teacher['TeachingLoad'] >= 4:  # Minimum hours needed per subject
                selected_teachers.append(teacher.copy())
                teacher['TeachingLoad'] -= random.randint(4, 7)  # Reduce available hours
        
        subject_teacher_assignments[subject] = selected_teachers
        
        # Remove overloaded teachers from pool
        available_teachers = [t for t in available_teachers if t['TeachingLoad'] > 0]
    
    return subject_teacher_assignments

def generate_teacher_performance_data(subject_teacher_assignments, academic_year):
    """Generate performance metrics for teachers by subject."""
    teacher_performance = []
    
    for subject, teachers_list in subject_teacher_assignments.items():
        for teacher in teachers_list:
            # Base performance influenced by teacher qualities
            base_performance = 50 + (teacher['YearsExperience'] * 0.8) + (teacher['PerformanceRating'] * 8)
            
            # Add qualification bonus
            qual_bonus = {'PhD': 8, 'M.Ed': 5, 'B.Ed': 3, 'B.Sc + PGDE': 4, 'HND + PGDE': 2}
            base_performance += qual_bonus.get(teacher['QualificationLevel'], 0)
            
            # Add some randomness
            avg_class_performance = max(45, min(95, base_performance + random.uniform(-8, 8)))
            
            teacher_performance.append({
                'TeacherID': teacher['TeacherID'],
                'TeacherName': teacher['Name'],
                'SubjectName': subject,
                'AcademicYear': academic_year,
                'AverageClassScore': round(avg_class_performance, 2),
                'NumberOfStudents': random.randint(28, 45),
                'PassRate': round(min(100, max(40, avg_class_performance + random.uniform(-10, 10))), 2),
                'StudentSatisfactionRating': round(teacher['PerformanceRating'] + random.uniform(-0.5, 0.3), 1),
                'ProfessionalDevelopmentHours': random.randint(15, 50),
                'ClassAttendanceRate': round(random.uniform(0.88, 0.98), 3)
            })
    
    return teacher_performance

def generate_cohort_data(num_students=2000):
    """Generate comprehensive student data including teacher assignments."""
    
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
        'Obinna', 'Fatima', 'Oluwadamilola', 'Chiamaka', 'Babatunde', 'Zainab',
        'Kelechi', 'Hauwa', 'Emeka', 'Blessing', 'Tunde', 'Funmilayo', 'Segun',
        'Chidinma', 'Rasheed', 'Adunni', 'Godwin', 'Zainab'
    ]
    
    last_names = [
        'Okonkwo', 'Adeyemi', 'Okafor', 'Mohammed', 'Olawale', 'Eze', 'Adebisi',
        'Uzoma', 'Ibrahim', 'Oludare', 'Abubakar', 'Ogunleye', 'Nnamdi', 'Afolabi',
        'Oluwaseyi', 'Ademola', 'Okoro', 'Bankole', 'Ogunbiyi', 'Nwachukwu',
        'Johnson', 'Garba', 'Adeleke', 'Anyanwu', 'Lawal', 'Etim', 'Oyeleke'
    ]

    # Generate teachers first
    print("Generating teacher data...")
    teachers = generate_teacher_data()
    df_teachers = pd.DataFrame(teachers)

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

    # Store teacher assignments and performance for both years
    all_teacher_assignments = {}
    all_teacher_performance = []

    for class_level, year_info in academic_years.items():
        print(f"\nGenerating data for {class_level} ({year_info['year']})...")
        
        # Create year-specific student data
        year_student_data = student_data.copy()
        year_student_data['CurrentClass'] = class_level
        year_student_data['AcademicYear'] = year_info['year']
        year_student_data['EnrollmentDate'] = year_info['start_date']
        
        df_students = pd.DataFrame(year_student_data)

        # Get all subjects for this year
        all_subjects = common_subjects.copy()
        all_subjects.extend(science_subjects)
        all_subjects.extend(arts_subjects)
        
        # Assign teachers to subjects
        subject_teacher_assignments = assign_teachers_to_subjects(teachers, all_subjects, class_level)
        all_teacher_assignments[class_level] = subject_teacher_assignments
        
        # Generate teacher performance data
        teacher_performance = generate_teacher_performance_data(subject_teacher_assignments, year_info['year'])
        all_teacher_performance.extend(teacher_performance)

        # Generate Subject Scores with teacher assignments
        assessment_types = ['Quiz', 'Assignment', 'Project', 'Examination']

        # Store base scores for each student-subject combination if it's SS2
        if class_level == 'SS2':
            student_subject_bases = {}
            student_teacher_assignments = {}  # Track which teacher teaches which student
            
            for student_id, stream in zip(df_students['StudentID'], df_students['Stream']):
                # Generate general student ability
                student_ability = np.random.normal(20, 5)
                
                student_subjects = common_subjects.copy()
                if stream == 'Science':
                    student_subjects.extend(science_subjects)
                else:  # Arts
                    student_subjects.extend(arts_subjects)
                
                student_subject_bases[student_id] = {}
                student_teacher_assignments[student_id] = {}
                
                for subject in student_subjects:
                    # Assign student to a teacher for this subject
                    available_teachers = subject_teacher_assignments[subject]
                    assigned_teacher = random.choice(available_teachers)
                    student_teacher_assignments[student_id][subject] = assigned_teacher
                    
                    # Generate base score with teacher effect
                    base_score = generate_subject_score(subject_parameters[subject], student_ability)
                    
                    # Apply teacher quality effect
                    teacher_quality_factor = (assigned_teacher['YearsExperience'] / 25) * 0.08
                    qualification_bonus = {'PhD': 0.06, 'M.Ed': 0.04, 'B.Ed': 0.02, 'B.Sc + PGDE': 0.03, 'HND + PGDE': 0.015}
                    teacher_effect = teacher_quality_factor + qualification_bonus.get(assigned_teacher['QualificationLevel'], 0)
                    performance_effect = (assigned_teacher['PerformanceRating'] - 3.5) * 0.05
                    
                    adjusted_score = base_score + (teacher_effect * 15) + (performance_effect * 10)
                    student_subject_bases[student_id][subject] = max(min(adjusted_score, 100), 0)
        
        score_data = []
        for idx, student in df_students.iterrows():
            student_id = student['StudentID']
            student_subjects = common_subjects.copy()
            
            if student['Stream'] == 'Science':
                student_subjects.extend(science_subjects)
            else:  # Arts
                student_subjects.extend(arts_subjects)

            for subject in student_subjects:
                # Get teacher assignment
                if class_level == 'SS2':
                    assigned_teacher = student_teacher_assignments[student_id][subject]
                else:  # SS3 - maintain same teacher assignments
                    assigned_teacher = student_teacher_assignments[student_id][subject]
                
                # Calculate progression for SS3
                if class_level == 'SS3':
                    # Get previous score and subject parameters
                    prev_score = student_subject_bases[student_id][subject]
                    subject_params = subject_parameters[subject]
                    
                    # Calculate potential for improvement (with continued teacher effect)
                    improvement_potential = (100 - prev_score) * 0.35
                    
                    # Higher difficulty subjects show more improvement with mastery
                    mastery_factor = subject_params['difficulty'] * 0.8
                    
                    # Calculate improvement based on starting point
                    if prev_score < 50:
                        improvement = random.uniform(6, 15) * mastery_factor
                    elif prev_score < 70:
                        improvement = random.uniform(4, 10) * mastery_factor
                    elif prev_score < 85:
                        improvement = random.uniform(2, 6) * mastery_factor
                    else:
                        improvement = random.uniform(0, 4) * mastery_factor
                    
                    # Add continued teacher effect
                    teacher_continuity_bonus = 0.02 * assigned_teacher['YearsExperience']
                    
                    # Add subject-specific variation
                    subject_variation = random.gauss(0, subject_params['std'] * 0.12)
                    
                    # Calculate final SS3 base score
                    base_score = min(100, prev_score + (improvement * improvement_potential / 100) + 
                                   subject_variation + teacher_continuity_bonus)
                else:
                    base_score = student_subject_bases[student_id][subject]

                for assessment in assessment_types:
                    for term in [1, 2, 3]:
                        # Add assessment-specific variations
                        assessment_variation = {
                            'Quiz': random.uniform(-5, 5),
                            'Assignment': random.uniform(-3, 7),
                            'Project': random.uniform(-4, 6),
                            'Examination': random.uniform(-8, 8)
                        }
                        
                        variation = assessment_variation[assessment]
                        final_score = min(max(base_score + variation, 0), 100)
                        
                        score_data.append({
                            'StudentID': student_id,
                            'SubjectName': subject,
                            'TeacherID': assigned_teacher['TeacherID'],
                            'TeacherName': assigned_teacher['Name'],
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
                                'Requires additional support', 'Consistent performance',
                                'Significant improvement noted', 'Maintain current effort'
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
                        'Religious holiday', 'Transport issues', 'Weather conditions'
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
                'Submitted all assignments on time',
                'Demonstrated excellent problem-solving skills',
                'Participated in academic competition'
            ],
            'Discipline': [
                'Exemplary behavior',
                'Followed all school rules',
                'Respectful to teachers and peers',
                'Maintains proper uniform',
                'Demonstrates good conduct',
                'Shows leadership in maintaining order',
                'Positive influence on classmates'
            ],
            'Extra-curricular': [
                'Participated in science fair',
                'Led sports team',
                'Organized school event',
                'Active in debate club',
                'Volunteers for community service',
                'Member of drama club',
                'Participated in inter-school competition'
            ],
            'Leadership': [
                'Class representative',
                'Mentored junior students',
                'Led group project successfully',
                'Organized study groups',
                'Assists teachers with class activities',
                'School prefect duties',
                'Led school committee'
            ]
        }

        for student_id in df_students['StudentID']:
            num_records = random.randint(6, 10) if class_level == 'SS2' else random.randint(8, 12)
            
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
                        'Certificate of achievement',
                        'Recognition in assembly',
                        'Leadership opportunity provided'
                    ]),
                    'RecordedBy': random.choice([t['Name'] for t in teachers])
                })
        df_behavioral = pd.DataFrame(behavior_data)

        # Store dataframes for this academic year
        all_data[class_level] = {
            'students': df_students,
            'scores': df_scores,
            'attendance': df_attendance,
            'behavioral': df_behavioral
        }

    # Create comprehensive teacher performance dataframe
    df_teacher_performance = pd.DataFrame(all_teacher_performance)

    # Save to Excel files
    for class_level, dfs in all_data.items():
        filename = f'student_records_{class_level}.xlsx'
        print(f"\nSaving {class_level} data to {filename}...")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            dfs['students'].to_excel(writer, sheet_name='Students', index=False)
            dfs['scores'].to_excel(writer, sheet_name='SubjectScores', index=False)
            dfs['attendance'].to_excel(writer, sheet_name='Attendance', index=False)
            dfs['behavioral'].to_excel(writer, sheet_name='BehavioralRecords', index=False)
            
            # Add teacher-related sheets
            if class_level == 'SS2':  # Only add once
                df_teachers.to_excel(writer, sheet_name='Teachers', index=False)
            
            # Filter teacher performance for this year
            year = academic_years[class_level]['year']
            year_teacher_performance = df_teacher_performance[df_teacher_performance['AcademicYear'] == year]
            year_teacher_performance.to_excel(writer, sheet_name='TeacherPerformance', index=False)

    print("\nData generation complete!")
    print(f"Generated data for {num_students} students across 2 academic years")
    print(f"Total teachers: {len(teachers)}")
    print(f"Teacher performance records: {len(df_teacher_performance)}")
    
    return all_data, teachers, df_teacher_performance

if __name__ == "__main__":
    data, teachers, teacher_performance = generate_cohort_data(2000)  # Generate data for 2000 students