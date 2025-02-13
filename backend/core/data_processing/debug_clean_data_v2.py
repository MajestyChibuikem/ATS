import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc

file_ss2 = '../../excel_files/student_records_SS2.xlsx'

dtype_mapping = {
    'StudentID': 'str',
    'Score': 'float32',
    'Status': 'category'
}

subject_scores_data = pd.read_excel(file_ss2, sheet_name='SubjectScores', dtype=dtype_mapping)
attendance_data = pd.read_excel(file_ss2, sheet_name='Attendance', dtype=dtype_mapping)
behavioral_records_data = pd.read_excel(file_ss2, sheet_name='BehavioralRecords', dtype=dtype_mapping)

subject_scores_data = subject_scores_data[['StudentID', 'Score', 'SubjectName']]
attendance_data = attendance_data[['StudentID', 'Status']]
behavioral_records_data = behavioral_records_data[['StudentID', 'Description']]

attendance_data = attendance_data.drop_duplicates(subset=['StudentID'])
behavioral_records_data = behavioral_records_data.drop_duplicates(subset=['StudentID'])

# Debug: Check before merging
print("Before Merge - Attendance Unique Status:", attendance_data['Status'].unique())

# Normalize StudentID format
attendance_data['StudentID'] = attendance_data['StudentID'].astype(str).str.strip()
subject_scores_data['StudentID'] = subject_scores_data['StudentID'].astype(str).str.strip()

# Merge data
merged_data = subject_scores_data.merge(attendance_data, on='StudentID', how='left', copy=False)
merged_data = merged_data.merge(behavioral_records_data, on='StudentID', how='left', copy=False)

del subject_scores_data, attendance_data, behavioral_records_data
gc.collect()

# Debug: Check after merge
print("After Merge - Unique Status:", merged_data['Status'].unique())
print(merged_data[['StudentID', 'Status']].head(10))

# Clean and map status
merged_data['Status'] = merged_data['Status'].astype(str).str.strip().str.title()
status_mapping = {'Present': 0, 'Absent': 1, 'Late': 2}
merged_data['Status'] = merged_data['Status'].replace(status_mapping)

# Debug: Check mapping
print("After Mapping - Unique Status:", merged_data['Status'].unique())

numerical_columns = ['Score']
merged_data[numerical_columns] = merged_data[numerical_columns].fillna(merged_data[numerical_columns].mean())

scaler = StandardScaler()
merged_data[numerical_columns] = scaler.fit_transform(merged_data[numerical_columns])

print(merged_data.info())
print(merged_data.head())

merged_data.to_csv('../../excel_files/cleaned_student_records.csv_v7', index=False)
