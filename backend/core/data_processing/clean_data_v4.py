import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc

# Load sheets efficiently
file_ss2 = '../../excel_files/student_records_SS2.xlsx'

# Specify only necessary columns and use efficient data types
dtype_mapping = {
    'StudentID': 'str',  
    'Score': 'float32',
    'Status': 'category'
}

# Load each sheet selectively with only required columns
subject_scores_data = pd.read_excel(file_ss2, sheet_name='SubjectScores', dtype=dtype_mapping)
attendance_data = pd.read_excel(file_ss2, sheet_name='Attendance', dtype=dtype_mapping)
behavioral_records_data = pd.read_excel(file_ss2, sheet_name='BehavioralRecords', dtype=dtype_mapping)

# Print column names for debugging
print("SubjectScores columns:", subject_scores_data.columns)
print("Attendance columns:", attendance_data.columns)
print("BehavioralRecords columns:", behavioral_records_data.columns)

# Drop unnecessary columns before merging
subject_scores_data = subject_scores_data[['StudentID', 'Score', 'SubjectName']]
attendance_data = attendance_data[['StudentID', 'Status']]
behavioral_records_data = behavioral_records_data[['StudentID', 'Description']]

# Ensure StudentID is unique to avoid excessive data merging
attendance_data = attendance_data.drop_duplicates(subset=['StudentID'])
behavioral_records_data = behavioral_records_data.drop_duplicates(subset=['StudentID'])

# Efficiently merge datasets
merged_data = subject_scores_data.merge(attendance_data, on='StudentID', how='left', copy=False)
merged_data = merged_data.merge(behavioral_records_data, on='StudentID', how='left', copy=False)

# Free up memory
del subject_scores_data, attendance_data, behavioral_records_data
gc.collect()

# Handle missing values efficiently
numerical_columns = merged_data.select_dtypes(include=['float32', 'int32']).columns
merged_data[numerical_columns] = merged_data[numerical_columns].fillna(merged_data[numerical_columns].mean())

categorical_columns = merged_data.select_dtypes(include=['category', 'object']).columns
for col in categorical_columns:
    if merged_data[col].nunique() > 1:  # Only fill if there's more than one unique value
        merged_data[col].fillna(merged_data[col].mode()[0], inplace=True)

# Normalize numerical data efficiently
scaler = StandardScaler()
merged_data[numerical_columns] = scaler.fit_transform(merged_data[numerical_columns])

# Clean and map the 'Status' column
print("Unique values in 'Status' before cleaning:", merged_data['Status'].unique())

# Clean the 'Status' column
merged_data['Status'] = merged_data['Status'].str.strip().str.lower()


# Add print statements to verify data at each step
print("Before cleaning - Status value counts:")
print(merged_data['Status'].value_counts(dropna=False))

# Clean the 'Status' column
merged_data['Status'] = merged_data['Status'].str.strip().str.lower()

print("\nAfter strip and lowercase - Status value counts:")
print(merged_data['Status'].value_counts(dropna=False))

# Define status mapping
status_mapping = {
    'absent': 0, 
    'late': 1,    
    'present': 2  
}

# Before applying mapping, validate all values will be mapped
unique_statuses = merged_data['Status'].unique()
unmapped_statuses = [status for status in unique_statuses if status not in status_mapping]
if unmapped_statuses:
    print(f"\nWARNING: Found unmapped status values: {unmapped_statuses}")

# Apply the mapping
merged_data['Status'] = merged_data['Status'].map(status_mapping)

print("\nAfter mapping - Status value counts:")
print(merged_data['Status'].value_counts(dropna=False))

# Verify the mapping worked as expected
print("\nFinal Status distribution:")
status_distribution = merged_data['Status'].value_counts(dropna=False).sort_index()
print("0 (Present):", status_distribution.get(0, 0))
print("1 (Absent):", status_distribution.get(1, 0))
print("2 (Late):", status_distribution.get(2, 0))
merged_data.to_csv('../../excel_files/cleaned_student_records_v5.csv', index=False)