import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import gc

# Load sheets efficiently
file_ss2 = '../../excel_files/student_records_SS2.xlsx'

# Specify only necessary columns and use efficient data types
dtype_mapping = {
    'StudentID': 'str',  # Change to string type
    'Score': 'float32',
    'Status': 'category'
}

# Load each sheet selectively with only required columns
subject_scores_data = pd.read_excel(file_ss2, sheet_name='SubjectScores', dtype=dtype_mapping)
attendance_data = pd.read_excel(file_ss2, sheet_name='Attendance', dtype=dtype_mapping)
behavioral_records_data = pd.read_excel(file_ss2, sheet_name='BehavioralRecords', dtype=dtype_mapping)

# Print column names to verify
print("SubjectScores columns:", subject_scores_data.columns)
print("Attendance columns:", attendance_data.columns)
print("BehavioralRecords columns:", behavioral_records_data.columns)

# Drop unnecessary columns before merging
# Use the correct column names from the DataFrame
subject_scores_data = subject_scores_data[['StudentID', 'Score', 'SubjectName']]  # Use 'SubjectName' instead of 'Subject'
attendance_data = attendance_data[['StudentID', 'Status']]
behavioral_records_data = behavioral_records_data[['StudentID', 'Description']]  # Use 'Description' instead of 'BehaviorScore'

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

# Encode categorical data using LabelEncoder for memory efficiency
label_encoder = LabelEncoder()
categorical_columns_to_encode = ['Status']  # Adjust based on actual categorical columns
for col in categorical_columns_to_encode:
    if col in merged_data.columns:
        merged_data[col] = label_encoder.fit_transform(merged_data[col])

# Display final processed dataset
print(merged_data.info())  # Check memory usage
print(merged_data.head())

# Save the cleaned dataset to a new file (optional)
merged_data.to_csv('../../excel_files/cleaned_student_records.csv_v2', index=False)