import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load sheets
file_ss2 = '../../excel_files/student_records_SS2.xlsx'
subject_scores_data = pd.read_excel(file_ss2, sheet_name='SubjectScores')
attendance_data = pd.read_excel(file_ss2, sheet_name='Attendance')
behavioral_records_data = pd.read_excel(file_ss2, sheet_name='BehavioralRecords')

# One-hot encode the 'Status' column in attendance data
attendance_encoded = pd.get_dummies(attendance_data, columns=['Status'], prefix='attendance')

# Merge data
merged_data = pd.merge(subject_scores_data, attendance_encoded, on='StudentID', how='left')
merged_data = pd.merge(merged_data, behavioral_records_data, on='StudentID', how='left')

# Clean and normalize the data
# Step 1: Handle missing values
# Fill numerical columns with their mean (or other appropriate strategy)
numerical_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
merged_data[numerical_columns] = merged_data[numerical_columns].fillna(merged_data[numerical_columns].mean())

# Fill categorical columns with the mode (most frequent value)
categorical_columns = merged_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    merged_data[col].fillna(merged_data[col].mode()[0], inplace=True)

# Step 2: Normalize numerical data
scaler = StandardScaler()
merged_data[numerical_columns] = scaler.fit_transform(merged_data[numerical_columns])

# Step 3: Encode categorical data (e.g., behavioral categories)
# Example: If 'Behavioral_Records_data' has a categorical column like 'Category'
categorical_columns_to_encode = ['Category']  # Replace with actual column names from BehavioralRecords
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
encoded_data = encoder.fit_transform(merged_data[categorical_columns_to_encode])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns_to_encode))

# Combine encoded data with the merged dataset
merged_data = pd.concat([merged_data, encoded_df], axis=1).drop(categorical_columns_to_encode, axis=1)

# Display the cleaned and normalized data
print(merged_data.head())