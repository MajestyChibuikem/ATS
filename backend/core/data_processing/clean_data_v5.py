import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc

def process_student_data(scores_file, student_file, output_file):
    # Define efficient data types
    dtype_mapping = {
        'StudentID': 'str',  
        'Score': 'float32'
    }
    
    # Load data efficiently
    df_scores = pd.read_excel(scores_file, sheet_name='SubjectScores', dtype=dtype_mapping, usecols=['StudentID', 'Score', 'SubjectName'])
    df_students = pd.read_excel(student_file, sheet_name='Students', dtype={'StudentID': 'str'})
    
    # Merge data efficiently
    merged_data = df_scores.merge(df_students, on='StudentID', how='left', copy=False)
    
    # Free memory
    del df_scores, df_students
    gc.collect()
    
    # Handle missing values
    numerical_columns = merged_data.select_dtypes(include=['float32', 'int32']).columns
    merged_data[numerical_columns] = merged_data[numerical_columns].fillna(merged_data[numerical_columns].mean())
    
    categorical_columns = merged_data.select_dtypes(include=['category', 'object']).columns
    for col in categorical_columns:
        if merged_data[col].nunique() > 1:
            merged_data[col].fillna(merged_data[col].mode()[0], inplace=True)
    
    # Normalize numerical data
    scaler = StandardScaler()
    merged_data[numerical_columns] = scaler.fit_transform(merged_data[numerical_columns])
    
    # Save processed data
    merged_data.to_csv(output_file, index=False)
    print(f'Processed data saved to {output_file}')
    
# Example Usage
if __name__ == "__main__":
    process_student_data('../../excel_files/student_records_SS2.xlsx', '../../excel_files/student_records_SS2.xlsx', '../../excel_files/cleaned_student_records_v5.csv')