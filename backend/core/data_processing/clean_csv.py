import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# File paths
file_ss2 = '../../excel_files/student_records_SS2.xlsx'
file_ss3 = '../../excel_files/student_records_SS3.xlsx'

# Read all sheets
dss2 = pd.read_excel(file_ss2, sheet_name=None)

cleaned_sheets = {}

# Define custom processing functions
def process_subject_scores(data):
    """Normalize the 'Score' column and one-hot encode categorical columns."""
    numerical_columns = ["Score"]
    categorical_columns = ["SubjectName", "AssessmentType", "Term", "Class", "Stream"]

    # Normalize numerical columns
    if any(col in data.columns for col in numerical_columns):
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Encode categorical columns
    if any(col in data.columns for col in categorical_columns):
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_data = encoder.fit_transform(data[categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
        data = pd.concat([data, encoded_df], axis=1).drop(categorical_columns, axis=1)

    return data

def process_attendance(data):
    """Convert dates to datetime format and fill missing status values."""
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors='coerce')

    if "Status" in data.columns:
        data["Status"].fillna("Absent", inplace=True)

    return data

def process_behavioral_records(data):
    """Clean text columns and categorize records."""
    if "Category" in data.columns:
        data["Category"] = data["Category"].str.strip().str.lower()

    if "Description" in data.columns:
        data["Description"] = data["Description"].fillna("No description")

    return data

# Apply custom processing based on sheet names
for sheet_name, data in dss2.items():
    print(f"Processing sheet: {sheet_name}...")

    if sheet_name.lower() == "subjectscores":
        cleaned_sheets[sheet_name] = process_subject_scores(data)

    elif sheet_name.lower() == "attendance":
        cleaned_sheets[sheet_name] = process_attendance(data)

    elif sheet_name.lower() == "behavioralrecords":
        cleaned_sheets[sheet_name] = process_behavioral_records(data)

    else:
        print(f"Skipping unknown sheet: {sheet_name}")

# Save cleaned data to a new Excel file
cleaned_sheets_path = '../../excel_files/cleaned_sheets.xlsx'
with pd.ExcelWriter(cleaned_sheets_path) as writer:
    for sheet_name, cleaned_data in cleaned_sheets.items():
        cleaned_data.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Data cleaning completed. Cleaned file saved as '{cleaned_sheets_path}'.")
