#!/usr/bin/env python3
"""
Check Excel File Structure
"""

import pandas as pd

def check_excel_structure():
    """Check the structure of Excel files."""
    print("📊 Checking Excel File Structure")
    print("=" * 40)
    
    # Check SS3 file
    try:
        ss3_file = pd.ExcelFile('student_records_SS3.xlsx')
        print(f"SS3 Sheets: {ss3_file.sheet_names}")
        
        if 'Teachers' in ss3_file.sheet_names:
            df_teachers = pd.read_excel('student_records_SS3.xlsx', sheet_name='Teachers')
            print(f"SS3 Teachers columns: {list(df_teachers.columns)}")
            print(f"SS3 Teachers count: {len(df_teachers)}")
            print("Sample teacher data:")
            print(df_teachers.head(3))
        else:
            print("❌ No 'Teachers' sheet found in SS3 file")
            
        if 'TeacherPerformance' in ss3_file.sheet_names:
            df_performance = pd.read_excel('student_records_SS3.xlsx', sheet_name='TeacherPerformance')
            print(f"SS3 TeacherPerformance columns: {list(df_performance.columns)}")
            print(f"SS3 TeacherPerformance count: {len(df_performance)}")
        else:
            print("❌ No 'TeacherPerformance' sheet found in SS3 file")
            
    except Exception as e:
        print(f"❌ Error reading SS3 file: {e}")
    
    # Check SS2 file
    try:
        ss2_file = pd.ExcelFile('student_records_SS2.xlsx')
        print(f"\nSS2 Sheets: {ss2_file.sheet_names}")
        
        if 'Teachers' in ss2_file.sheet_names:
            df_teachers = pd.read_excel('student_records_SS2.xlsx', sheet_name='Teachers')
            print(f"SS2 Teachers columns: {list(df_teachers.columns)}")
            print(f"SS2 Teachers count: {len(df_teachers)}")
        else:
            print("❌ No 'Teachers' sheet found in SS2 file")
            
    except Exception as e:
        print(f"❌ Error reading SS2 file: {e}")

if __name__ == "__main__":
    check_excel_structure()
