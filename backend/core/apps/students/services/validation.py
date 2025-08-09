"""
Data validation service for student data.
"""

import pandas as pd
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation service for student data."""
    
    @staticmethod
    def validate_student_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate student data DataFrame."""
        errors = []
        
        # Check required columns
        required_columns = ['StudentID', 'FirstName', 'LastName', 'DateOfBirth', 'Gender']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate student IDs
        if 'StudentID' in df.columns:
            duplicates = df[df['StudentID'].duplicated()]['StudentID'].tolist()
            if duplicates:
                errors.append(f"Duplicate student IDs found: {duplicates[:5]}...")
        
        # Check for valid gender values
        if 'Gender' in df.columns:
            valid_genders = ['Male', 'Female']
            invalid_genders = df[~df['Gender'].isin(valid_genders)]['Gender'].unique()
            if len(invalid_genders) > 0:
                errors.append(f"Invalid gender values: {invalid_genders}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_score_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate score data DataFrame."""
        errors = []
        
        # Check required columns
        required_columns = ['StudentID', 'SubjectName', 'Score']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for valid score ranges
        if 'Score' in df.columns:
            invalid_scores = df[(df['Score'] < 0) | (df['Score'] > 100)]['Score'].tolist()
            if invalid_scores:
                errors.append(f"Invalid scores (not between 0-100): {invalid_scores[:5]}...")
        
        return len(errors) == 0, errors 