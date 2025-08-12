"""
Base Feature Engineering for SSAS ML Models.

This module provides the foundation for feature engineering across all three tiers
of the performance prediction system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging

from core.apps.students.models import Student, StudentScore, Teacher, TeacherPerformance

logger = logging.getLogger(__name__)


class BaseFeatureEngineer(ABC):
    """
    Abstract base class for feature engineering.
    
    Provides common functionality for all tier-specific feature engineers.
    """
    
    def __init__(self, tier_name: str):
        """
        Initialize the feature engineer.
        
        Args:
            tier_name: Name of the tier (critical, science, arts)
        """
        self.tier_name = tier_name
        self.feature_columns = []
        self.categorical_features = []
        self.numerical_features = []
        
        logger.info(f"Initialized {tier_name} feature engineer")
    
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for the specific tier.
        
        Args:
            df: Raw training data
            
        Returns:
            DataFrame with engineered features
        """
        pass
    
    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns for this tier."""
        return self.feature_columns
    
    def get_categorical_features(self) -> List[str]:
        """Get the list of categorical features."""
        return self.categorical_features
    
    def get_numerical_features(self) -> List[str]:
        """Get the list of numerical features."""
        return self.numerical_features


class CommonFeatureEngineer:
    """
    Common feature engineering functionality shared across all tiers.
    """
    
    @staticmethod
    def add_student_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add student-level features."""
        # Student performance averages
        df['student_performance_avg'] = df.groupby('student_id')['total_score'].transform('mean')
        df['student_performance_std'] = df.groupby('student_id')['total_score'].transform('std')
        df['student_subject_count'] = df.groupby('student_id')['subject_name'].transform('nunique')
        
        # Student age and demographics
        df['student_age'] = df['student_age'].fillna(0)
        
        return df
    
    @staticmethod
    def add_subject_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add subject-level features."""
        # Subject difficulty and performance
        df['subject_difficulty'] = df.groupby('subject_name')['total_score'].transform('mean')
        df['subject_performance_std'] = df.groupby('subject_name')['total_score'].transform('std')
        
        return df
    
    @staticmethod
    def add_teacher_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add teacher quality features."""
        # Teacher quality score
        df['teacher_quality_score'] = (
            df['teacher_experience'] * 0.3 +
            df['teacher_performance_rating'] * 0.4 +
            (df['teacher_years_at_school'] / 10) * 0.3
        )
        
        # Qualification level encoding
        qualification_weights = {
            'PhD': 5, 'M.Ed': 4, 'B.Ed': 3, 'B.Sc + PGDE': 3, 'HND + PGDE': 2
        }
        df['qualification_weight'] = df['teacher_qualification'].map(qualification_weights).fillna(2)
        
        # Specialization alignment
        df['specialization_alignment'] = df.apply(
            lambda row: CommonFeatureEngineer._calculate_specialization_alignment(row), axis=1
        )
        
        return df
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        # Term progression
        df['term_progression'] = df['term'].map({'First Term': 1, 'Second Term': 2, 'Third Term': 3})
        
        # Academic progression
        df['academic_progression'] = df['student_class'].map({'SS1': 1, 'SS2': 2, 'SS3': 3})
        
        return df
    
    @staticmethod
    def add_performance_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add performance trend features."""
        # Score trends
        df['score_trend'] = df.groupby(['student_id', 'subject_name'])['total_score'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # Class context
        df['class_performance_rank'] = df.groupby(['subject_name', 'academic_year', 'term'])['total_score'].rank(pct=True)
        df['performance_vs_class_avg'] = df['total_score'] - df['class_average']
        
        # Stream-specific performance
        df['stream_performance_avg'] = df.groupby(['student_stream', 'subject_name'])['total_score'].transform('mean')
        
        return df
    
    @staticmethod
    def _calculate_specialization_alignment(row: pd.Series) -> float:
        """Calculate alignment between teacher specialization and subject."""
        subject = row['subject_name']
        specialization = row['teacher_specialization']
        
        alignment_map = {
            'Mathematics': ['Mathematics', 'Sciences'],
            'English Language': ['Languages', 'General'],
            'Physics': ['Sciences', 'Mathematics'],
            'Chemistry': ['Sciences'],
            'Biology': ['Sciences'],
            'Government': ['Arts', 'General'],
            'Economics': ['Arts', 'General'],
            'Literature': ['Languages', 'Arts'],
        }
        
        if subject in alignment_map:
            return 1.0 if specialization in alignment_map[subject] else 0.3
        else:
            return 0.5
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for feature engineering."""
        # Fill missing values
        df = df.fillna(0)
        
        # Ensure numeric types
        numeric_columns = ['total_score', 'continuous_assessment', 'examination_score', 
                          'class_average', 'teacher_experience', 'teacher_performance_rating']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df


class FeatureEngineeringPipeline:
    """
    Pipeline for orchestrating feature engineering across all tiers.
    """
    
    def __init__(self):
        """Initialize the feature engineering pipeline."""
        self.common_engineer = CommonFeatureEngineer()
        self.tier_engineers = {}
        
    def add_tier_engineer(self, tier_name: str, engineer: BaseFeatureEngineer):
        """Add a tier-specific feature engineer."""
        self.tier_engineers[tier_name] = engineer
        logger.info(f"Added {tier_name} feature engineer to pipeline")
    
    def engineer_features(self, df: pd.DataFrame, tier_name: str) -> pd.DataFrame:
        """
        Engineer features for a specific tier.
        
        Args:
            df: Raw training data
            tier_name: Tier name (critical, science, arts)
            
        Returns:
            DataFrame with engineered features
        """
        logger.info(f"Engineering features for {tier_name} tier")
        
        # Apply common features
        df = self.common_engineer.add_student_features(df)
        df = self.common_engineer.add_subject_features(df)
        df = self.common_engineer.add_teacher_features(df)
        df = self.common_engineer.add_temporal_features(df)
        df = self.common_engineer.add_performance_features(df)
        
        # Apply tier-specific features
        if tier_name in self.tier_engineers:
            df = self.tier_engineers[tier_name].engineer_features(df)
        
        # Clean data
        df = self.common_engineer.clean_data(df)
        
        logger.info(f"Engineered {len(df.columns)} features for {tier_name} tier")
        return df
    
    def get_feature_columns(self, tier_name: str) -> List[str]:
        """Get feature columns for a specific tier."""
        if tier_name in self.tier_engineers:
            return self.tier_engineers[tier_name].get_feature_columns()
        return []
