"""
Critical Tier Feature Engineering for SSAS ML Models.

This module provides specialized feature engineering for critical subjects
(Mathematics, English Language, Further Mathematics) with highest complexity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from core.apps.ml.models.feature_engineer import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class CriticalFeaturesEngineer(BaseFeatureEngineer):
    """
    Feature engineer for critical subjects (Mathematics, English).
    
    Implements highest complexity features including:
    - Advanced mathematical patterns
    - Language proficiency indicators
    - Cross-subject dependencies
    - Performance trajectory analysis
    """
    
    def __init__(self):
        """Initialize critical features engineer."""
        super().__init__('critical')
        
        # Define feature columns for critical tier
        self.feature_columns = [
            # Student performance features
            'student_performance_avg', 'student_performance_std', 'student_subject_count',
            'student_age', 'student_performance_rank',
            
            # Subject-specific features
            'subject_difficulty', 'subject_performance_std',
            'math_english_correlation', 'critical_subject_mastery',
            
            # Teacher quality features
            'teacher_quality_score', 'qualification_weight', 'specialization_alignment',
            'teacher_experience', 'teacher_performance_rating',
            
            # Temporal features
            'term_progression', 'academic_progression',
            'performance_trend', 'learning_acceleration',
            
            # Performance context
            'class_performance_rank', 'performance_vs_class_avg',
            'stream_performance_avg', 'critical_subject_gap',
            
            # Advanced features
            'mathematical_reasoning_score', 'language_proficiency_score',
            'problem_solving_ability', 'analytical_thinking_score',
            'consistency_score', 'improvement_rate',
            'difficulty_adaptation', 'stress_handling_score'
        ]
        
        # Categorical features
        self.categorical_features = [
            'student_gender', 'student_stream', 'teacher_qualification',
            'teacher_specialization', 'term', 'student_class'
        ]
        
        # Numerical features
        self.numerical_features = [col for col in self.feature_columns 
                                 if col not in self.categorical_features]
        
        logger.info(f"Initialized Critical Features Engineer with {len(self.feature_columns)} features")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specific to critical subjects.
        
        Args:
            df: Raw training data (should already have base features from pipeline)
            
        Returns:
            DataFrame with critical tier features
        """
        logger.info("Engineering critical tier features")
        
        # Apply base features first if not already present
        df = df.copy()
        
        # Ensure base features are present
        if 'student_performance_avg' not in df.columns:
            from core.apps.ml.models.feature_engineer import CommonFeatureEngineer
            common_engineer = CommonFeatureEngineer()
            df = common_engineer.add_student_features(df)
            df = common_engineer.add_subject_features(df)
            df = common_engineer.add_teacher_features(df)
            df = common_engineer.add_temporal_features(df)
            df = common_engineer.add_performance_features(df)
            df = common_engineer.clean_data(df)
        
        # Critical subject-specific features
        df = self._add_critical_subject_features(df)
        df = self._add_mathematical_reasoning_features(df)
        df = self._add_language_proficiency_features(df)
        df = self._add_advanced_performance_features(df)
        df = self._add_cross_subject_features(df)
        df = self._add_learning_pattern_features(df)
        
        # Ensure all required features are present
        df = self._ensure_feature_completeness(df)
        
        logger.info(f"Engineered {len(df.columns)} features for critical tier")
        return df
    
    def _add_critical_subject_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to critical subjects."""
        # Critical subject mastery (average performance in Math/English)
        critical_subjects = ['Mathematics', 'English Language', 'Further Mathematics']
        df['critical_subject_mastery'] = df[df['subject_name'].isin(critical_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        # Critical subject gap (difference from other subjects)
        df['critical_subject_gap'] = df['critical_subject_mastery'] - df['student_performance_avg']
        
        # Subject difficulty adaptation
        df['difficulty_adaptation'] = df.groupby('student_id')['total_score'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
        )
        
        return df
    
    def _add_mathematical_reasoning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mathematical reasoning features."""
        # Mathematical reasoning score (based on Math performance patterns)
        math_scores = df[df['subject_name'] == 'Mathematics'].groupby('student_id')['total_score'].agg(['mean', 'std', 'count'])
        
        df['mathematical_reasoning_score'] = df['student_id'].map(
            math_scores['mean'].fillna(df['student_performance_avg'])
        )
        
        # Problem solving ability (consistency in Math performance)
        df['problem_solving_ability'] = df['student_id'].map(
            (1 / (1 + math_scores['std'])).fillna(0.5)
        )
        
        # Analytical thinking (performance in analytical subjects)
        analytical_subjects = ['Mathematics', 'Physics', 'Chemistry']
        df['analytical_thinking_score'] = df[df['subject_name'].isin(analytical_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        return df
    
    def _add_language_proficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add language proficiency features."""
        # Language proficiency score (based on English performance)
        english_scores = df[df['subject_name'] == 'English Language'].groupby('student_id')['total_score'].agg(['mean', 'std', 'count'])
        
        df['language_proficiency_score'] = df['student_id'].map(
            english_scores['mean'].fillna(df['student_performance_avg'])
        )
        
        # Communication skills (performance in language-heavy subjects)
        language_subjects = ['English Language', 'Literature', 'Government']
        df['communication_skills'] = df[df['subject_name'].isin(language_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        return df
    
    def _add_advanced_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced performance analysis features."""
        # Performance consistency
        df['consistency_score'] = df.groupby('student_id')['total_score'].transform(
            lambda x: 1 / (1 + x.std()) if x.std() > 0 else 1.0
        )
        
        # Improvement rate (trend analysis)
        df['improvement_rate'] = df.groupby(['student_id', 'subject_name'])['total_score'].transform(
            lambda x: x.diff().rolling(window=2, min_periods=1).mean().fillna(0)
        )
        
        # Learning acceleration (rate of improvement)
        df['learning_acceleration'] = df.groupby(['student_id', 'subject_name'])['total_score'].transform(
            lambda x: x.diff().diff().rolling(window=3, min_periods=1).mean().fillna(0)
        )
        
        # Stress handling (performance under pressure - exam vs continuous assessment)
        df['stress_handling_score'] = (
            df['examination_score'] - df['continuous_assessment']
        ) / (df['continuous_assessment'] + 1)  # Avoid division by zero
        
        return df
    
    def _add_cross_subject_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-subject dependency features."""
        # Math-English correlation (how well Math and English performance correlate)
        math_english_data = df[df['subject_name'].isin(['Mathematics', 'English Language'])]
        
        # Calculate correlation for each student
        correlations = {}
        for student_id in df['student_id'].unique():
            student_data = math_english_data[math_english_data['student_id'] == student_id]
            if len(student_data) >= 2:
                math_scores = student_data[student_data['subject_name'] == 'Mathematics']['total_score']
                english_scores = student_data[student_data['subject_name'] == 'English Language']['total_score']
                if len(math_scores) > 0 and len(english_scores) > 0:
                    corr = np.corrcoef(math_scores, english_scores)[0, 1]
                    correlations[student_id] = corr if not np.isnan(corr) else 0.0
                else:
                    correlations[student_id] = 0.0
            else:
                correlations[student_id] = 0.0
        
        df['math_english_correlation'] = df['student_id'].map(correlations)
        
        # Prerequisite subject performance
        df['prerequisite_performance'] = df.groupby('student_id')['total_score'].transform(
            lambda x: x.shift(1).fillna(x.mean())
        )
        
        return df
    
    def _add_learning_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add learning pattern and behavior features."""
        # Study pattern consistency
        df['study_pattern_consistency'] = df.groupby('student_id')['total_score'].transform(
            lambda x: 1 - (x.rolling(window=3, min_periods=1).std() / x.rolling(window=3, min_periods=1).mean()).fillna(0)
        )
        
        # Performance volatility
        df['performance_volatility'] = df.groupby('student_id')['total_score'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
        )
        
        # Recovery ability (bounce back from low scores)
        df['recovery_ability'] = df.groupby(['student_id', 'subject_name'])['total_score'].transform(
            lambda x: (x - x.rolling(window=3, min_periods=1).min()) / (x.rolling(window=3, min_periods=1).max() - x.rolling(window=3, min_periods=1).min() + 1)
        )
        
        # Peak performance indicator
        df['peak_performance'] = df.groupby('student_id')['total_score'].transform(
            lambda x: (x == x.max()).astype(int)
        )
        
        return df
    
    def _ensure_feature_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present and properly filled."""
        # Fill missing values for engineered features
        feature_fill_values = {
            'critical_subject_mastery': df['student_performance_avg'],
            'critical_subject_gap': 0.0,
            'difficulty_adaptation': 0.0,
            'mathematical_reasoning_score': df['student_performance_avg'],
            'problem_solving_ability': 0.5,
            'analytical_thinking_score': df['student_performance_avg'],
            'language_proficiency_score': df['student_performance_avg'],
            'communication_skills': df['student_performance_avg'],
            'consistency_score': 0.5,
            'improvement_rate': 0.0,
            'learning_acceleration': 0.0,
            'stress_handling_score': 0.0,
            'math_english_correlation': 0.0,
            'prerequisite_performance': df['student_performance_avg'],
            'study_pattern_consistency': 0.5,
            'performance_volatility': 0.0,
            'recovery_ability': 0.5,
            'peak_performance': 0.0
        }
        
        for feature, fill_value in feature_fill_values.items():
            if feature not in df.columns:
                df[feature] = fill_value
            else:
                df[feature] = df[feature].fillna(fill_value)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        return df
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Get feature importance weights for critical tier."""
        return {
            'student_performance_avg': 0.15,
            'mathematical_reasoning_score': 0.12,
            'language_proficiency_score': 0.12,
            'critical_subject_mastery': 0.10,
            'teacher_quality_score': 0.08,
            'consistency_score': 0.08,
            'improvement_rate': 0.07,
            'analytical_thinking_score': 0.06,
            'problem_solving_ability': 0.06,
            'stress_handling_score': 0.05,
            'math_english_correlation': 0.04,
            'difficulty_adaptation': 0.03,
            'learning_acceleration': 0.02,
            'recovery_ability': 0.02
        }
