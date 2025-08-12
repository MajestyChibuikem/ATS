"""
Arts Tier Feature Engineering for SSAS ML Models.

This module provides simplified, efficient feature engineering for arts subjects
(Government, Economics, History, Literature, Geography, Christian Religious Studies, Civic Education).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from core.apps.ml.models.feature_engineer import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class ArtsFeaturesEngineer(BaseFeatureEngineer):
    """
    Feature engineer for arts subjects (Government, Economics, History, Literature, etc.).
    
    Implements simplified, efficient features including:
    - Broader feature sets for arts/social sciences
    - Computational efficiency focus
    - Subject-specific patterns
    - Cross-subject relationships
    """
    
    def __init__(self):
        """Initialize arts features engineer."""
        super().__init__('arts')
        
        # Define feature columns for arts tier (simplified but comprehensive)
        self.feature_columns = [
            # Student performance features
            'student_performance_avg', 'student_performance_std', 'student_subject_count',
            'student_age', 'student_performance_rank',
            
            # Subject-specific features
            'subject_difficulty', 'subject_performance_std',
            'arts_subject_mastery', 'general_academic_ability',
            
            # Teacher quality features
            'teacher_quality_score', 'qualification_weight', 'specialization_alignment',
            'teacher_experience', 'teacher_performance_rating',
            
            # Temporal features
            'term_progression', 'academic_progression',
            'performance_trend', 'learning_acceleration',
            
            # Performance context
            'class_performance_rank', 'performance_vs_class_avg',
            'stream_performance_avg', 'arts_subject_gap',
            
            # Arts-specific features
            'analytical_thinking_arts', 'critical_thinking_score',
            'communication_skills', 'creativity_indicator',
            'social_science_aptitude', 'humanities_orientation',
            'writing_ability', 'reading_comprehension',
            'cultural_awareness', 'logical_reasoning_arts',
            'memorization_skills', 'interpretation_ability'
        ]
        
        # Categorical features
        self.categorical_features = [
            'student_gender', 'student_stream', 'teacher_qualification',
            'teacher_specialization', 'term', 'student_class'
        ]
        
        # Numerical features
        self.numerical_features = [col for col in self.feature_columns 
                                 if col not in self.categorical_features]
        
        # Arts subject categories
        self.arts_categories = {
            'social_sciences': ['Government', 'Economics', 'History', 'Geography'],
            'languages_literature': ['Literature', 'English Language'],
            'religious_studies': ['Christian Religious Studies'],
            'civic_education': ['Civic Education']
        }
        
        logger.info(f"Initialized Arts Features Engineer with {len(self.feature_columns)} features")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specific to arts subjects.
        
        Args:
            df: Raw training data (should already have base features from pipeline)
            
        Returns:
            DataFrame with arts tier features
        """
        logger.info("Engineering arts tier features")
        
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
        
        # Arts subject-specific features
        df = self._add_arts_subject_features(df)
        df = self._add_analytical_thinking_features(df)
        df = self._add_communication_skills_features(df)
        df = self._add_social_science_features(df)
        df = self._add_humanities_features(df)
        df = self._add_cross_arts_features(df)
        
        # Ensure all required features are present
        df = self._ensure_feature_completeness(df)
        
        logger.info(f"Engineered {len(df.columns)} features for arts tier")
        return df
    
    def _add_arts_subject_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to arts subjects."""
        # Arts subject mastery (average performance in arts subjects)
        arts_subjects = ['Government', 'Economics', 'History', 'Literature', 'Geography', 
                        'Christian Religious Studies', 'Civic Education']
        df['arts_subject_mastery'] = df[df['subject_name'].isin(arts_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        # Arts subject gap (difference from other subjects)
        df['arts_subject_gap'] = df['arts_subject_mastery'] - df['student_performance_avg']
        
        # General academic ability (broader than just arts)
        df['general_academic_ability'] = df.groupby('student_id')['total_score'].transform('mean')
        
        # Arts difficulty adaptation
        df['arts_difficulty_adaptation'] = df.groupby('student_id')['total_score'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
        )
        
        return df
    
    def _add_analytical_thinking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add analytical thinking features for arts subjects."""
        # Analytical thinking in arts context
        analytical_subjects = ['Government', 'Economics', 'History', 'Geography']
        df['analytical_thinking_arts'] = df[df['subject_name'].isin(analytical_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        # Critical thinking score (based on performance consistency)
        df['critical_thinking_score'] = df.groupby('student_id')['total_score'].transform(
            lambda x: 1 / (1 + x.std()) if x.std() > 0 else 1.0
        )
        
        # Logical reasoning in arts
        df['logical_reasoning_arts'] = df.groupby('student_id')['total_score'].transform(
            lambda x: x.rolling(window=2, min_periods=1).mean().fillna(x.mean())
        )
        
        return df
    
    def _add_communication_skills_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add communication skills features."""
        # Writing ability (based on continuous assessment)
        df['writing_ability'] = df['continuous_assessment'].fillna(0)
        
        # Reading comprehension (based on examination performance)
        df['reading_comprehension'] = df['examination_score'].fillna(0)
        
        # Communication skills (combination of writing and reading)
        df['communication_skills'] = (df['writing_ability'] + df['reading_comprehension']) / 2
        
        # Creativity indicator (based on performance variance)
        df['creativity_indicator'] = df.groupby('student_id')['total_score'].transform(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        
        return df
    
    def _add_social_science_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add social science specific features."""
        # Social science aptitude
        social_science_subjects = ['Government', 'Economics', 'History', 'Geography']
        df['social_science_aptitude'] = df[df['subject_name'].isin(social_science_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        # Cultural awareness (based on performance in cultural subjects)
        cultural_subjects = ['History', 'Geography', 'Christian Religious Studies']
        df['cultural_awareness'] = df[df['subject_name'].isin(cultural_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        # Interpretation ability (based on performance consistency in interpretation-heavy subjects)
        interpretation_subjects = ['Literature', 'History', 'Christian Religious Studies']
        df['interpretation_ability'] = df[df['subject_name'].isin(interpretation_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        return df
    
    def _add_humanities_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add humanities specific features."""
        # Humanities orientation
        humanities_subjects = ['Literature', 'History', 'Christian Religious Studies', 'Civic Education']
        df['humanities_orientation'] = df[df['subject_name'].isin(humanities_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        # Memorization skills (based on performance in memorization-heavy subjects)
        memorization_subjects = ['History', 'Christian Religious Studies', 'Civic Education']
        df['memorization_skills'] = df[df['subject_name'].isin(memorization_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        return df
    
    def _add_cross_arts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-arts subject features."""
        # Cross-arts correlation (how well performance correlates across arts subjects)
        arts_subjects = ['Government', 'Economics', 'History', 'Literature', 'Geography', 
                        'Christian Religious Studies', 'Civic Education']
        arts_data = df[df['subject_name'].isin(arts_subjects)]
        
        # Calculate correlation for each student across arts subjects
        correlations = {}
        for student_id in df['student_id'].unique():
            student_data = arts_data[arts_data['student_id'] == student_id]
            if len(student_data) >= 2:
                # Calculate correlation between different arts subjects
                subject_scores = student_data.groupby('subject_name')['total_score'].mean()
                if len(subject_scores) >= 2:
                    try:
                        if len(subject_scores) == 2:
                            corr = np.corrcoef(subject_scores.values)[0, 1]
                        else:
                            corr_matrix = np.corrcoef(subject_scores.values)
                            upper_tri = np.triu_indices(len(subject_scores), k=1)
                            corr = np.mean(corr_matrix[upper_tri])
                        correlations[student_id] = corr if not np.isnan(corr) else 0.0
                    except (ValueError, IndexError):
                        correlations[student_id] = 0.0
                else:
                    correlations[student_id] = 0.0
            else:
                correlations[student_id] = 0.0
        
        df['cross_arts_correlation'] = df['student_id'].map(correlations)
        
        # Arts subject progression (performance improvement across arts subjects)
        df['arts_progression'] = df.groupby(['student_id', 'subject_name'])['total_score'].transform(
            lambda x: x.diff().rolling(window=2, min_periods=1).mean().fillna(0)
        )
        
        # Arts specialization (which arts subject student performs best in)
        arts_performance = df[df['subject_name'].isin(arts_subjects)].groupby(['student_id', 'subject_name'])['total_score'].mean().reset_index()
        if not arts_performance.empty:
            best_arts = arts_performance.loc[arts_performance.groupby('student_id')['total_score'].idxmax()]
            df['best_arts_subject'] = df['student_id'].map(
                dict(zip(best_arts['student_id'], best_arts['subject_name']))
            )
        else:
            df['best_arts_subject'] = 'Unknown'
        
        return df
    
    def _ensure_feature_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present and properly filled."""
        # Fill missing values for engineered features
        feature_fill_values = {
            'arts_subject_mastery': df['student_performance_avg'],
            'arts_subject_gap': 0.0,
            'general_academic_ability': df['student_performance_avg'],
            'arts_difficulty_adaptation': 0.0,
            'analytical_thinking_arts': df['student_performance_avg'],
            'critical_thinking_score': 0.5,
            'logical_reasoning_arts': df['student_performance_avg'],
            'writing_ability': df['continuous_assessment'].fillna(0),
            'reading_comprehension': df['examination_score'].fillna(0),
            'communication_skills': df['continuous_assessment'].fillna(0),
            'creativity_indicator': 0.0,
            'social_science_aptitude': df['student_performance_avg'],
            'cultural_awareness': df['student_performance_avg'],
            'interpretation_ability': df['student_performance_avg'],
            'humanities_orientation': df['student_performance_avg'],
            'memorization_skills': df['student_performance_avg'],
            'cross_arts_correlation': 0.0,
            'arts_progression': 0.0,
            'best_arts_subject': 'Unknown'
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
        """Get feature importance weights for arts tier."""
        return {
            'general_academic_ability': 0.15,
            'arts_subject_mastery': 0.12,
            'analytical_thinking_arts': 0.10,
            'communication_skills': 0.10,
            'teacher_quality_score': 0.08,
            'critical_thinking_score': 0.08,
            'social_science_aptitude': 0.07,
            'humanities_orientation': 0.07,
            'cultural_awareness': 0.06,
            'interpretation_ability': 0.05,
            'logical_reasoning_arts': 0.04,
            'memorization_skills': 0.03,
            'cross_arts_correlation': 0.02,
            'creativity_indicator': 0.01
        }
    
    def get_arts_categories(self) -> Dict[str, List[str]]:
        """Get arts subject categories."""
        return self.arts_categories
    
    def get_subject_category(self, subject_name: str) -> Optional[str]:
        """Get the category of an arts subject."""
        for category, subjects in self.arts_categories.items():
            if subject_name in subjects:
                return category
        return None
