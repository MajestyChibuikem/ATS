"""
Science Tier Feature Engineering for SSAS ML Models.

This module provides specialized feature engineering for science subjects
(Physics, Chemistry, Biology, Agricultural Science) with moderate complexity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from core.apps.ml.models.feature_engineer import BaseFeatureEngineer

logger = logging.getLogger(__name__)


class ScienceFeaturesEngineer(BaseFeatureEngineer):
    """
    Feature engineer for science subjects (Physics, Chemistry, Biology).
    
    Implements moderate complexity features including:
    - Prerequisite subject relationships (Math → Physics, Chemistry)
    - Laboratory performance indicators
    - Scientific reasoning patterns
    - Cross-subject dependencies
    """
    
    def __init__(self):
        """Initialize science features engineer."""
        super().__init__('science')
        
        # Define feature columns for science tier
        self.feature_columns = [
            # Student performance features
            'student_performance_avg', 'student_performance_std', 'student_subject_count',
            'student_age', 'student_performance_rank',
            
            # Subject-specific features
            'subject_difficulty', 'subject_performance_std',
            'science_subject_mastery', 'prerequisite_performance',
            
            # Teacher quality features
            'teacher_quality_score', 'qualification_weight', 'specialization_alignment',
            'teacher_experience', 'teacher_performance_rating',
            
            # Temporal features
            'term_progression', 'academic_progression',
            'performance_trend', 'learning_acceleration',
            
            # Performance context
            'class_performance_rank', 'performance_vs_class_avg',
            'stream_performance_avg', 'science_subject_gap',
            
            # Science-specific features
            'mathematical_foundation', 'scientific_reasoning_score',
            'laboratory_performance', 'theoretical_vs_practical_ratio',
            'prerequisite_alignment', 'cross_science_correlation',
            'experimental_skills', 'analytical_thinking_science',
            'problem_solving_science', 'conceptual_understanding'
        ]
        
        # Categorical features
        self.categorical_features = [
            'student_gender', 'student_stream', 'teacher_qualification',
            'teacher_specialization', 'term', 'student_class'
        ]
        
        # Numerical features
        self.numerical_features = [col for col in self.feature_columns 
                                 if col not in self.categorical_features]
        
        # Prerequisite relationships
        self.prerequisites = {
            'Physics': ['Mathematics'],
            'Chemistry': ['Mathematics'],
            'Biology': ['Chemistry'],  # Basic chemistry helps with biology
            'Agricultural Science': ['Biology']  # Biology helps with agriculture
        }
        
        logger.info(f"Initialized Science Features Engineer with {len(self.feature_columns)} features")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specific to science subjects.
        
        Args:
            df: Raw training data (should already have base features from pipeline)
            
        Returns:
            DataFrame with science tier features
        """
        logger.info("Engineering science tier features")
        
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
        
        # Science subject-specific features
        df = self._add_science_subject_features(df)
        df = self._add_prerequisite_features(df)
        df = self._add_mathematical_foundation_features(df)
        df = self._add_scientific_reasoning_features(df)
        df = self._add_laboratory_features(df)
        df = self._add_cross_science_features(df)
        
        # Ensure all required features are present
        df = self._ensure_feature_completeness(df)
        
        logger.info(f"Engineered {len(df.columns)} features for science tier")
        return df
    
    def _add_science_subject_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to science subjects."""
        # Science subject mastery (average performance in science subjects)
        science_subjects = ['Physics', 'Chemistry', 'Biology', 'Agricultural Science']
        df['science_subject_mastery'] = df[df['subject_name'].isin(science_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        # Science subject gap (difference from other subjects)
        df['science_subject_gap'] = df['science_subject_mastery'] - df['student_performance_avg']
        
        # Science difficulty adaptation
        df['science_difficulty_adaptation'] = df.groupby('student_id')['total_score'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
        )
        
        return df
    
    def _add_prerequisite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prerequisite relationship features."""
        # Prerequisite performance for each subject
        df['prerequisite_performance'] = 0.0
        df['prerequisite_alignment'] = 0.0
        
        for subject, prereqs in self.prerequisites.items():
            subject_mask = df['subject_name'] == subject
            
            for prereq in prereqs:
                # Get prerequisite scores for each student
                prereq_scores = df[df['subject_name'] == prereq].groupby('student_id')['total_score'].agg(['mean', 'std'])
                
                # Map prerequisite performance to students taking the subject
                df.loc[subject_mask, 'prerequisite_performance'] = df.loc[subject_mask, 'student_id'].map(
                    prereq_scores['mean'].fillna(df['student_performance_avg'])
                )
                
                # Calculate alignment (how well prerequisite performance predicts subject performance)
                df.loc[subject_mask, 'prerequisite_alignment'] = df.loc[subject_mask, 'student_id'].map(
                    (1 / (1 + prereq_scores['std'])).fillna(0.5)
                )
        
        return df
    
    def _add_mathematical_foundation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mathematical foundation features for science subjects."""
        # Mathematical foundation score (based on Math performance)
        math_scores = df[df['subject_name'] == 'Mathematics'].groupby('student_id')['total_score'].agg(['mean', 'std', 'count'])
        
        df['mathematical_foundation'] = df['student_id'].map(
            math_scores['mean'].fillna(df['student_performance_avg'])
        )
        
        # Math consistency for science (how consistent is math performance)
        df['math_consistency_for_science'] = df['student_id'].map(
            (1 / (1 + math_scores['std'])).fillna(0.5)
        )
        
        # Advanced math skills (for physics and chemistry)
        physics_chemistry_mask = df['subject_name'].isin(['Physics', 'Chemistry'])
        df.loc[physics_chemistry_mask, 'advanced_math_requirement'] = 1.0
        df.loc[~physics_chemistry_mask, 'advanced_math_requirement'] = 0.0
        
        return df
    
    def _add_scientific_reasoning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add scientific reasoning features."""
        # Scientific reasoning score (based on science performance patterns)
        science_scores = df[df['subject_name'].isin(['Physics', 'Chemistry', 'Biology'])].groupby('student_id')['total_score'].agg(['mean', 'std'])
        
        df['scientific_reasoning_score'] = df['student_id'].map(
            science_scores['mean'].fillna(df['student_performance_avg'])
        )
        
        # Problem solving in science context
        df['problem_solving_science'] = df['student_id'].map(
            (1 / (1 + science_scores['std'])).fillna(0.5)
        )
        
        # Analytical thinking in science
        analytical_subjects = ['Physics', 'Chemistry', 'Mathematics']
        df['analytical_thinking_science'] = df[df['subject_name'].isin(analytical_subjects)].groupby('student_id')['total_score'].transform('mean')
        
        # Conceptual understanding (consistency across science subjects)
        df['conceptual_understanding'] = df.groupby('student_id')['total_score'].transform(
            lambda x: 1 / (1 + x.std()) if x.std() > 0 else 1.0
        )
        
        return df
    
    def _add_laboratory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add laboratory and practical performance features."""
        # Laboratory performance (based on continuous assessment vs examination)
        df['laboratory_performance'] = df['continuous_assessment'].fillna(0)
        
        # Theoretical vs practical ratio
        df['theoretical_vs_practical_ratio'] = (
            df['examination_score'] / (df['continuous_assessment'] + 1)  # Avoid division by zero
        )
        
        # Experimental skills (consistency in practical work)
        df['experimental_skills'] = df.groupby(['student_id', 'subject_name'])['continuous_assessment'].transform(
            lambda x: 1 / (1 + x.std()) if x.std() > 0 else 1.0
        )
        
        # Practical vs theoretical balance
        df['practical_theoretical_balance'] = abs(
            df['continuous_assessment'] - df['examination_score']
        ) / (df['total_score'] + 1)  # Normalized difference
        
        return df
    
    def _add_cross_science_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-science subject features."""
        # Cross-science correlation (how well performance correlates across science subjects)
        science_subjects = ['Physics', 'Chemistry', 'Biology']
        science_data = df[df['subject_name'].isin(science_subjects)]
        
        # Calculate correlation for each student across science subjects
        correlations = {}
        for student_id in df['student_id'].unique():
            student_data = science_data[science_data['student_id'] == student_id]
            if len(student_data) >= 2:
                # Calculate correlation between different science subjects
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
        
        df['cross_science_correlation'] = df['student_id'].map(correlations)
        
        # Science subject progression (performance improvement across science subjects)
        df['science_progression'] = df.groupby(['student_id', 'subject_name'])['total_score'].transform(
            lambda x: x.diff().rolling(window=2, min_periods=1).mean().fillna(0)
        )
        
        # Science specialization (which science subject student performs best in)
        science_performance = df[df['subject_name'].isin(science_subjects)].groupby(['student_id', 'subject_name'])['total_score'].mean().reset_index()
        best_science = science_performance.loc[science_performance.groupby('student_id')['total_score'].idxmax()]
        df['best_science_subject'] = df['student_id'].map(
            dict(zip(best_science['student_id'], best_science['subject_name']))
        )
        
        return df
    
    def _ensure_feature_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required features are present and properly filled."""
        # Fill missing values for engineered features
        feature_fill_values = {
            'science_subject_mastery': df['student_performance_avg'],
            'science_subject_gap': 0.0,
            'science_difficulty_adaptation': 0.0,
            'prerequisite_performance': df['student_performance_avg'],
            'prerequisite_alignment': 0.5,
            'mathematical_foundation': df['student_performance_avg'],
            'math_consistency_for_science': 0.5,
            'advanced_math_requirement': 0.0,
            'scientific_reasoning_score': df['student_performance_avg'],
            'problem_solving_science': 0.5,
            'analytical_thinking_science': df['student_performance_avg'],
            'conceptual_understanding': 0.5,
            'laboratory_performance': df['continuous_assessment'].fillna(0),
            'theoretical_vs_practical_ratio': 1.0,
            'experimental_skills': 0.5,
            'practical_theoretical_balance': 0.0,
            'cross_science_correlation': 0.0,
            'science_progression': 0.0,
            'best_science_subject': 'Unknown'
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
        """Get feature importance weights for science tier."""
        return {
            'mathematical_foundation': 0.15,
            'prerequisite_performance': 0.12,
            'scientific_reasoning_score': 0.12,
            'science_subject_mastery': 0.10,
            'teacher_quality_score': 0.08,
            'laboratory_performance': 0.08,
            'analytical_thinking_science': 0.07,
            'problem_solving_science': 0.06,
            'prerequisite_alignment': 0.05,
            'experimental_skills': 0.05,
            'cross_science_correlation': 0.04,
            'conceptual_understanding': 0.03,
            'science_progression': 0.02,
            'practical_theoretical_balance': 0.01
        }
    
    def get_prerequisites(self, subject_name: str) -> List[str]:
        """Get prerequisite subjects for a given science subject."""
        return self.prerequisites.get(subject_name, [])
    
    def validate_prerequisites(self, student_id: str, subject_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate if a student meets prerequisites for a science subject."""
        prereqs = self.get_prerequisites(subject_name)
        
        if not prereqs:
            return {'meets_prerequisites': True, 'prerequisite_score': 100.0}
        
        # Get prerequisite scores for the student
        prereq_scores = []
        for prereq in prereqs:
            prereq_data = df[(df['student_id'] == student_id) & (df['subject_name'] == prereq)]
            if not prereq_data.empty:
                prereq_scores.append(prereq_data['total_score'].mean())
        
        if not prereq_scores:
            return {'meets_prerequisites': False, 'prerequisite_score': 0.0}
        
        avg_prereq_score = np.mean(prereq_scores)
        meets_prereqs = avg_prereq_score >= 50.0  # Minimum 50% in prerequisites
        
        return {
            'meets_prerequisites': meets_prereqs,
            'prerequisite_score': avg_prereq_score,
            'prerequisites': prereqs,
            'individual_scores': dict(zip(prereqs, prereq_scores))
        }
