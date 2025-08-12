"""
Cross-Subject Validation for SSAS ML Models.

This module provides validation strategies for subject interactions and prerequisite
relationships, ensuring consistency across related subjects.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class CrossSubjectValidator:
    """
    Cross-subject validation for ML models with subject interaction awareness.
    
    Implements validation strategies for educational data:
    - Prerequisite subject validation
    - Cross-subject performance consistency
    - Subject correlation analysis
    - Performance prediction across related subjects
    """
    
    def __init__(self):
        """Initialize cross-subject validator."""
        # Define prerequisite relationships
        self.prerequisites = {
            'Physics': ['Mathematics'],
            'Chemistry': ['Mathematics'],
            'Biology': ['Chemistry'],
            'Agricultural Science': ['Biology'],
            'Further Mathematics': ['Mathematics'],
            'Economics': ['Mathematics'],
            'Geography': ['Mathematics']  # Basic math for statistics
        }
        
        # Define subject categories
        self.subject_categories = {
            'mathematics': ['Mathematics', 'Further Mathematics'],
            'sciences': ['Physics', 'Chemistry', 'Biology', 'Agricultural Science'],
            'languages': ['English Language', 'Literature'],
            'social_sciences': ['Government', 'Economics', 'History', 'Geography'],
            'arts': ['Christian Religious Studies', 'Civic Education']
        }
        
        logger.info("Initialized Cross-Subject Validator")
    
    def validate_prerequisites(self, X: pd.DataFrame, y: pd.Series, 
                             subject_name: str, model) -> Dict[str, Any]:
        """
        Validate model performance considering prerequisite subjects.
        
        Args:
            X: Feature matrix
            y: Target variable
            subject_name: Name of the subject being predicted
            model: Trained model
            
        Returns:
            Dictionary with prerequisite validation results
        """
        logger.info(f"Validating prerequisites for {subject_name}")
        
        prereqs = self.prerequisites.get(subject_name, [])
        
        if not prereqs:
            return {
                'subject': subject_name,
                'prerequisites': [],
                'prerequisite_validation': 'no_prerequisites',
                'performance_with_prereqs': None,
                'performance_without_prereqs': None
            }
        
        # Split data based on prerequisite performance
        prereq_performance = self._get_prerequisite_performance(X, prereqs)
        
        if prereq_performance is None:
            return {
                'subject': subject_name,
                'prerequisites': prereqs,
                'prerequisite_validation': 'no_prerequisite_data',
                'performance_with_prereqs': None,
                'performance_without_prereqs': None
            }
        
        # Split into high and low prerequisite performance groups
        high_prereq_mask = prereq_performance >= prereq_performance.median()
        low_prereq_mask = prereq_performance < prereq_performance.median()
        
        # Validate model performance on each group
        high_prereq_performance = self._validate_group(X[high_prereq_mask], y[high_prereq_mask], model)
        low_prereq_performance = self._validate_group(X[low_prereq_mask], y[low_prereq_mask], model)
        
        # Calculate prerequisite effect
        prereq_effect = high_prereq_performance['r2'] - low_prereq_performance['r2']
        
        return {
            'subject': subject_name,
            'prerequisites': prereqs,
            'prerequisite_validation': 'completed',
            'performance_with_prereqs': high_prereq_performance,
            'performance_without_prereqs': low_prereq_performance,
            'prerequisite_effect': prereq_effect,
            'prerequisite_importance': abs(prereq_effect) > 0.1
        }
    
    def validate_cross_subject_consistency(self, X: pd.DataFrame, y: pd.Series,
                                         subject_name: str, model) -> Dict[str, Any]:
        """
        Validate consistency across related subjects.
        
        Args:
            X: Feature matrix
            y: Target variable
            subject_name: Name of the subject being predicted
            model: Trained model
            
        Returns:
            Dictionary with cross-subject consistency results
        """
        logger.info(f"Validating cross-subject consistency for {subject_name}")
        
        # Find related subjects
        related_subjects = self._get_related_subjects(subject_name)
        
        if not related_subjects:
            return {
                'subject': subject_name,
                'related_subjects': [],
                'consistency_validation': 'no_related_subjects',
                'cross_subject_correlation': None,
                'performance_consistency': None
            }
        
        # Calculate cross-subject correlations
        correlations = {}
        for related_subject in related_subjects:
            correlation = self._calculate_subject_correlation(X, subject_name, related_subject)
            correlations[related_subject] = correlation
        
        # Validate performance consistency
        consistency_metrics = self._validate_performance_consistency(X, y, subject_name, related_subjects)
        
        return {
            'subject': subject_name,
            'related_subjects': related_subjects,
            'consistency_validation': 'completed',
            'cross_subject_correlations': correlations,
            'performance_consistency': consistency_metrics,
            'overall_consistency': np.mean(list(correlations.values())) if correlations else 0.0
        }
    
    def validate_subject_category_performance(self, X: pd.DataFrame, y: pd.Series,
                                            subject_name: str, model) -> Dict[str, Any]:
        """
        Validate performance within subject category.
        
        Args:
            X: Feature matrix
            y: Target variable
            subject_name: Name of the subject being predicted
            model: Trained model
            
        Returns:
            Dictionary with category performance results
        """
        logger.info(f"Validating category performance for {subject_name}")
        
        # Find subject category
        category = self._get_subject_category(subject_name)
        
        if not category:
            return {
                'subject': subject_name,
                'category': 'unknown',
                'category_validation': 'unknown_category',
                'category_performance': None
            }
        
        # Get category subjects
        category_subjects = self.subject_categories[category]
        
        # Calculate category-specific metrics
        category_performance = self._calculate_category_performance(X, y, category_subjects)
        
        return {
            'subject': subject_name,
            'category': category,
            'category_validation': 'completed',
            'category_performance': category_performance,
            'category_subjects': category_subjects
        }
    
    def validate_science_subjects(self, X: pd.DataFrame, y: pd.Series, 
                                model, subject_name: str) -> Dict[str, Any]:
        """
        Validate science subjects with prerequisite and cross-subject analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Trained model
            subject_name: Subject name
            
        Returns:
            Dictionary with comprehensive validation results
        """
        logger.info(f"Validating science subject: {subject_name}")
        
        # Prerequisite validation
        prereq_results = self.validate_prerequisites(X, y, subject_name, model)
        
        # Cross-subject consistency
        consistency_results = self.validate_cross_subject_consistency(X, y, subject_name, model)
        
        # Category performance
        category_results = self.validate_subject_category_performance(X, y, subject_name, model)
        
        # Science-specific validation
        science_validation = self._validate_science_specific_patterns(X, y, subject_name)
        
        return {
            'subject': subject_name,
            'tier': 'science',
            'prerequisite_validation': prereq_results,
            'cross_subject_consistency': consistency_results,
            'category_performance': category_results,
            'science_specific_validation': science_validation,
            'overall_validation_score': self._calculate_overall_validation_score(
                prereq_results, consistency_results, category_results, science_validation
            )
        }
    
    def _get_prerequisite_performance(self, X: pd.DataFrame, prerequisites: List[str]) -> Optional[pd.Series]:
        """Get prerequisite performance for each student."""
        prereq_columns = []
        
        for prereq in prerequisites:
            prereq_col = f'{prereq.lower().replace(" ", "_")}_performance'
            if prereq_col in X.columns:
                prereq_columns.append(prereq_col)
        
        if not prereq_columns:
            return None
        
        # Average prerequisite performance
        prereq_performance = X[prereq_columns].mean(axis=1)
        return prereq_performance
    
    def _validate_group(self, X: pd.DataFrame, y: pd.Series, model) -> Dict[str, float]:
        """Validate model performance on a specific group."""
        if len(X) == 0:
            return {'r2': 0.0, 'mae': 0.0, 'mse': 0.0}
        
        try:
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            return {
                'r2': r2,
                'mae': mae,
                'mse': mse,
                'sample_count': len(X)
            }
        except Exception as e:
            logger.warning(f"Error validating group: {e}")
            return {'r2': 0.0, 'mae': 0.0, 'mse': 0.0, 'sample_count': len(X)}
    
    def _get_related_subjects(self, subject_name: str) -> List[str]:
        """Get subjects related to the given subject."""
        related = []
        
        # Add prerequisite subjects
        if subject_name in self.prerequisites:
            related.extend(self.prerequisites[subject_name])
        
        # Add subjects that have this subject as prerequisite
        for subject, prereqs in self.prerequisites.items():
            if subject_name in prereqs:
                related.append(subject)
        
        # Add subjects from same category
        category = self._get_subject_category(subject_name)
        if category and category in self.subject_categories:
            category_subjects = self.subject_categories[category]
            related.extend([s for s in category_subjects if s != subject_name])
        
        return list(set(related))  # Remove duplicates
    
    def _calculate_subject_correlation(self, X: pd.DataFrame, subject1: str, subject2: str) -> float:
        """Calculate correlation between two subjects."""
        # Look for subject performance columns
        subject1_col = f'{subject1.lower().replace(" ", "_")}_performance'
        subject2_col = f'{subject2.lower().replace(" ", "_")}_performance'
        
        if subject1_col in X.columns and subject2_col in X.columns:
            correlation = X[subject1_col].corr(X[subject2_col])
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _validate_performance_consistency(self, X: pd.DataFrame, y: pd.Series,
                                        subject_name: str, related_subjects: List[str]) -> Dict[str, Any]:
        """Validate performance consistency across related subjects."""
        consistency_metrics = {}
        
        for related_subject in related_subjects:
            # Calculate performance correlation
            correlation = self._calculate_subject_correlation(X, subject_name, related_subject)
            
            # Calculate performance difference
            subject_perf_col = f'{subject_name.lower().replace(" ", "_")}_performance'
            related_perf_col = f'{related_subject.lower().replace(" ", "_")}_performance'
            
            if subject_perf_col in X.columns and related_perf_col in X.columns:
                perf_diff = abs(X[subject_perf_col] - X[related_perf_col]).mean()
            else:
                perf_diff = 0.0
            
            consistency_metrics[related_subject] = {
                'correlation': correlation,
                'performance_difference': perf_diff,
                'consistency_score': 1 - perf_diff / 100  # Normalize to 0-1
            }
        
        return consistency_metrics
    
    def _get_subject_category(self, subject_name: str) -> Optional[str]:
        """Get the category of a subject."""
        for category, subjects in self.subject_categories.items():
            if subject_name in subjects:
                return category
        return None
    
    def _calculate_category_performance(self, X: pd.DataFrame, y: pd.Series,
                                      category_subjects: List[str]) -> Dict[str, Any]:
        """Calculate performance metrics for a subject category."""
        category_performance = {}
        
        for subject in category_subjects:
            subject_col = f'{subject.lower().replace(" ", "_")}_performance'
            if subject_col in X.columns:
                category_performance[subject] = {
                    'mean_performance': X[subject_col].mean(),
                    'std_performance': X[subject_col].std(),
                    'correlation_with_target': X[subject_col].corr(y)
                }
        
        return category_performance
    
    def _validate_science_specific_patterns(self, X: pd.DataFrame, y: pd.Series,
                                          subject_name: str) -> Dict[str, Any]:
        """Validate science-specific patterns."""
        science_patterns = {}
        
        # Check for laboratory performance
        if 'laboratory_performance' in X.columns:
            lab_correlation = X['laboratory_performance'].corr(y)
            science_patterns['laboratory_correlation'] = lab_correlation if not np.isnan(lab_correlation) else 0.0
        
        # Check for theoretical vs practical ratio
        if 'theoretical_vs_practical_ratio' in X.columns:
            ratio_correlation = X['theoretical_vs_practical_ratio'].corr(y)
            science_patterns['theoretical_practical_correlation'] = ratio_correlation if not np.isnan(ratio_correlation) else 0.0
        
        # Check for mathematical foundation
        if 'mathematical_foundation' in X.columns:
            math_correlation = X['mathematical_foundation'].corr(y)
            science_patterns['mathematical_foundation_correlation'] = math_correlation if not np.isnan(math_correlation) else 0.0
        
        # Check for scientific reasoning
        if 'scientific_reasoning_score' in X.columns:
            reasoning_correlation = X['scientific_reasoning_score'].corr(y)
            science_patterns['scientific_reasoning_correlation'] = reasoning_correlation if not np.isnan(reasoning_correlation) else 0.0
        
        return science_patterns
    
    def _calculate_overall_validation_score(self, prereq_results: Dict, consistency_results: Dict,
                                          category_results: Dict, science_validation: Dict) -> float:
        """Calculate overall validation score."""
        scores = []
        
        # Prerequisite validation score
        if prereq_results.get('prerequisite_validation') == 'completed':
            prereq_effect = prereq_results.get('prerequisite_effect', 0.0)
            scores.append(min(abs(prereq_effect) * 10, 1.0))  # Scale to 0-1
        
        # Consistency validation score
        if consistency_results.get('consistency_validation') == 'completed':
            overall_consistency = consistency_results.get('overall_consistency', 0.0)
            scores.append(max(overall_consistency, 0.0))
        
        # Category validation score
        if category_results.get('category_validation') == 'completed':
            category_performance = category_results.get('category_performance', {})
            if category_performance:
                avg_correlation = np.mean([
                    perf.get('correlation_with_target', 0.0) 
                    for perf in category_performance.values()
                ])
                scores.append(max(avg_correlation, 0.0))
        
        # Science-specific validation score
        if science_validation:
            science_correlations = [
                val for val in science_validation.values() 
                if isinstance(val, (int, float)) and not np.isnan(val)
            ]
            if science_correlations:
                avg_science_correlation = np.mean(science_correlations)
                scores.append(max(avg_science_correlation, 0.0))
        
        return np.mean(scores) if scores else 0.0
    
    def get_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of validation results."""
        summary = {
            'subject': results.get('subject'),
            'tier': results.get('tier'),
            'overall_validation_score': results.get('overall_validation_score', 0.0),
            'prerequisite_validation': results.get('prerequisite_validation', {}).get('prerequisite_validation'),
            'cross_subject_consistency': results.get('cross_subject_consistency', {}).get('consistency_validation'),
            'category_validation': results.get('category_performance', {}).get('category_validation')
        }
        
        return summary
