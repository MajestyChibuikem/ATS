"""
Temporal Validation for SSAS ML Models.

This module provides temporal validation strategies for the three-tier model system,
with special focus on critical subjects that require sophisticated time-series analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class TemporalValidator:
    """
    Temporal validation for ML models with time-series awareness.
    
    Implements sophisticated validation strategies for educational data:
    - TimeSeriesSplit for temporal consistency
    - Academic year progression validation
    - Term-based validation
    - Cross-temporal performance analysis
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        """
        Initialize temporal validator.
        
        Args:
            n_splits: Number of splits for TimeSeriesSplit
            test_size: Proportion of data for testing
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(1/test_size))
        
        logger.info(f"Initialized Temporal Validator with {n_splits} splits")
    
    def validate_critical_subjects(self, X: pd.DataFrame, y: pd.Series, 
                                 model, tier_name: str = 'critical') -> Dict[str, Any]:
        """
        Validate critical subjects with sophisticated temporal analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Trained model
            tier_name: Tier name for logging
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {tier_name} tier with temporal split")
        
        # Ensure temporal ordering
        X_sorted, y_sorted = self._sort_by_time(X, y)
        
        # Perform temporal cross-validation
        cv_scores = self._temporal_cross_validation(X_sorted, y_sorted, model)
        
        # Additional temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(X_sorted, y_sorted, model)
        
        # Performance degradation analysis
        degradation_analysis = self._analyze_performance_degradation(cv_scores)
        
        results = {
            'tier': tier_name,
            'cv_scores': cv_scores,
            'temporal_analysis': temporal_analysis,
            'degradation_analysis': degradation_analysis,
            'validation_strategy': 'TimeSeriesSplit',
            'n_splits': self.n_splits
        }
        
        logger.info(f"Temporal validation completed for {tier_name} tier")
        return results
    
    def validate_science_subjects(self, X: pd.DataFrame, y: pd.Series, 
                                model, tier_name: str = 'science') -> Dict[str, Any]:
        """
        Validate science subjects with moderate temporal complexity.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Trained model
            tier_name: Tier name for logging
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {tier_name} tier with temporal split")
        
        # Ensure temporal ordering
        X_sorted, y_sorted = self._sort_by_time(X, y)
        
        # Perform temporal cross-validation with fewer splits
        cv_scores = self._temporal_cross_validation(X_sorted, y_sorted, model, n_splits=3)
        
        # Basic temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(X_sorted, y_sorted, model, simplified=True)
        
        results = {
            'tier': tier_name,
            'cv_scores': cv_scores,
            'temporal_analysis': temporal_analysis,
            'validation_strategy': 'TimeSeriesSplit',
            'n_splits': 3
        }
        
        logger.info(f"Temporal validation completed for {tier_name} tier")
        return results
    
    def validate_arts_subjects(self, X: pd.DataFrame, y: pd.Series, 
                             model, tier_name: str = 'arts') -> Dict[str, Any]:
        """
        Validate arts subjects with simplified temporal analysis.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Trained model
            tier_name: Tier name for logging
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {tier_name} tier with simplified temporal split")
        
        # Ensure temporal ordering
        X_sorted, y_sorted = self._sort_by_time(X, y)
        
        # Simple temporal cross-validation
        cv_scores = self._temporal_cross_validation(X_sorted, y_sorted, model, n_splits=2)
        
        # Minimal temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(X_sorted, y_sorted, model, simplified=True)
        
        results = {
            'tier': tier_name,
            'cv_scores': cv_scores,
            'temporal_analysis': temporal_analysis,
            'validation_strategy': 'TimeSeriesSplit',
            'n_splits': 2
        }
        
        logger.info(f"Temporal validation completed for {tier_name} tier")
        return results
    
    def _sort_by_time(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Sort data by temporal order."""
        # If temporal columns exist, use them for sorting
        temporal_columns = ['academic_year', 'term_progression', 'academic_progression']
        
        sort_columns = []
        for col in temporal_columns:
            if col in X.columns:
                sort_columns.append(col)
        
        if sort_columns:
            # Sort by temporal columns
            sorted_indices = X[sort_columns].sort_values(sort_columns).index
            X_sorted = X.loc[sorted_indices].reset_index(drop=True)
            y_sorted = y.loc[sorted_indices].reset_index(drop=True)
        else:
            # No temporal columns, assume data is already ordered
            X_sorted = X.copy()
            y_sorted = y.copy()
        
        return X_sorted, y_sorted
    
    def _temporal_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                 model, n_splits: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Perform temporal cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Model to validate
            n_splits: Number of splits (overrides default)
            
        Returns:
            Dictionary with validation scores
        """
        if n_splits is None:
            n_splits = self.n_splits
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        mse_scores = []
        mae_scores = []
        r2_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model on temporal training set
            model.fit(X_train, y_train)
            
            # Predict on temporal test set
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)
        
        return {
            'mse': mse_scores,
            'mae': mae_scores,
            'r2': r2_scores,
            'mse_mean': np.mean(mse_scores),
            'mae_mean': np.mean(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'mse_std': np.std(mse_scores),
            'mae_std': np.std(mae_scores),
            'r2_std': np.std(r2_scores)
        }
    
    def _analyze_temporal_patterns(self, X: pd.DataFrame, y: pd.Series, 
                                 model, simplified: bool = False) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the data and model performance.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Trained model
            simplified: Whether to use simplified analysis
            
        Returns:
            Dictionary with temporal analysis results
        """
        analysis = {}
        
        if simplified:
            # Simplified temporal analysis
            analysis['temporal_trend'] = self._calculate_simple_trend(y)
            analysis['seasonality_detected'] = False
        else:
            # Advanced temporal analysis for critical subjects
            analysis['temporal_trend'] = self._calculate_advanced_trend(X, y)
            analysis['seasonality_detected'] = self._detect_seasonality(y)
            analysis['performance_stability'] = self._calculate_performance_stability(y)
            analysis['learning_curve'] = self._analyze_learning_curve(X, y, model)
        
        return analysis
    
    def _calculate_simple_trend(self, y: pd.Series) -> Dict[str, float]:
        """Calculate simple linear trend."""
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        
        return {
            'slope': slope,
            'intercept': intercept,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
            'trend_strength': abs(slope)
        }
    
    def _calculate_advanced_trend(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Calculate advanced temporal trends."""
        # Multiple trend analysis
        trends = {}
        
        # Overall trend
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        trends['overall'] = {
            'slope': slope,
            'intercept': intercept,
            'direction': 'increasing' if slope > 0 else 'decreasing'
        }
        
        # Academic progression trend
        if 'academic_progression' in X.columns:
            prog_trend = np.polyfit(X['academic_progression'], y, 1)
            trends['academic_progression'] = {
                'slope': prog_trend[0],
                'direction': 'improving' if prog_trend[0] > 0 else 'declining'
            }
        
        # Term progression trend
        if 'term_progression' in X.columns:
            term_trend = np.polyfit(X['term_progression'], y, 1)
            trends['term_progression'] = {
                'slope': term_trend[0],
                'direction': 'improving' if term_trend[0] > 0 else 'declining'
            }
        
        return trends
    
    def _detect_seasonality(self, y: pd.Series) -> bool:
        """Detect seasonality in performance data."""
        # Simple seasonality detection using autocorrelation
        if len(y) < 6:
            return False
        
        # Calculate autocorrelation
        autocorr = np.corrcoef(y[:-1], y[1:])[0, 1]
        
        # Consider seasonal if autocorrelation is significant
        return abs(autocorr) > 0.3
    
    def _calculate_performance_stability(self, y: pd.Series) -> Dict[str, float]:
        """Calculate performance stability metrics."""
        # Rolling standard deviation
        rolling_std = y.rolling(window=3, min_periods=1).std()
        
        return {
            'mean_stability': 1 / (1 + rolling_std.mean()),
            'stability_trend': np.polyfit(np.arange(len(rolling_std)), rolling_std, 1)[0],
            'volatility': rolling_std.std()
        }
    
    def _analyze_learning_curve(self, X: pd.DataFrame, y: pd.Series, model) -> Dict[str, Any]:
        """Analyze learning curve patterns."""
        # Simulate learning curve with different training set sizes
        train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
        train_scores = []
        test_scores = []
        
        for size in train_sizes:
            split_idx = int(len(X) * size)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if len(X_train) > 0 and len(X_test) > 0:
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                train_scores.append(train_score)
                test_scores.append(test_score)
        
        # Only calculate learning efficiency if we have enough data points
        learning_efficiency = 0.0
        if len(test_scores) >= 2:
            try:
                learning_efficiency = np.polyfit(train_sizes[:len(test_scores)], test_scores, 1)[0]
            except (ValueError, TypeError):
                learning_efficiency = 0.0
        
        return {
            'train_scores': train_scores,
            'test_scores': test_scores,
            'learning_efficiency': learning_efficiency,
            'overfitting_risk': np.mean(train_scores) - np.mean(test_scores) if len(test_scores) > 0 else 0
        }
    
    def _analyze_performance_degradation(self, cv_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze performance degradation over time."""
        r2_scores = cv_scores['r2']
        
        if len(r2_scores) < 2:
            return {'degradation_detected': False}
        
        # Calculate degradation trend
        degradation_trend = np.polyfit(range(len(r2_scores)), r2_scores, 1)[0]
        
        # Detect significant degradation
        degradation_detected = degradation_trend < -0.05
        
        return {
            'degradation_detected': degradation_detected,
            'degradation_rate': degradation_trend,
            'performance_trend': 'declining' if degradation_trend < 0 else 'stable',
            'last_performance': r2_scores[-1] if r2_scores else 0
        }
    
    def get_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of validation results."""
        cv_scores = results['cv_scores']
        
        summary = {
            'tier': results['tier'],
            'validation_strategy': results['validation_strategy'],
            'n_splits': results['n_splits'],
            'mean_r2': cv_scores['r2_mean'],
            'mean_mae': cv_scores['mae_mean'],
            'mean_mse': cv_scores['mse_mean'],
            'r2_std': cv_scores['r2_std'],
            'performance_stability': cv_scores['r2_std'] < 0.1,
            'temporal_consistency': results.get('temporal_analysis', {}),
            'degradation_analysis': results.get('degradation_analysis', {})
        }
        
        return summary
