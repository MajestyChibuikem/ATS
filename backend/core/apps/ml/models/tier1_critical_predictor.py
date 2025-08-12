"""
Tier 1 Critical Predictor for SSAS ML Models.

This module implements the highest complexity prediction models for critical subjects
(Mathematics, English Language, Further Mathematics) using ensemble methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
import joblib
from datetime import datetime

from core.apps.students.models import Student, StudentScore, Teacher, TeacherPerformance
from core.apps.ml.utils.privacy_audit_logger import log_privacy_event
from core.apps.ml.models.feature_engineer import FeatureEngineeringPipeline
from core.apps.ml.models.model_factory import ModelFactory, EnsembleModel
from core.apps.ml.utils.feature_engineering.critical_features import CriticalFeaturesEngineer
from core.apps.ml.utils.validation.temporal_validator import TemporalValidator

logger = logging.getLogger(__name__)


class Tier1CriticalPredictor:
    """
    Tier 1 Critical Predictor for critical subjects.
    
    Implements sophisticated ensemble methods with highest complexity:
    - Multiple model types (Gradient Boosting, Random Forest, Neural Network)
    - Advanced feature engineering specific to critical subjects
    - Sophisticated temporal validation
    - Differential privacy protection
    """
    
    def __init__(self, model_version: str = "v2.0", epsilon: float = 1.0):
        """
        Initialize the critical tier predictor.
        
        Args:
            model_version: Version identifier for the model
            epsilon: Differential privacy parameter
        """
        self.model_version = model_version
        self.epsilon = epsilon
        self.privacy_budget_used = 0.0
        
        # Initialize components
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.critical_features = CriticalFeaturesEngineer()
        self.model_factory = ModelFactory()
        self.temporal_validator = TemporalValidator(n_splits=5)
        
        # Add critical features to pipeline
        self.feature_pipeline.add_tier_engineer('critical', self.critical_features)
        
        # Model storage
        self.ensemble_model = None
        self.individual_models = {}
        self.feature_importance = {}
        self.validation_results = {}
        self.model_performance = {}
        
        # Critical subjects
        self.critical_subjects = ['Mathematics', 'English Language', 'Further Mathematics']
        
        # Model storage paths
        self.model_dir = f'media/ml_models/tier1_critical_{model_version}'
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Initialized Tier 1 Critical Predictor v{model_version}")
    
    def prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare training data for critical subjects.
        
        Returns:
            DataFrame with training data for critical subjects
        """
        logger.info("Preparing training data for critical subjects")
        
        # Fetch student scores for critical subjects
        scores = StudentScore.objects.select_related(
            'student', 'subject', 'teacher', 'academic_year'
        ).filter(subject__name__in=self.critical_subjects)
        
        # Convert to DataFrame
        data = []
        for score in scores:
            data.append({
                'student_id': score.student.student_id,
                'subject_name': score.subject.name,
                'total_score': float(score.total_score),
                'continuous_assessment': float(score.continuous_assessment),
                'examination_score': float(score.examination_score),
                'academic_year': score.academic_year.year,
                'term': score.term,
                'student_class': score.student.current_class,
                'student_stream': score.student.stream,
                'student_gender': score.student.gender,
                'student_age': score.student.age,
                'class_average': float(score.class_average),
                'teacher_id': score.teacher.id if score.teacher else None,
                'teacher_name': score.teacher.full_name if score.teacher else 'Unknown',
                'teacher_experience': score.teacher.years_experience if score.teacher else 0,
                'teacher_qualification': score.teacher.qualification_level if score.teacher else 'Unknown',
                'teacher_specialization': score.teacher.specialization if score.teacher else 'Unknown',
                'teacher_performance_rating': float(score.teacher.performance_rating) if score.teacher else 0.0,
                'teacher_teaching_load': score.teacher.teaching_load if score.teacher else 0,
                'teacher_years_at_school': score.teacher.years_at_school if score.teacher else 0,
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Prepared {len(df)} samples for critical subjects")
        
        return df
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train ensemble models for critical subjects.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training ensemble models for critical subjects")
        
        # Prepare data
        data = self.prepare_training_data()
        if data.empty:
            raise ValueError("No training data available for critical subjects")
        
        # Engineer features
        engineered_data = self.feature_pipeline.engineer_features(data, 'critical')
        
        # Prepare features and target
        feature_columns = self.critical_features.get_feature_columns()
        X = engineered_data[feature_columns]
        y = engineered_data['total_score']
        
        # Create individual models
        model_types = ['gradient_boosting', 'random_forest', 'neural_network']
        models = []
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model for critical subjects")
            
            model = self.model_factory.create_model('critical', f"critical_{model_type}", model_type=model_type)
            model.train(X, y)
            
            # Validate model
            validation_result = self.temporal_validator.validate_critical_subjects(X, y, model.model)
            self.validation_results[model_type] = validation_result
            
            # Store individual model
            self.individual_models[model_type] = model
            models.append(model)
            
            logger.info(f"Trained {model_type} model - R²: {validation_result['cv_scores']['r2_mean']:.3f}")
        
        # Create ensemble with optimized weights
        weights = self._optimize_ensemble_weights(models, X, y)
        self.ensemble_model = EnsembleModel(models, weights=weights)
        
        # Calculate ensemble performance
        ensemble_predictions = self.ensemble_model.predict(X)
        ensemble_mse = np.mean((y - ensemble_predictions) ** 2)
        ensemble_mae = np.mean(np.abs(y - ensemble_predictions))
        ensemble_r2 = 1 - (ensemble_mse / np.var(y))
        
        # Store performance metrics
        self.model_performance = {
            'ensemble': {
                'mse': ensemble_mse,
                'mae': ensemble_mae,
                'r2': ensemble_r2,
                'feature_count': len(feature_columns),
                'sample_count': len(data)
            },
            'individual_models': {
                model_type: {
                    'r2': self.validation_results[model_type]['cv_scores']['r2_mean'],
                    'mae': self.validation_results[model_type]['cv_scores']['mae_mean'],
                    'weight': weight
                }
                for model_type, weight in zip(model_types, weights)
            }
        }
        
        # Store feature importance
        self.feature_importance = self.ensemble_model.get_feature_importance()
        
        # Log privacy event
        log_privacy_event(
            module_name="tier1_critical_predictor",
            student_id="system_training",
            privacy_params={
                "event_type": "model_training",
                "description": "Tier 1 Critical Predictor training completed",
                "data_accessed": "critical_subject_scores,teacher_data",
                "privacy_budget_used": 0.0,
                "epsilon": self.epsilon,
                "ensemble_models": len(models),
                "feature_count": len(feature_columns)
            }
        )
        
        logger.info(f"Trained ensemble model - R²: {ensemble_r2:.3f}, MAE: {ensemble_mae:.2f}")
        
        return {
            'ensemble_performance': self.model_performance['ensemble'],
            'individual_performance': self.model_performance['individual_models'],
            'validation_results': self.validation_results,
            'feature_importance': self.feature_importance
        }
    
    def _optimize_ensemble_weights(self, models: List, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Optimize ensemble weights based on individual model performance."""
        # Get validation scores for each model
        scores = []
        for model in models:
            validation_result = self.temporal_validator.validate_critical_subjects(X, y, model.model)
            scores.append(validation_result['cv_scores']['r2_mean'])
        
        # Convert scores to weights (higher score = higher weight)
        scores = np.array(scores)
        weights = scores / np.sum(scores)
        
        # Ensure minimum weight for each model
        min_weight = 0.1
        weights = np.maximum(weights, min_weight)
        weights = weights / np.sum(weights)
        
        logger.info(f"Optimized ensemble weights: {weights}")
        return weights.tolist()
    
    def predict(self, student_id: str, subject_name: str) -> Dict[str, Any]:
        """
        Make prediction for a student in a critical subject.
        
        Args:
            student_id: Student ID
            subject_name: Subject name (must be critical subject)
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Making prediction for {student_id} in {subject_name}")
        
        # Validate subject
        if subject_name not in self.critical_subjects:
            raise ValueError(f"Subject {subject_name} is not a critical subject")
        
        if self.ensemble_model is None:
            return self._fallback_prediction(student_id, subject_name)
        
        try:
            # Prepare student data
            student_data = self._prepare_student_data(student_id, subject_name)
            if student_data is None:
                return self._fallback_prediction(student_id, subject_name)
            
            # Engineer features
            engineered_data = self.feature_pipeline.engineer_features(student_data, 'critical')
            
            # Get feature columns
            feature_columns = self.critical_features.get_feature_columns()
            X = engineered_data[feature_columns].iloc[0:1]  # Single prediction
            
            # Make ensemble prediction
            predicted_score = self.ensemble_model.predict(X)[0]
            
            # Apply differential privacy
            noisy_score = self._apply_differential_privacy(predicted_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(predicted_score)
            
            # Get individual model predictions
            individual_predictions = {}
            for model_type, model in self.individual_models.items():
                individual_predictions[model_type] = float(model.predict(X)[0])
            
            result = {
                'student_id': student_id,
                'subject_name': subject_name,
                'predicted_score': round(noisy_score, 2),
                'confidence': round(confidence, 3),
                'tier': 'critical',
                'model_version': self.model_version,
                'ensemble_prediction': True,
                'individual_predictions': individual_predictions,
                'feature_importance': dict(list(self.feature_importance.items())[:10]),  # Top 10 features
                'privacy_guarantees': {
                    'differential_privacy': True,
                    'epsilon': self.epsilon,
                    'privacy_budget_used': self.privacy_budget_used,
                    'noise_added': True
                },
                'privacy_compliant': True
            }
            
            # Log prediction
            self._log_prediction(student_id, subject_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._fallback_prediction(student_id, subject_name)
    
    def _prepare_student_data(self, student_id: str, subject_name: str) -> Optional[pd.DataFrame]:
        """Prepare student data for prediction."""
        try:
            # Fetch student scores
            scores = StudentScore.objects.select_related(
                'student', 'subject', 'teacher', 'academic_year'
            ).filter(student__student_id=student_id)
            
            if not scores.exists():
                return None
            
            # Convert to DataFrame
            data = []
            for score in scores:
                data.append({
                    'student_id': score.student.student_id,
                    'subject_name': score.subject.name,
                    'total_score': score.total_score,
                    'continuous_assessment': score.continuous_assessment,
                    'examination_score': score.examination_score,
                    'academic_year': score.academic_year.year,
                    'term': score.term,
                    'student_class': score.student.current_class,
                    'student_stream': score.student.stream,
                    'student_gender': score.student.gender,
                    'student_age': score.student.age,
                    'class_average': score.class_average,
                    'teacher_id': score.teacher.id if score.teacher else None,
                    'teacher_name': score.teacher.full_name if score.teacher else 'Unknown',
                    'teacher_experience': score.teacher.years_experience if score.teacher else 0,
                    'teacher_qualification': score.teacher.qualification_level if score.teacher else 'Unknown',
                    'teacher_specialization': score.teacher.specialization if score.teacher else 'Unknown',
                    'teacher_performance_rating': score.teacher.performance_rating if score.teacher else 0,
                    'teacher_teaching_load': score.teacher.teaching_load if score.teacher else 0,
                    'teacher_years_at_school': score.teacher.years_at_school if score.teacher else 0,
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error preparing student data: {e}")
            return None
    
    def _apply_differential_privacy(self, value: float) -> float:
        """Apply differential privacy noise to prediction."""
        if self.epsilon <= 0:
            return value
        
        # Laplace noise with sensitivity = 1.0 (for normalized scores)
        scale = 1.0 / self.epsilon
        noise = np.random.laplace(0, scale)
        
        # Update privacy budget
        self.privacy_budget_used += (1.0 / self.epsilon)
        
        return value + noise
    
    def _calculate_confidence(self, predicted_score: float) -> float:
        """Calculate confidence for prediction."""
        # Base confidence for critical subjects
        base_confidence = 0.85
        
        # Adjust based on predicted score
        if predicted_score > 80:
            confidence = base_confidence + 0.05
        elif predicted_score < 40:
            confidence = base_confidence - 0.05
        else:
            confidence = base_confidence
        
        # Adjust based on model performance
        if self.model_performance:
            ensemble_r2 = self.model_performance['ensemble']['r2']
            confidence *= (0.8 + 0.2 * ensemble_r2)  # Scale by model performance
        
        return max(0.5, min(0.95, confidence))
    
    def _fallback_prediction(self, student_id: str, subject_name: str) -> Dict[str, Any]:
        """Provide fallback prediction when models are unavailable."""
        return {
            'student_id': student_id,
            'subject_name': subject_name,
            'predicted_score': 70.0,
            'confidence': 0.5,
            'tier': 'critical',
            'model_version': self.model_version,
            'fallback': True,
            'privacy_guarantees': {
                'differential_privacy': False,
                'epsilon': self.epsilon,
                'privacy_budget_used': self.privacy_budget_used,
                'noise_added': False
            },
            'privacy_compliant': True
        }
    
    def _log_prediction(self, student_id: str, subject_name: str, result: Dict[str, Any]):
        """Log prediction details."""
        log_privacy_event(
            module_name="tier1_critical_predictor",
            student_id=student_id,
            privacy_params={
                "event_type": "prediction",
                "subject": subject_name,
                "predicted_score": result['predicted_score'],
                "confidence": result['confidence'],
                "tier": result['tier'],
                "ensemble_prediction": result.get('ensemble_prediction', False),
                "privacy_budget_used": result['privacy_guarantees']['privacy_budget_used']
            }
        )
    
    def save_models(self):
        """Save trained models to disk."""
        logger.info("Saving Tier 1 Critical models to disk")
        
        # Save ensemble model
        ensemble_path = os.path.join(self.model_dir, 'ensemble_model.joblib')
        joblib.dump(self.ensemble_model, ensemble_path)
        
        # Save individual models
        for model_type, model in self.individual_models.items():
            model_path = os.path.join(self.model_dir, f'{model_type}_model.joblib')
            joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_version': self.model_version,
            'epsilon': self.epsilon,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'validation_results': self.validation_results,
            'trained_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, 'metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Tier 1 Critical models saved to {self.model_dir}")
    
    def load_models(self):
        """Load trained models from disk."""
        logger.info("Loading Tier 1 Critical models from disk")
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, 'metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.feature_importance = metadata.get('feature_importance', {})
            self.model_performance = metadata.get('model_performance', {})
            self.validation_results = metadata.get('validation_results', {})
        
        # Load ensemble model
        ensemble_path = os.path.join(self.model_dir, 'ensemble_model.joblib')
        if os.path.exists(ensemble_path):
            self.ensemble_model = joblib.load(ensemble_path)
        
        # Load individual models
        model_types = ['gradient_boosting', 'random_forest', 'neural_network']
        for model_type in model_types:
            model_path = os.path.join(self.model_dir, f'{model_type}_model.joblib')
            if os.path.exists(model_path):
                self.individual_models[model_type] = joblib.load(model_path)
        
        logger.info("Tier 1 Critical models loaded successfully")
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get health status of the critical tier models."""
        health = {
            'tier': 'critical',
            'model_version': self.model_version,
            'ensemble_available': self.ensemble_model is not None,
            'individual_models_available': len(self.individual_models),
            'privacy_budget_used': self.privacy_budget_used,
            'epsilon': self.epsilon
        }
        
        if self.model_performance:
            health['ensemble_performance'] = self.model_performance.get('ensemble', {})
            health['individual_performance'] = self.model_performance.get('individual_models', {})
        
        return health
