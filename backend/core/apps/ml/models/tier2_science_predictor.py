"""
Tier 2 Science Predictor for SSAS ML Models.

This module implements moderate complexity prediction models for science subjects
(Physics, Chemistry, Biology, Agricultural Science) with prerequisite awareness.
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
from core.apps.ml.models.model_factory import ModelFactory
from core.apps.ml.utils.feature_engineering.science_features import ScienceFeaturesEngineer
from core.apps.ml.utils.validation.temporal_validator import TemporalValidator
from core.apps.ml.utils.validation.cross_subject_validator import CrossSubjectValidator

logger = logging.getLogger(__name__)


class Tier2SciencePredictor:
    """
    Tier 2 Science Predictor for science subjects.
    
    Implements moderate complexity models with prerequisite awareness:
    - Gradient Boosting for primary predictions
    - Prerequisite-aware feature engineering
    - Cross-subject validation
    - Laboratory performance integration
    """
    
    def __init__(self, model_version: str = "v2.0", epsilon: float = 1.0):
        """
        Initialize the science tier predictor.
        
        Args:
            model_version: Version identifier for the model
            epsilon: Differential privacy parameter
        """
        self.model_version = model_version
        self.epsilon = epsilon
        self.privacy_budget_used = 0.0
        
        # Initialize components
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.science_features = ScienceFeaturesEngineer()
        self.model_factory = ModelFactory()
        self.temporal_validator = TemporalValidator(n_splits=3)
        self.cross_subject_validator = CrossSubjectValidator()
        
        # Add science features to pipeline
        self.feature_pipeline.add_tier_engineer('science', self.science_features)
        
        # Model storage
        self.models = {}
        self.feature_importance = {}
        self.validation_results = {}
        self.model_performance = {}
        
        # Science subjects
        self.science_subjects = ['Physics', 'Chemistry', 'Biology', 'Agricultural Science']
        
        # Model storage paths
        self.model_dir = f'media/ml_models/tier2_science_{model_version}'
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Initialized Tier 2 Science Predictor v{model_version}")
    
    def prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare training data for science subjects.
        
        Returns:
            DataFrame with training data for science subjects
        """
        logger.info("Preparing training data for science subjects")
        
        # Fetch student scores for science subjects
        scores = StudentScore.objects.select_related(
            'student', 'subject', 'teacher', 'academic_year'
        ).filter(subject__name__in=self.science_subjects)
        
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
        logger.info(f"Prepared {len(df)} samples for science subjects")
        
        return df
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train models for science subjects.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training models for science subjects")
        
        # Prepare data
        data = self.prepare_training_data()
        if data.empty:
            raise ValueError("No training data available for science subjects")
        
        # Engineer features
        engineered_data = self.feature_pipeline.engineer_features(data, 'science')
        
        # Prepare features and target
        feature_columns = self.science_features.get_feature_columns()
        X = engineered_data[feature_columns]
        y = engineered_data['total_score']
        
        # Train models for each science subject
        results = {}
        
        for subject in self.science_subjects:
            logger.info(f"Training model for {subject}")
            
            # Filter data for this subject
            subject_mask = engineered_data['subject_name'] == subject
            X_subject = X[subject_mask]
            y_subject = y[subject_mask]
            
            if len(X_subject) == 0:
                logger.warning(f"No data available for {subject}")
                continue
            
            # Create and train model
            model = self.model_factory.create_model('science', f"science_{subject.lower().replace(' ', '_')}", model_type='gradient_boosting')
            model.train(X_subject, y_subject)
            
            # Validate model
            temporal_result = self.temporal_validator.validate_science_subjects(X_subject, y_subject, model.model)
            cross_subject_result = self.cross_subject_validator.validate_science_subjects(X_subject, y_subject, model.model, subject)
            
            # Store model and results
            self.models[subject] = model
            self.validation_results[subject] = {
                'temporal': temporal_result,
                'cross_subject': cross_subject_result
            }
            
            # Calculate performance metrics
            predictions = model.predict(X_subject)
            mse = np.mean((y_subject - predictions) ** 2)
            mae = np.mean(np.abs(y_subject - predictions))
            r2 = 1 - (mse / np.var(y_subject))
            
            self.model_performance[subject] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'feature_count': len(feature_columns),
                'sample_count': len(X_subject)
            }
            
            # Store feature importance
            self.feature_importance[subject] = model.get_feature_importance()
            
            results[subject] = {
                'performance': self.model_performance[subject],
                'validation': self.validation_results[subject],
                'feature_importance': self.feature_importance[subject]
            }
            
            logger.info(f"Trained {subject} model - R²: {r2:.3f}, MAE: {mae:.2f}")
        
        # Log privacy event
        log_privacy_event(
            module_name="tier2_science_predictor",
            student_id="system_training",
            privacy_params={
                "event_type": "model_training",
                "description": "Tier 2 Science Predictor training completed",
                "data_accessed": "science_subject_scores,teacher_data",
                "privacy_budget_used": 0.0,
                "epsilon": self.epsilon,
                "subjects_trained": len(self.models),
                "feature_count": len(feature_columns)
            }
        )
        
        return results
    
    def predict(self, student_id: str, subject_name: str) -> Dict[str, Any]:
        """
        Make prediction for a student in a science subject.
        
        Args:
            student_id: Student ID
            subject_name: Subject name (must be science subject)
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Making prediction for {student_id} in {subject_name}")
        
        # Validate subject
        if subject_name not in self.science_subjects:
            raise ValueError(f"Subject {subject_name} is not a science subject")
        
        if subject_name not in self.models:
            return self._fallback_prediction(student_id, subject_name)
        
        try:
            # Prepare student data
            student_data = self._prepare_student_data(student_id, subject_name)
            if student_data is None:
                return self._fallback_prediction(student_id, subject_name)
            
            # Engineer features
            engineered_data = self.feature_pipeline.engineer_features(student_data, 'science')
            
            # Get feature columns
            feature_columns = self.science_features.get_feature_columns()
            X = engineered_data[feature_columns].iloc[0:1]  # Single prediction
            
            # Make prediction
            predicted_score = self.models[subject_name].predict(X)[0]
            
            # Apply differential privacy
            noisy_score = self._apply_differential_privacy(predicted_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(subject_name, predicted_score)
            
            # Validate prerequisites
            prereq_validation = self.science_features.validate_prerequisites(student_id, subject_name, student_data)
            
            # Get feature importance for this subject
            feature_importance = self.feature_importance.get(subject_name, {})
            
            result = {
                'student_id': student_id,
                'subject_name': subject_name,
                'predicted_score': round(noisy_score, 2),
                'confidence': round(confidence, 3),
                'tier': 'science',
                'model_version': self.model_version,
                'prerequisite_validation': prereq_validation,
                'feature_importance': dict(list(feature_importance.items())[:10]),  # Top 10 features
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
            
            # Convert to DataFrame (convert Decimal to float for ML compatibility)
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
    
    def _calculate_confidence(self, subject_name: str, predicted_score: float) -> float:
        """Calculate confidence for prediction."""
        # Base confidence for science subjects
        base_confidence = 0.80
        
        # Adjust based on predicted score
        if predicted_score > 80:
            confidence = base_confidence + 0.05
        elif predicted_score < 40:
            confidence = base_confidence - 0.05
        else:
            confidence = base_confidence
        
        # Adjust based on model performance
        if subject_name in self.model_performance:
            subject_r2 = self.model_performance[subject_name]['r2']
            confidence *= (0.8 + 0.2 * subject_r2)  # Scale by model performance
        
        # Adjust based on prerequisite validation
        prereqs = self.science_features.get_prerequisites(subject_name)
        if prereqs:
            confidence *= 0.95  # Slightly lower confidence due to prerequisite dependencies
        
        return max(0.5, min(0.95, confidence))
    
    def _fallback_prediction(self, student_id: str, subject_name: str) -> Dict[str, Any]:
        """Provide fallback prediction when models are unavailable."""
        return {
            'student_id': student_id,
            'subject_name': subject_name,
            'predicted_score': 68.0,
            'confidence': 0.5,
            'tier': 'science',
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
            module_name="tier2_science_predictor",
            student_id=student_id,
            privacy_params={
                "event_type": "prediction",
                "subject": subject_name,
                "predicted_score": result['predicted_score'],
                "confidence": result['confidence'],
                "tier": result['tier'],
                "prerequisite_validation": result.get('prerequisite_validation', {}),
                "privacy_budget_used": result['privacy_guarantees']['privacy_budget_used']
            }
        )
    
    def save_models(self):
        """Save trained models to disk."""
        logger.info("Saving Tier 2 Science models to disk")
        
        # Save individual subject models
        for subject_name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f'{subject_name.lower().replace(" ", "_")}_model.joblib')
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
        
        logger.info(f"Tier 2 Science models saved to {self.model_dir}")
    
    def load_models(self):
        """Load trained models from disk."""
        logger.info("Loading Tier 2 Science models from disk")
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, 'metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.feature_importance = metadata.get('feature_importance', {})
            self.model_performance = metadata.get('model_performance', {})
            self.validation_results = metadata.get('validation_results', {})
        
        # Load individual subject models
        for subject in self.science_subjects:
            model_path = os.path.join(self.model_dir, f'{subject.lower().replace(" ", "_")}_model.joblib')
            if os.path.exists(model_path):
                self.models[subject] = joblib.load(model_path)
                logger.info(f"Loaded {subject} model")
        
        logger.info("Tier 2 Science models loaded successfully")
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get health status of the science tier models."""
        health = {
            'tier': 'science',
            'model_version': self.model_version,
            'models_available': len(self.models),
            'subjects_covered': list(self.models.keys()),
            'privacy_budget_used': self.privacy_budget_used,
            'epsilon': self.epsilon
        }
        
        if self.model_performance:
            health['overall_performance'] = {
                'avg_r2': np.mean([perf['r2'] for perf in self.model_performance.values()]),
                'avg_mae': np.mean([perf['mae'] for perf in self.model_performance.values()]),
                'total_samples': sum([perf['sample_count'] for perf in self.model_performance.values()])
            }
        
        return health
    
    def get_prerequisites(self, subject_name: str) -> List[str]:
        """Get prerequisite subjects for a science subject."""
        return self.science_features.get_prerequisites(subject_name)
    
    def validate_prerequisites(self, student_id: str, subject_name: str) -> Dict[str, Any]:
        """Validate if a student meets prerequisites for a science subject."""
        if subject_name not in self.science_subjects:
            return {'meets_prerequisites': False, 'error': 'Not a science subject'}
        
        # Get student data
        student_data = self._prepare_student_data(student_id, subject_name)
        if student_data is None:
            return {'meets_prerequisites': False, 'error': 'No student data available'}
        
        return self.science_features.validate_prerequisites(student_id, subject_name, student_data)
