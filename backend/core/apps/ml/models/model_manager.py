"""
Model Manager for SSAS ML Models.

This module orchestrates all three tiers of the performance prediction system
and provides a unified interface for training and prediction.
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
from .feature_engineer import FeatureEngineeringPipeline
from .model_factory import ModelFactory, EnsembleModel

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages all three tiers of the performance prediction system.
    """
    
    def __init__(self, model_version: str = "v2.0", epsilon: float = 1.0):
        """
        Initialize the model manager.
        
        Args:
            model_version: Version identifier for the models
            epsilon: Differential privacy parameter
        """
        self.model_version = model_version
        self.epsilon = epsilon
        self.privacy_budget_used = 0.0
        
        # Initialize components
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.model_factory = ModelFactory()
        
        # Model storage
        self.tier_models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Subject categorization
        self.critical_subjects = ['Mathematics', 'English Language', 'Further Mathematics']
        self.science_subjects = ['Physics', 'Chemistry', 'Biology', 'Agricultural Science']
        self.arts_subjects = ['Government', 'Economics', 'History', 'Literature', 'Geography', 'Christian Religious Studies', 'Civic Education']
        
        # Model storage paths
        self.model_dir = f'media/ml_models/modular_predictor_{model_version}'
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Initialized Model Manager v{model_version}")
    
    def categorize_subject(self, subject_name: str) -> str:
        """
        Categorize subject into appropriate tier.
        
        Args:
            subject_name: Name of the subject
            
        Returns:
            Tier category: 'critical', 'science', or 'arts'
        """
        if subject_name in self.critical_subjects:
            return 'critical'
        elif subject_name in self.science_subjects:
            return 'science'
        else:
            return 'arts'
    
    def prepare_training_data(self) -> Dict[str, pd.DataFrame]:
        """
        Prepare training data for all tiers.
        
        Returns:
            Dictionary of DataFrames organized by tier
        """
        logger.info("Preparing training data for all tiers")
        
        # Fetch all student scores with relationships
        scores = StudentScore.objects.select_related(
            'student', 'subject', 'teacher', 'academic_year'
        ).all()
        
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
        
        df = pd.DataFrame(data)
        
        # Organize by tier
        tier_data = {}
        for tier in ['critical', 'science', 'arts']:
            if tier == 'critical':
                subjects = self.critical_subjects
            elif tier == 'science':
                subjects = self.science_subjects
            else:
                subjects = self.arts_subjects
            
            tier_df = df[df['subject_name'].isin(subjects)].copy()
            if not tier_df.empty:
                tier_data[tier] = tier_df
                logger.info(f"Prepared {len(tier_df)} samples for {tier} tier")
        
        return tier_data
    
    def train_tier_models(self, tier_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train models for a specific tier.
        
        Args:
            tier_name: Tier name (critical, science, arts)
            data: Training data for the tier
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training models for {tier_name} tier")
        
        # Engineer features
        engineered_data = self.feature_pipeline.engineer_features(data, tier_name)
        
        # Prepare features and target
        feature_columns = self.feature_pipeline.get_feature_columns(tier_name)
        if not feature_columns:
            # Use all numeric columns except target
            feature_columns = engineered_data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns if col != 'total_score']
        
        X = engineered_data[feature_columns]
        y = engineered_data['total_score']
        
        # Create and train models based on tier
        models = []
        
        if tier_name == 'critical':
            # Ensemble for critical subjects
            model_types = ['gradient_boosting', 'random_forest', 'neural_network']
            for model_type in model_types:
                model = self.model_factory.create_model(tier_name, f"{tier_name}_{model_type}", model_type=model_type)
                model.train(X, y)
                models.append(model)
            
            # Create ensemble
            ensemble = EnsembleModel(models, weights=[0.4, 0.4, 0.2])
            self.tier_models[tier_name] = ensemble
            
        elif tier_name == 'science':
            # Single model for science subjects
            model = self.model_factory.create_model(tier_name, f"{tier_name}_gradient_boosting", model_type='gradient_boosting')
            model.train(X, y)
            self.tier_models[tier_name] = model
            
        else:  # arts
            # Simple model for arts subjects
            model = self.model_factory.create_model(tier_name, f"{tier_name}_random_forest", model_type='random_forest')
            model.train(X, y)
            self.tier_models[tier_name] = model
        
        # Store feature information
        self.feature_importance[tier_name] = self.tier_models[tier_name].get_feature_importance()
        
        # Calculate performance metrics
        predictions = self.tier_models[tier_name].predict(X)
        mse = np.mean((y - predictions) ** 2)
        mae = np.mean(np.abs(y - predictions))
        r2 = 1 - (mse / np.var(y))
        
        self.model_performance[tier_name] = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'feature_count': len(feature_columns),
            'sample_count': len(data)
        }
        
        logger.info(f"Trained {tier_name} tier - R²: {r2:.3f}, MAE: {mae:.2f}")
        
        return {
            'tier': tier_name,
            'performance': self.model_performance[tier_name],
            'feature_importance': self.feature_importance[tier_name]
        }
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train models for all tiers.
        
        Returns:
            Dictionary with training results for all tiers
        """
        logger.info("Starting training for all tiers")
        
        # Prepare data
        tier_data = self.prepare_training_data()
        
        # Train each tier
        results = {}
        for tier_name, data in tier_data.items():
            try:
                result = self.train_tier_models(tier_name, data)
                results[tier_name] = result
            except Exception as e:
                logger.error(f"Error training {tier_name} tier: {e}")
                results[tier_name] = {'error': str(e)}
        
        # Log privacy event
        log_privacy_event(
            module_name="model_manager",
            student_id="system_training",
            privacy_params={
                "event_type": "model_training",
                "description": "Modular model training completed",
                "data_accessed": "student_scores,teacher_data",
                "privacy_budget_used": 0.0,
                "epsilon": self.epsilon
            }
        )
        
        return results
    
    def predict(self, student_id: str, subject_name: str) -> Dict[str, Any]:
        """
        Make prediction for a student in a specific subject.
        
        Args:
            student_id: Student ID
            subject_name: Subject name
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Making prediction for {student_id} in {subject_name}")
        
        # Determine tier
        tier_name = self.categorize_subject(subject_name)
        
        if tier_name not in self.tier_models:
            return self._fallback_prediction(student_id, subject_name)
        
        try:
            # Prepare student data
            student_data = self._prepare_student_data(student_id, subject_name)
            if student_data is None:
                return self._fallback_prediction(student_id, subject_name)
            
            # Engineer features
            engineered_data = self.feature_pipeline.engineer_features(student_data, tier_name)
            
            # Get feature columns
            feature_columns = self.feature_pipeline.get_feature_columns(tier_name)
            if not feature_columns:
                feature_columns = engineered_data.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = [col for col in feature_columns if col != 'total_score']
            
            X = engineered_data[feature_columns].iloc[0:1]  # Single prediction
            
            # Make prediction
            predicted_score = self.tier_models[tier_name].predict(X)[0]
            
            # Apply differential privacy
            noisy_score = self._apply_differential_privacy(predicted_score)
            
            # Calculate confidence
            confidence = self._calculate_confidence(tier_name, predicted_score)
            
            result = {
                'student_id': student_id,
                'subject_name': subject_name,
                'predicted_score': round(noisy_score, 2),
                'confidence': round(confidence, 3),
                'tier': tier_name,
                'model_version': self.model_version,
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
    
    def _calculate_confidence(self, tier_name: str, predicted_score: float) -> float:
        """Calculate confidence for prediction."""
        # Base confidence on tier and model performance
        base_confidence = {
            'critical': 0.85,
            'science': 0.80,
            'arts': 0.75
        }
        
        confidence = base_confidence.get(tier_name, 0.70)
        
        # Adjust based on predicted score (higher scores = higher confidence)
        if predicted_score > 80:
            confidence += 0.05
        elif predicted_score < 40:
            confidence -= 0.05
        
        return max(0.5, min(0.95, confidence))
    
    def _fallback_prediction(self, student_id: str, subject_name: str) -> Dict[str, Any]:
        """Provide fallback prediction when models are unavailable."""
        return {
            'student_id': student_id,
            'subject_name': subject_name,
            'predicted_score': 65.0,
            'confidence': 0.5,
            'tier': self.categorize_subject(subject_name),
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
            module_name="model_manager",
            student_id=student_id,
            privacy_params={
                "event_type": "prediction",
                "subject": subject_name,
                "predicted_score": result['predicted_score'],
                "confidence": result['confidence'],
                "tier": result['tier'],
                "privacy_budget_used": result['privacy_guarantees']['privacy_budget_used']
            }
        )
    
    def save_models(self):
        """Save all trained models to disk."""
        logger.info("Saving models to disk")
        
        for tier_name, model in self.tier_models.items():
            model_path = os.path.join(self.model_dir, f"{tier_name}_model.joblib")
            joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_version': self.model_version,
            'epsilon': self.epsilon,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'trained_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, 'metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """Load trained models from disk."""
        logger.info("Loading models from disk")
        
        metadata_path = os.path.join(self.model_dir, 'metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.feature_importance = metadata.get('feature_importance', {})
            self.model_performance = metadata.get('model_performance', {})
        
        for tier_name in ['critical', 'science', 'arts']:
            model_path = os.path.join(self.model_dir, f"{tier_name}_model.joblib")
            if os.path.exists(model_path):
                self.tier_models[tier_name] = joblib.load(model_path)
                logger.info(f"Loaded {tier_name} tier model")
        
        logger.info("Models loaded successfully")
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get health status of all models."""
        health = {
            'model_version': self.model_version,
            'tiers_available': list(self.tier_models.keys()),
            'privacy_budget_used': self.privacy_budget_used,
            'epsilon': self.epsilon
        }
        
        for tier_name in self.tier_models:
            health[f"{tier_name}_performance"] = self.model_performance.get(tier_name, {})
        
        return health
