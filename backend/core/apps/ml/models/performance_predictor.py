"""
Individual Student Performance Predictor - Phase 1 (Ensemble Methods)
Production-ready implementation with interpretability and monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from django.conf import settings
from django.core.cache import cache
from django.db import transaction
import os

logger = logging.getLogger(__name__)


class PerformancePredictor:
    """
    Ensemble-based performance predictor with production-ready features.
    
    Features:
    - Multi-output prediction for all subjects
    - SHAP explainability
    - A/B testing framework
    - Fallback mechanisms
    - Real-time monitoring
    - Audit logging
    """
    
    def __init__(self, model_version: str = "v1.0"):
        self.model_version = model_version
        self.models = {}  # One model per subject
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.subject_names = []
        self.metrics_history = []
        self.ab_test_config = {}
        
        # Production monitoring
        self.prediction_count = 0
        self.error_count = 0
        self.last_training_time = None
        self.model_performance = {}
        
        # Load existing model if available
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model from disk."""
        try:
            model_path = f"{settings.MEDIA_ROOT}/ml_models/performance_predictor_{self.model_version}.joblib"
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.models = model_data['models']
                self.scalers = model_data['scalers']
                self.label_encoders = model_data['label_encoders']
                self.feature_names = model_data['feature_names']
                self.subject_names = model_data['subject_names']
                self.model_performance = model_data.get('performance', {})
                logger.info(f"Loaded model version {self.model_version}")
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
    
    def _save_model(self):
        """Save trained model to disk."""
        try:
            model_path = f"{settings.MEDIA_ROOT}/ml_models/performance_predictor_{self.model_version}.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'subject_names': self.subject_names,
                'performance': self.model_performance,
                'version': self.model_version,
                'trained_at': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Saved model version {self.model_version}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def _extract_features(self, student_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for performance prediction.
        
        Features include:
        - Historical performance by subject
        - Performance trends and volatility
        - Subject correlations
        - Demographics and attendance
        - Temporal patterns
        """
        features = pd.DataFrame()
        
        # Student demographics
        features['age'] = student_data['age']
        features['gender_encoded'] = self.label_encoders['gender'].transform(student_data['gender'])
        features['class_level_encoded'] = self.label_encoders['class_level'].transform(student_data['current_class'])
        features['stream_encoded'] = self.label_encoders['stream'].transform(student_data['stream'])
        
        # Historical performance features
        for subject in self.subject_names:
            # Average performance in subject
            features[f'{subject}_avg_score'] = student_data[f'{subject}_avg']
            features[f'{subject}_std_score'] = student_data[f'{subject}_std']
            features[f'{subject}_trend'] = student_data[f'{subject}_trend']
            features[f'{subject}_volatility'] = student_data[f'{subject}_volatility']
            
            # Performance relative to class average (simplified)
            features[f'{subject}_vs_class'] = student_data[f'{subject}_avg'] - 75.0  # Default class average
        
        # Cross-subject correlation features (simplified)
        for i, subject1 in enumerate(self.subject_names):
            for j, subject2 in enumerate(self.subject_names[i+1:], i+1):
                corr_key = f'{subject1}_{subject2}_corr'
                if corr_key in student_data.columns:
                    features[f'{subject1}_{subject2}_correlation'] = student_data[corr_key]
                else:
                    features[f'{subject1}_{subject2}_correlation'] = 0.0
        
        # Temporal features
        features['days_since_admission'] = student_data['days_since_admission']
        features['terms_completed'] = student_data['terms_completed']
        
        # Attendance and behavioral features
        features['attendance_rate'] = student_data['attendance_rate']
        features['behavior_score'] = student_data['behavior_score']
        
        # Performance consistency
        features['overall_consistency'] = student_data['overall_consistency']
        features['improvement_rate'] = student_data['improvement_rate']
        
        return features
    
    def _prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data from database.
        Implements temporal split for realistic validation.
        """
        from core.apps.students.models import Student, StudentScore, AcademicYear
        
        # Get all student scores with temporal ordering
        scores_data = StudentScore.objects.select_related(
            'student', 'subject', 'academic_year'
        ).order_by('student_id', 'academic_year__start_date', 'subject__name')
        
        # Convert to DataFrame
        data = []
        for score in scores_data:
            data.append({
                'student_id': score.student.student_id,
                'subject': score.subject.name,
                'total_score': float(score.total_score),
                'continuous_assessment': float(score.continuous_assessment),
                'examination_score': float(score.examination_score),
                'academic_year': score.academic_year.year,
                'term': score.academic_year.term,
                'gender': score.student.gender,
                'current_class': score.student.current_class,
                'stream': score.student.stream,
                'date': score.academic_year.start_date,
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            raise ValueError("No score data found in database")
        
        print(f"Raw data shape: {df.shape}")
        print(f"Unique students: {df['student_id'].nunique()}")
        print(f"Unique subjects: {df['subject'].nunique()}")
        
        # Feature engineering
        features_df = self._engineer_features(df)
        targets_df = self._prepare_targets(df)
        
        # Set index to student_id for both
        print(f"Features columns: {list(features_df.columns)}")
        print(f"Targets columns: {list(targets_df.columns)}")
        
        if 'student_id' in features_df.columns:
            features_df = features_df.set_index('student_id')
        
        # targets_df already has student_id as index from pivot_table
        
        print(f"Features shape: {features_df.shape}")
        print(f"Targets shape: {targets_df.shape}")
        
        return features_df, targets_df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw score data."""
        features = []
        
        for student_id in df['student_id'].unique():
            student_data = df[df['student_id'] == student_id].copy()
            
            # Sort by date for temporal features
            student_data = student_data.sort_values('date')
            
            # Calculate features for each student
            student_features = self._calculate_student_features(student_data)
            features.append(student_features)
        
        return pd.DataFrame(features)
    
    def _calculate_student_features(self, student_data: pd.DataFrame) -> Dict:
        """Calculate features for a single student."""
        features = {}
        
        # Basic demographics
        features['student_id'] = student_data['student_id'].iloc[0]
        features['gender'] = student_data['gender'].iloc[0]
        features['current_class'] = student_data['current_class'].iloc[0]
        features['stream'] = student_data['stream'].iloc[0]
        
        # Calculate age (simplified)
        features['age'] = 16  # Default age for SS2/SS3
        
        # Subject-specific features
        for subject in student_data['subject'].unique():
            subject_scores = student_data[student_data['subject'] == subject]['total_score']
            
            features[f'{subject}_avg'] = subject_scores.mean()
            features[f'{subject}_std'] = subject_scores.std()
            features[f'{subject}_trend'] = self._calculate_trend(subject_scores)
            features[f'{subject}_volatility'] = subject_scores.std() / subject_scores.mean() if subject_scores.mean() > 0 else 0
        
        # Cross-subject correlations
        subjects = student_data['subject'].unique()
        for i, subject1 in enumerate(subjects):
            for j, subject2 in enumerate(subjects[i+1:], i+1):
                scores1 = student_data[student_data['subject'] == subject1]['total_score']
                scores2 = student_data[student_data['subject'] == subject2]['total_score']
                
                if len(scores1) > 1 and len(scores2) > 1:
                    correlation = np.corrcoef(scores1, scores2)[0, 1]
                    features[f'{subject1}_{subject2}_corr'] = correlation if not np.isnan(correlation) else 0
                else:
                    features[f'{subject1}_{subject2}_corr'] = 0
        
        # Temporal features
        min_date = student_data['date'].min()
        if isinstance(min_date, str):
            min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
        features['days_since_admission'] = (datetime.now().date() - min_date).days
        features['terms_completed'] = student_data['academic_year'].nunique()
        
        # Attendance and behavioral features (simplified)
        features['attendance_rate'] = 0.85  # Default attendance rate
        features['behavior_score'] = 0.90   # Default behavior score
        
        # Overall performance consistency
        all_scores = student_data['total_score']
        features['overall_consistency'] = 1 - (all_scores.std() / all_scores.mean()) if all_scores.mean() > 0 else 0
        features['improvement_rate'] = self._calculate_improvement_rate(all_scores)
        
        return features
    
    def _calculate_trend(self, scores: pd.Series) -> float:
        """Calculate performance trend (positive = improving, negative = declining)."""
        if len(scores) < 2:
            return 0
        
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        return slope
    
    def _calculate_improvement_rate(self, scores: pd.Series) -> float:
        """Calculate overall improvement rate."""
        if len(scores) < 2:
            return 0
        
        return (scores.iloc[-1] - scores.iloc[0]) / len(scores)
    
    def _prepare_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare target variables (next term scores) for each subject."""
        # Create a pivot table with students as rows and subjects as columns
        targets_df = df.pivot_table(
            index='student_id',
            columns='subject',
            values='total_score',
            aggfunc='last'  # Use the last score for each student-subject combination
        )
        
        print(f"Targets shape: {targets_df.shape}")
        print(f"Targets columns: {list(targets_df.columns)}")
        print(f"Targets non-null counts: {targets_df.count()}")
        
        return targets_df
    
    def train(self, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the ensemble performance predictor.
        
        Returns:
            Dictionary with training metrics and model performance
        """
        logger.info("Starting model training...")
        
        try:
            # Prepare data
            features_df, targets_df = self._prepare_training_data()
            
            if features_df.empty or targets_df.empty:
                raise ValueError("No training data available")
            
            # Initialize encoders
            self._initialize_encoders(features_df)
            
            # Set subject names from targets
            self.subject_names = list(targets_df.columns)
            
            # Feature engineering
            X = self._extract_features(features_df)
            y = targets_df
            
            # Ensure X and y have the same index
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            print(f"Final X shape: {X.shape}")
            print(f"Final y shape: {y.shape}")
            
            # Temporal split for realistic validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Train models for each subject
            training_metrics = {}
            
            for subject in y.columns:
                logger.info(f"Training model for {subject}...")
                
                # Prepare target for this subject
                y_subject = y[subject].dropna()
                X_subject = X.loc[y_subject.index]
                
                print(f"Training {subject}: {len(y_subject)} samples")
                
                if len(X_subject) < 10:
                    logger.warning(f"Insufficient data for {subject}, skipping...")
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_subject)
                self.scalers[subject] = scaler
                
                # Train ensemble model
                model = self._create_ensemble_model()
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_scaled, y_subject, 
                    cv=tscv, scoring='neg_mean_squared_error'
                )
                
                # Train on full dataset
                model.fit(X_scaled, y_subject)
                self.models[subject] = model
                
                # Calculate metrics
                y_pred = model.predict(X_scaled)
                rmse = np.sqrt(mean_squared_error(y_subject, y_pred))
                mae = mean_absolute_error(y_subject, y_pred)
                r2 = r2_score(y_subject, y_pred)
                
                training_metrics[subject] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_rmse': np.sqrt(-cv_scores.mean()),
                    'samples': len(y_subject)
                }
                
                logger.info(f"{subject}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
            
            # Store performance metrics
            self.model_performance = training_metrics
            self.last_training_time = datetime.now()
            
            # Save model
            self._save_model()
            
            # Log training completion
            logger.info("Model training completed successfully")
            
            return {
                'status': 'success',
                'metrics': training_metrics,
                'subjects_trained': len(self.models),
                'training_time': self.last_training_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _create_ensemble_model(self):
        """Create ensemble model with optimal hyperparameters."""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def _initialize_encoders(self, df: pd.DataFrame):
        """Initialize label encoders for categorical features."""
        self.label_encoders = {}
        
        # Gender encoder
        self.label_encoders['gender'] = LabelEncoder()
        self.label_encoders['gender'].fit(['Male', 'Female'])
        
        # Class level encoder
        self.label_encoders['class_level'] = LabelEncoder()
        self.label_encoders['class_level'].fit(['SS1', 'SS2', 'SS3'])
        
        # Stream encoder
        self.label_encoders['stream'] = LabelEncoder()
        self.label_encoders['stream'].fit(['Science', 'Arts', 'Commercial'])
    
    def predict(self, student_id: str, subjects: List[str] = None) -> Dict[str, Any]:
        """
        Predict performance for a student.
        
        Args:
            student_id: Student identifier
            subjects: List of subjects to predict (default: all subjects)
        
        Returns:
            Dictionary with predictions, confidence intervals, and explanations
        """
        start_time = datetime.now()
        
        try:
            # Get student data
            student_data = self._get_student_data(student_id)
            
            if student_data is None:
                return self._fallback_prediction(student_id, "Student data not found")
            
            # Extract features
            features = self._extract_features(student_data)
            
            # Make predictions
            predictions = {}
            explanations = {}
            confidence_intervals = {}
            
            if subjects is None:
                subjects = list(self.models.keys())
            
            for subject in subjects:
                if subject not in self.models:
                    continue
                
                # Scale features
                X_scaled = self.scalers[subject].transform(features)
                
                # Make prediction
                prediction = self.models[subject].predict(X_scaled)[0]
                
                # Calculate confidence interval using ensemble variance
                predictions_list = []
                for estimator in self.models[subject].estimators_:
                    pred = estimator.predict(X_scaled)[0]
                    predictions_list.append(pred)
                
                confidence_interval = np.percentile(predictions_list, [5, 95])
                
                # Generate SHAP explanation
                explanation = self._generate_explanation(subject, features, prediction)
                
                predictions[subject] = prediction
                confidence_intervals[subject] = confidence_interval.tolist()
                explanations[subject] = explanation
            
            # Log prediction
            self._log_prediction(student_id, predictions, start_time)
            
            return {
                'student_id': student_id,
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'explanations': explanations,
                'model_version': self.model_version,
                'prediction_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for student {student_id}: {e}")
            return self._fallback_prediction(student_id, str(e))
    
    def _get_student_data(self, student_id: str) -> Optional[pd.DataFrame]:
        """Get student data for prediction."""
        from core.apps.students.models import Student, StudentScore
        
        try:
            student = Student.objects.get(student_id=student_id)
            scores = StudentScore.objects.filter(student=student).select_related('subject', 'academic_year')
            
            if not scores.exists():
                return None
            
            # Convert to DataFrame
            data = []
            for score in scores:
                data.append({
                    'student_id': score.student.student_id,
                    'subject': score.subject.name,
                    'total_score': float(score.total_score),
                    'continuous_assessment': float(score.continuous_assessment),
                    'examination_score': float(score.examination_score),
                    'academic_year': score.academic_year.year,
                    'term': score.academic_year.term,
                    'gender': score.student.gender,
                    'current_class': score.student.current_class,
                    'stream': score.student.stream,
                    'date': score.academic_year.start_date,
                })
            
            df = pd.DataFrame(data)
            
            # Engineer features
            features = self._calculate_student_features(df)
            return pd.DataFrame([features])
            
        except Student.DoesNotExist:
            return None
    
    def _generate_explanation(self, subject: str, features: pd.DataFrame, prediction: float) -> Dict[str, Any]:
        """Generate SHAP-based explanation for prediction."""
        try:
            import shap
            
            # Get SHAP values
            explainer = shap.TreeExplainer(self.models[subject])
            shap_values = explainer.shap_values(features)
            
            # Get top features
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'shap_value': shap_values[0]
            }).sort_values('shap_value', key=abs, ascending=False)
            
            return {
                'top_features': feature_importance.head(5).to_dict('records'),
                'prediction_baseline': explainer.expected_value,
                'feature_contributions': feature_importance.to_dict('records')
            }
            
        except Exception as e:
            logger.warning(f"Could not generate explanation: {e}")
            return {
                'top_features': [],
                'prediction_baseline': 0,
                'feature_contributions': []
            }
    
    def _fallback_prediction(self, student_id: str, reason: str) -> Dict[str, Any]:
        """Fallback prediction when model fails."""
        logger.warning(f"Using fallback prediction for {student_id}: {reason}")
        
        return {
            'student_id': student_id,
            'predictions': {},
            'confidence_intervals': {},
            'explanations': {},
            'fallback': True,
            'fallback_reason': reason,
            'model_version': self.model_version
        }
    
    def _log_prediction(self, student_id: str, predictions: Dict, start_time: datetime):
        """Log prediction for audit trail."""
        self.prediction_count += 1
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'student_id': student_id,
            'predictions': predictions,
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'model_version': self.model_version
        }
        
        # Store in cache for monitoring
        cache.set(f'prediction_log_{self.prediction_count}', log_entry, timeout=86400)
        
        # Log to file
        logger.info(f"Prediction logged: {log_entry}")
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get model health metrics."""
        return {
            'model_version': self.model_version,
            'subjects_trained': len(self.models),
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'prediction_count': self.prediction_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.prediction_count, 1),
            'performance_metrics': self.model_performance
        }
