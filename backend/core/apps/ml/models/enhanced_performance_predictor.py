"""
Enhanced Performance Predictor with Three-Tier Model System.

This module implements the sophisticated three-tier model architecture for student performance prediction:
- Tier 1: Critical Subjects (Mathematics & English) - Ensemble methods with highest complexity
- Tier 2: Science Subjects (Physics, Chemistry, Biology) - Moderate complexity with subject-specific weightings
- Tier 3: Arts/Social Science Subjects - Simplified models with broader feature sets

Based on model_specs.md requirements for WAEC-focused prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

from core.apps.students.models import Student, StudentScore, Teacher, TeacherPerformance
from core.apps.ml.utils.privacy_audit_logger import log_privacy_event

logger = logging.getLogger(__name__)


class EnhancedPerformancePredictor:
    """
    Enhanced Performance Predictor with Three-Tier Model System.
    
    Implements sophisticated prediction models based on subject criticality:
    - Critical subjects (Math/English): Ensemble methods with highest complexity
    - Science subjects: Moderate complexity with subject-specific features
    - Arts subjects: Simplified models with broader feature sets
    """
    
    def __init__(self, model_version="v2.0", epsilon=1.0):
        """
        Initialize the enhanced performance predictor.
        
        Args:
            model_version: Version identifier for the model
            epsilon: Differential privacy parameter
        """
        self.model_version = model_version
        self.epsilon = epsilon
        self.privacy_budget_used = 0.0
        self.privacy_violations = 0
        self.query_count = 0
        
        # Three-tier model architecture
        self.tier_1_models = {}  # Critical subjects (Math, English)
        self.tier_2_models = {}  # Science subjects
        self.tier_3_models = {}  # Arts subjects
        
        # Feature engineering components
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
        # Model performance tracking
        self.model_performance = {}
        self.confidence_calibration = {}
        
        # Subject categorization
        self.critical_subjects = ['Mathematics', 'English Language', 'Further Mathematics']
        self.science_subjects = ['Physics', 'Chemistry', 'Biology', 'Agricultural Science']
        self.arts_subjects = ['Government', 'Economics', 'History', 'Literature', 
                             'Geography', 'Christian Religious Studies', 'Civic Education']
        
        # Model storage paths
        self.model_dir = f'media/ml_models/enhanced_predictor_{model_version}'
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Enhanced Performance Predictor v{model_version} initialized")
    
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
        Prepare training data with enhanced feature engineering.
        
        Returns:
            Dictionary of DataFrames organized by subject tier
        """
        logger.info("Preparing enhanced training data...")
        
        # Fetch all student scores with teacher information
        scores = StudentScore.objects.select_related(
            'student', 'subject', 'teacher', 'academic_year'
        ).all()
        
        # Convert to DataFrame
        data = []
        for score in scores:
            data.append({
                'student_id': score.student.student_id,
                'subject_name': score.subject.name,
                'total_score': float(score.total_score),
                'continuous_assessment': float(score.continuous_assessment),
                'examination_score': float(score.examination_score),
                'class_average': float(score.class_average),
                'grade': score.grade,
                'academic_year': score.academic_year.year,
                'term': score.academic_year.term,
                'student_class': score.student.current_class,
                'student_stream': score.student.stream,
                'student_gender': score.student.gender,
                'student_age': score.student.age,
                'teacher_id': score.teacher.teacher_id if score.teacher else None,
                'teacher_experience': score.teacher.years_experience if score.teacher else 0,
                'teacher_qualification': score.teacher.qualification_level if score.teacher else 'Unknown',
                'teacher_specialization': score.teacher.specialization if score.teacher else 'Unknown',
                'teacher_performance_rating': float(score.teacher.performance_rating) if score.teacher else 0.0,
                'teacher_teaching_load': score.teacher.teaching_load if score.teacher else 0,
                'teacher_years_at_school': score.teacher.years_at_school if score.teacher else 0,
            })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            logger.warning("No training data available")
            return {}
        
        # Enhanced feature engineering
        df = self._engineer_features(df)
        
        # Organize by subject tier
        tier_data = {'critical': [], 'science': [], 'arts': []}
        
        for subject_name in df['subject_name'].unique():
            tier = self.categorize_subject(subject_name)
            subject_data = df[df['subject_name'] == subject_name].copy()
            tier_data[tier].append(subject_data)
        
        # Combine data for each tier
        result = {}
        for tier, data_list in tier_data.items():
            if data_list:
                result[tier] = pd.concat(data_list, ignore_index=True)
                logger.info(f"Tier {tier}: {len(result[tier])} records for {len(data_list)} subjects")
        
        return result
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features for ML models.
        
        Args:
            df: Raw training data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering advanced features...")
        
        # Student-level features
        df['student_performance_avg'] = df.groupby('student_id')['total_score'].transform('mean')
        df['student_performance_std'] = df.groupby('student_id')['total_score'].transform('std')
        df['student_subject_count'] = df.groupby('student_id')['subject_name'].transform('nunique')
        
        # Subject-level features
        df['subject_difficulty'] = df.groupby('subject_name')['total_score'].transform('mean')
        df['subject_performance_std'] = df.groupby('subject_name')['total_score'].transform('std')
        
        # Teacher quality features
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
            lambda row: self._calculate_specialization_alignment(row), axis=1
        )
        
        # Temporal features
        df['term_progression'] = df['term'].map({'First Term': 1, 'Second Term': 2, 'Third Term': 3})
        df['academic_progression'] = df['student_class'].map({'SS1': 1, 'SS2': 2, 'SS3': 3})
        
        # Performance trend features
        df['score_trend'] = df.groupby(['student_id', 'subject_name'])['total_score'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # Class context features
        df['class_performance_rank'] = df.groupby(['subject_name', 'academic_year', 'term'])['total_score'].rank(pct=True)
        df['performance_vs_class_avg'] = df['total_score'] - df['class_average']
        
        # Stream-specific features
        df['stream_performance_avg'] = df.groupby(['student_stream', 'subject_name'])['total_score'].transform('mean')
        
        # Fill missing values
        df = df.fillna(0)
        
        logger.info(f"Engineered {len(df.columns)} features")
        return df
    
    def _calculate_specialization_alignment(self, row: pd.Series) -> float:
        """
        Calculate alignment between teacher specialization and subject.
        
        Args:
            row: DataFrame row with teacher and subject information
            
        Returns:
            Alignment score (0-1)
        """
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
            return 0.5  # Default alignment for other subjects
    
    def train_tier_1_model(self, subject_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Tier 1 model for critical subjects (Mathematics & English).
        
        Uses ensemble methods with highest complexity as per model specs.
        
        Args:
            subject_name: Name of the subject
            data: Training data for the subject
            
        Returns:
            Dictionary containing trained models and metadata
        """
        logger.info(f"Training Tier 1 model for {subject_name}")
        
        # Prepare features for critical subjects
        feature_columns = [
            'continuous_assessment', 'examination_score', 'class_average',
            'student_performance_avg', 'student_performance_std', 'student_subject_count',
            'subject_difficulty', 'subject_performance_std', 'teacher_quality_score',
            'qualification_weight', 'specialization_alignment', 'term_progression',
            'academic_progression', 'score_trend', 'class_performance_rank',
            'performance_vs_class_avg', 'stream_performance_avg',
            'teacher_experience', 'teacher_performance_rating', 'teacher_teaching_load'
        ]
        
        X = data[feature_columns].copy()
        y = data['total_score']
        
        # Handle categorical variables
        categorical_features = ['student_stream', 'student_gender', 'teacher_qualification', 'teacher_specialization']
        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(data[feature].astype(str))
                self.label_encoders[f"{subject_name}_{feature}"] = le
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[subject_name] = scaler
        
        # Ensemble approach for critical subjects
        models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150, max_depth=10, random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
            )
        }
        
        # Train each model
        trained_models = {}
        model_scores = {}
        
        for name, model in models.items():
            # Time series cross-validation for critical subjects
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            
            # Train final model
            model.fit(X_scaled, y)
            trained_models[name] = model
            model_scores[name] = {
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std(),
                'r2': r2_score(y, model.predict(X_scaled))
            }
            
            logger.info(f"{subject_name} - {name}: RMSE={rmse_scores.mean():.2f}±{rmse_scores.std():.2f}, R²={model_scores[name]['r2']:.3f}")
        
        # Feature importance for interpretability
        feature_importance = {}
        for name, model in trained_models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = dict(zip(feature_columns, model.feature_importances_))
        
        self.feature_importance[subject_name] = feature_importance
        
        return {
            'models': trained_models,
            'scores': model_scores,
            'feature_importance': feature_importance,
            'feature_columns': feature_columns,
            'scaler': scaler,
            'label_encoders': {k: v for k, v in self.label_encoders.items() if k.startswith(subject_name)}
        }
    
    def train_tier_2_model(self, subject_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Tier 2 model for science subjects.
        
        Moderate complexity with subject-specific weightings and prerequisite relationships.
        
        Args:
            subject_name: Name of the subject
            data: Training data for the subject
            
        Returns:
            Dictionary containing trained model and metadata
        """
        logger.info(f"Training Tier 2 model for {subject_name}")
        
        # Science-specific features with prerequisite relationships
        feature_columns = [
            'continuous_assessment', 'examination_score', 'class_average',
            'student_performance_avg', 'student_performance_std',
            'teacher_quality_score', 'qualification_weight', 'specialization_alignment',
            'term_progression', 'academic_progression', 'score_trend',
            'class_performance_rank', 'performance_vs_class_avg',
            'teacher_experience', 'teacher_performance_rating'
        ]
        
        # Add prerequisite subject performance for sciences
        if subject_name == 'Physics':
            # Physics depends on Mathematics
            math_scores = self._get_prerequisite_scores(data, 'Mathematics')
            if math_scores is not None:
                feature_columns.append('math_prerequisite')
                data['math_prerequisite'] = math_scores
        
        X = data[feature_columns].copy()
        y = data['total_score']
        
        # Handle categorical variables
        categorical_features = ['student_stream', 'student_gender', 'teacher_qualification']
        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(data[feature].astype(str))
                self.label_encoders[f"{subject_name}_{feature}"] = le
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[subject_name] = scaler
        
        # Single best-performing algorithm for science subjects
        model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        )
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        
        # Train final model
        model.fit(X_scaled, y)
        
        model_score = {
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'r2': r2_score(y, model.predict(X_scaled))
        }
        
        logger.info(f"{subject_name}: RMSE={rmse_scores.mean():.2f}±{rmse_scores.std():.2f}, R²={model_score['r2']:.3f}")
        
        return {
            'model': model,
            'score': model_score,
            'feature_columns': feature_columns,
            'scaler': scaler,
            'label_encoders': {k: v for k, v in self.label_encoders.items() if k.startswith(subject_name)}
        }
    
    def train_tier_3_model(self, subject_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Tier 3 model for arts/social science subjects.
        
        Simplified models with broader feature sets and computational efficiency focus.
        
        Args:
            subject_name: Name of the subject
            data: Training data for the subject
            
        Returns:
            Dictionary containing trained model and metadata
        """
        logger.info(f"Training Tier 3 model for {subject_name}")
        
        # Broader feature set for arts subjects
        feature_columns = [
            'continuous_assessment', 'examination_score', 'class_average',
            'student_performance_avg', 'student_performance_std',
            'teacher_quality_score', 'qualification_weight',
            'term_progression', 'academic_progression',
            'class_performance_rank', 'performance_vs_class_avg',
            'teacher_experience', 'teacher_performance_rating'
        ]
        
        X = data[feature_columns].copy()
        y = data['total_score']
        
        # Handle categorical variables
        categorical_features = ['student_stream', 'student_gender', 'teacher_qualification']
        for feature in categorical_features:
            if feature in data.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(data[feature].astype(str))
                self.label_encoders[f"{subject_name}_{feature}"] = le
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[subject_name] = scaler
        
        # Simplified Random Forest for computational efficiency
        model = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42
        )
        
        # Simple cross-validation
        scores = cross_val_score(model, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        
        # Train final model
        model.fit(X_scaled, y)
        
        model_score = {
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'r2': r2_score(y, model.predict(X_scaled))
        }
        
        logger.info(f"{subject_name}: RMSE={rmse_scores.mean():.2f}±{rmse_scores.std():.2f}, R²={model_score['r2']:.3f}")
        
        return {
            'model': model,
            'score': model_score,
            'feature_columns': feature_columns,
            'scaler': scaler,
            'label_encoders': {k: v for k, v in self.label_encoders.items() if k.startswith(subject_name)}
        }
    
    def _get_prerequisite_scores(self, data: pd.DataFrame, prerequisite_subject: str) -> Optional[pd.Series]:
        """
        Get prerequisite subject scores for students.
        
        Args:
            data: Current subject data
            prerequisite_subject: Name of prerequisite subject
            
        Returns:
            Series of prerequisite scores or None if not available
        """
        try:
            # Get prerequisite scores for the same students
            student_ids = data['student_id'].unique()
            prerequisite_scores = StudentScore.objects.filter(
                student__student_id__in=student_ids,
                subject__name=prerequisite_subject
            ).values('student__student_id', 'total_score')
            
            if prerequisite_scores:
                prereq_df = pd.DataFrame(prerequisite_scores)
                prereq_df = prereq_df.groupby('student__student_id')['total_score'].mean()
                return data['student_id'].map(prereq_df)
            
        except Exception as e:
            logger.warning(f"Could not get prerequisite scores for {prerequisite_subject}: {e}")
        
        return None
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all models for all subjects using the three-tier system.
        
        Returns:
            Dictionary containing training results and performance metrics
        """
        logger.info("Starting comprehensive model training with three-tier system...")
        
        # Prepare training data
        tier_data = self.prepare_training_data()
        
        if not tier_data:
            logger.error("No training data available")
            return {}
        
        training_results = {
            'tier_1_models': {},
            'tier_2_models': {},
            'tier_3_models': {},
            'overall_performance': {},
            'training_summary': {}
        }
        
        # Train Tier 1 models (Critical subjects)
        if 'critical' in tier_data:
            for subject_name in tier_data['critical']['subject_name'].unique():
                subject_data = tier_data['critical'][tier_data['critical']['subject_name'] == subject_name]
                if len(subject_data) >= 50:  # Minimum data requirement
                    result = self.train_tier_1_model(subject_name, subject_data)
                    self.tier_1_models[subject_name] = result
                    training_results['tier_1_models'][subject_name] = result['scores']
        
        # Train Tier 2 models (Science subjects)
        if 'science' in tier_data:
            for subject_name in tier_data['science']['subject_name'].unique():
                subject_data = tier_data['science'][tier_data['science']['subject_name'] == subject_name]
                if len(subject_data) >= 30:  # Minimum data requirement
                    result = self.train_tier_2_model(subject_name, subject_data)
                    self.tier_2_models[subject_name] = result
                    training_results['tier_2_models'][subject_name] = result['score']
        
        # Train Tier 3 models (Arts subjects)
        if 'arts' in tier_data:
            for subject_name in tier_data['arts']['subject_name'].unique():
                subject_data = tier_data['arts'][tier_data['arts']['subject_name'] == subject_name]
                if len(subject_data) >= 20:  # Minimum data requirement
                    result = self.train_tier_3_model(subject_name, subject_data)
                    self.tier_3_models[subject_name] = result
                    training_results['tier_3_models'][subject_name] = result['score']
        
        # Calculate overall performance
        all_scores = []
        for tier_results in training_results.values():
            if isinstance(tier_results, dict):
                for subject_results in tier_results.values():
                    if isinstance(subject_results, dict):
                        if 'rmse_mean' in subject_results:
                            all_scores.append(subject_results['rmse_mean'])
                        elif isinstance(subject_results, dict):
                            for model_score in subject_results.values():
                                if isinstance(model_score, dict) and 'rmse_mean' in model_score:
                                    all_scores.append(model_score['rmse_mean'])
        
        if all_scores:
            training_results['overall_performance'] = {
                'mean_rmse': np.mean(all_scores),
                'std_rmse': np.std(all_scores),
                'total_subjects': len(all_scores)
            }
        
        # Save models
        self.save_models()
        
        # Log privacy event
        log_privacy_event(
            module_name="enhanced_performance_predictor",
            student_id="system_training",
            privacy_params={
                "event_type": "model_training",
                "description": "Enhanced Performance Predictor training completed",
                "data_accessed": "student_scores,teacher_data",
                "privacy_budget_used": 0.0,
                "epsilon": self.epsilon
            }
        )
        
        logger.info(f"Training completed. Overall RMSE: {training_results.get('overall_performance', {}).get('mean_rmse', 'N/A')}")
        
        return training_results
    
    def predict(self, student_id: str, subject_name: str, 
                current_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Make prediction for a student in a specific subject.
        
        Args:
            student_id: Student identifier
            subject_name: Subject name
            current_scores: Current scores for feature engineering
            
        Returns:
            Dictionary containing prediction results with privacy guarantees
        """
        try:
            # Get student data
            student = Student.objects.get(student_id=student_id)
            
            # Determine subject tier
            tier = self.categorize_subject(subject_name)
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(student, subject_name, current_scores)
            
            if features is None:
                return self._fallback_prediction(student_id, subject_name)
            
            # Make prediction based on tier
            if tier == 'critical' and subject_name in self.tier_1_models:
                prediction = self._predict_tier_1(subject_name, features)
            elif tier == 'science' and subject_name in self.tier_2_models:
                prediction = self._predict_tier_2(subject_name, features)
            elif tier == 'arts' and subject_name in self.tier_3_models:
                prediction = self._predict_tier_3(subject_name, features)
            else:
                return self._fallback_prediction(student_id, subject_name)
            
            # Apply differential privacy
            prediction = self._apply_differential_privacy_to_predictions(prediction)
            
            # Add metadata
            prediction.update({
                'student_id': student_id,
                'subject_name': subject_name,
                'tier': tier,
                'model_version': self.model_version,
                'prediction_timestamp': datetime.now().isoformat(),
                'privacy_guarantees': {
                    'differential_privacy': True,
                    'epsilon': self.epsilon,
                    'privacy_budget_used': self.privacy_budget_used,
                    'noise_added': True
                },
                'privacy_compliant': True
            })
            
            # Log prediction
            self._log_prediction(student_id, subject_name, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error for {student_id} in {subject_name}: {e}")
            return self._fallback_prediction(student_id, subject_name)
    
    def _prepare_prediction_features(self, student: Student, subject_name: str, 
                                   current_scores: Dict[str, float] = None) -> Optional[pd.DataFrame]:
        """
        Prepare features for prediction.
        
        Args:
            student: Student object
            subject_name: Subject name
            current_scores: Current scores for feature engineering
            
        Returns:
            DataFrame with features or None if insufficient data
        """
        try:
            # Get student's historical scores
            scores = StudentScore.objects.filter(
                student=student
            ).select_related('subject', 'teacher', 'academic_year')
            
            if not scores.exists():
                return None
            
            # Convert to DataFrame
            data = []
            for score in scores:
                data.append({
                    'student_id': student.student_id,
                    'subject_name': score.subject.name,
                    'total_score': float(score.total_score),
                    'continuous_assessment': float(score.continuous_assessment),
                    'examination_score': float(score.examination_score),
                    'class_average': float(score.class_average),
                    'academic_year': score.academic_year.year,
                    'term': score.academic_year.term,
                    'student_class': student.current_class,
                    'student_stream': student.stream,
                    'student_gender': student.gender,
                    'teacher_experience': score.teacher.years_experience if score.teacher else 0,
                    'teacher_performance_rating': float(score.teacher.performance_rating) if score.teacher else 0.0,
                    'teacher_qualification': score.teacher.qualification_level if score.teacher else 'Unknown',
                    'teacher_specialization': score.teacher.specialization if score.teacher else 'Unknown',
                    'teacher_teaching_load': score.teacher.teaching_load if score.teacher else 0,
                    'teacher_years_at_school': score.teacher.years_at_school if score.teacher else 0,
                })
            
            df = pd.DataFrame(data)
            
            # Engineer features
            df = self._engineer_features(df)
            
            # Get features for the specific subject
            subject_data = df[df['subject_name'] == subject_name]
            
            if subject_data.empty:
                return None
            
            # Use the most recent data point
            latest_data = subject_data.iloc[-1]
            
            # Create feature vector
            tier = self.categorize_subject(subject_name)
            
            if tier == 'critical':
                feature_columns = [
                    'continuous_assessment', 'examination_score', 'class_average',
                    'student_performance_avg', 'student_performance_std', 'student_subject_count',
                    'subject_difficulty', 'subject_performance_std', 'teacher_quality_score',
                    'qualification_weight', 'specialization_alignment', 'term_progression',
                    'academic_progression', 'score_trend', 'class_performance_rank',
                    'performance_vs_class_avg', 'stream_performance_avg',
                    'teacher_experience', 'teacher_performance_rating', 'teacher_teaching_load'
                ]
            elif tier == 'science':
                feature_columns = [
                    'continuous_assessment', 'examination_score', 'class_average',
                    'student_performance_avg', 'student_performance_std',
                    'teacher_quality_score', 'qualification_weight', 'specialization_alignment',
                    'term_progression', 'academic_progression', 'score_trend',
                    'class_performance_rank', 'performance_vs_class_avg',
                    'teacher_experience', 'teacher_performance_rating'
                ]
            else:  # arts
                feature_columns = [
                    'continuous_assessment', 'examination_score', 'class_average',
                    'student_performance_avg', 'student_performance_std',
                    'teacher_quality_score', 'qualification_weight',
                    'term_progression', 'academic_progression',
                    'class_performance_rank', 'performance_vs_class_avg',
                    'teacher_experience', 'teacher_performance_rating'
                ]
            
            features = latest_data[feature_columns].fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return None
    
    def _predict_tier_1(self, subject_name: str, features: pd.Series) -> Dict[str, Any]:
        """Make prediction using Tier 1 ensemble model."""
        model_data = self.tier_1_models[subject_name]
        
        # Prepare features
        X = features.values.reshape(1, -1)
        X_scaled = model_data['scaler'].transform(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in model_data['models'].items():
            pred = model.predict(X_scaled)[0]
            predictions[name] = max(0, min(100, pred))  # Clamp to 0-100
        
        # Ensemble prediction (weighted average)
        weights = {
            'gradient_boosting': 0.4,
            'random_forest': 0.35,
            'neural_network': 0.25
        }
        
        ensemble_prediction = sum(predictions[name] * weights[name] for name in predictions)
        
        # Calculate confidence based on model agreement
        prediction_std = np.std(list(predictions.values()))
        confidence = max(0.1, 1.0 - (prediction_std / 20.0))  # Higher agreement = higher confidence
        
        return {
            'predicted_score': round(ensemble_prediction, 2),
            'confidence': round(confidence, 3),
            'model_predictions': predictions,
            'model_agreement': round(1.0 - (prediction_std / 20.0), 3),
            'tier': 'critical',
            'ensemble_method': 'weighted_average'
        }
    
    def _predict_tier_2(self, subject_name: str, features: pd.Series) -> Dict[str, Any]:
        """Make prediction using Tier 2 model."""
        model_data = self.tier_2_models[subject_name]
        
        # Prepare features
        X = features.values.reshape(1, -1)
        X_scaled = model_data['scaler'].transform(X)
        
        # Make prediction
        prediction = model_data['model'].predict(X_scaled)[0]
        prediction = max(0, min(100, prediction))  # Clamp to 0-100
        
        # Calculate confidence based on model performance
        confidence = max(0.1, min(0.9, model_data['score']['r2']))
        
        return {
            'predicted_score': round(prediction, 2),
            'confidence': round(confidence, 3),
            'tier': 'science',
            'model_type': 'gradient_boosting'
        }
    
    def _predict_tier_3(self, subject_name: str, features: pd.Series) -> Dict[str, Any]:
        """Make prediction using Tier 3 model."""
        model_data = self.tier_3_models[subject_name]
        
        # Prepare features
        X = features.values.reshape(1, -1)
        X_scaled = model_data['scaler'].transform(X)
        
        # Make prediction
        prediction = model_data['model'].predict(X_scaled)[0]
        prediction = max(0, min(100, prediction))  # Clamp to 0-100
        
        # Calculate confidence based on model performance
        confidence = max(0.1, min(0.8, model_data['score']['r2']))
        
        return {
            'predicted_score': round(prediction, 2),
            'confidence': round(confidence, 3),
            'tier': 'arts',
            'model_type': 'random_forest'
        }
    
    def _apply_differential_privacy_to_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply differential privacy noise to prediction results.
        
        Args:
            predictions: Dictionary containing prediction results
            
        Returns:
            Privacy-protected predictions
        """
        private_predictions = predictions.copy()
        
        # Add noise to predicted score
        if 'predicted_score' in predictions:
            noisy_score = self._add_privacy_noise(predictions['predicted_score'], sensitivity=100.0)
            private_predictions['predicted_score'] = round(max(0, min(100, noisy_score)), 2)
        
        # Add noise to confidence
        if 'confidence' in predictions:
            noisy_confidence = self._add_privacy_noise(predictions['confidence'], sensitivity=1.0)
            private_predictions['confidence'] = round(max(0.1, min(1.0, noisy_confidence)), 3)
        
        # Update privacy budget
        self.privacy_budget_used += self.epsilon
        self.query_count += 1
        
        return private_predictions
    
    def _add_privacy_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            value: Original value
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value
        """
        import numpy as np
        
        # Laplace noise with scale = sensitivity / epsilon
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        
        return value + noise
    
    def _fallback_prediction(self, student_id: str, subject_name: str) -> Dict[str, Any]:
        """Provide fallback prediction when models are unavailable."""
        return {
            'student_id': student_id,
            'subject_name': subject_name,
            'predicted_score': 70.0,  # Default prediction
            'confidence': 0.3,  # Low confidence
            'tier': self.categorize_subject(subject_name),
            'model_version': self.model_version,
            'fallback_used': True,
            'privacy_guarantees': {
                'differential_privacy': True,
                'epsilon': self.epsilon,
                'privacy_budget_used': 0.0,
                'noise_added': False
            },
            'privacy_compliant': True
        }
    
    def _log_prediction(self, student_id: str, subject_name: str, prediction: Dict[str, Any]):
        """Log prediction for audit trail."""
        logger.info(f"Prediction: {student_id} - {subject_name} - Score: {prediction.get('predicted_score')} - Confidence: {prediction.get('confidence')}")
        
        # Log privacy event
        log_privacy_event(
            module_name="enhanced_performance_predictor",
            student_id=student_id,
            privacy_params={
                "event_type": "prediction",
                "description": f"Performance prediction for {student_id} in {subject_name}",
                "data_accessed": "student_scores,teacher_data",
                "privacy_budget_used": prediction.get('privacy_guarantees', {}).get('privacy_budget_used', 0.0),
                "epsilon": self.epsilon
            }
        )
    
    def save_models(self):
        """Save all trained models to disk."""
        try:
            # Save Tier 1 models
            for subject_name, model_data in self.tier_1_models.items():
                model_path = os.path.join(self.model_dir, f'tier1_{subject_name.lower().replace(" ", "_")}.joblib')
                joblib.dump(model_data, model_path)
            
            # Save Tier 2 models
            for subject_name, model_data in self.tier_2_models.items():
                model_path = os.path.join(self.model_dir, f'tier2_{subject_name.lower().replace(" ", "_")}.joblib')
                joblib.dump(model_data, model_path)
            
            # Save Tier 3 models
            for subject_name, model_data in self.tier_3_models.items():
                model_path = os.path.join(self.model_dir, f'tier3_{subject_name.lower().replace(" ", "_")}.joblib')
                joblib.dump(model_data, model_path)
            
            logger.info(f"All models saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load all trained models from disk."""
        try:
            # Load Tier 1 models
            for subject_name in self.critical_subjects:
                model_path = os.path.join(self.model_dir, f'tier1_{subject_name.lower().replace(" ", "_")}.joblib')
                if os.path.exists(model_path):
                    self.tier_1_models[subject_name] = joblib.load(model_path)
            
            # Load Tier 2 models
            for subject_name in self.science_subjects:
                model_path = os.path.join(self.model_dir, f'tier2_{subject_name.lower().replace(" ", "_")}.joblib')
                if os.path.exists(model_path):
                    self.tier_2_models[subject_name] = joblib.load(model_path)
            
            # Load Tier 3 models
            for subject_name in self.arts_subjects:
                model_path = os.path.join(self.model_dir, f'tier3_{subject_name.lower().replace(" ", "_")}.joblib')
                if os.path.exists(model_path):
                    self.tier_3_models[subject_name] = joblib.load(model_path)
            
            logger.info(f"Loaded {len(self.tier_1_models)} Tier 1, {len(self.tier_2_models)} Tier 2, {len(self.tier_3_models)} Tier 3 models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get comprehensive model health information."""
        health_info = {
            'model_version': self.model_version,
            'tier_1_models': len(self.tier_1_models),
            'tier_2_models': len(self.tier_2_models),
            'tier_3_models': len(self.tier_3_models),
            'total_models': len(self.tier_1_models) + len(self.tier_2_models) + len(self.tier_3_models),
            'privacy_settings': {
                'epsilon': self.epsilon,
                'privacy_budget_used': self.privacy_budget_used,
                'differential_privacy': True
            },
            'query_count': self.query_count,
            'model_performance': self.model_performance,
            'feature_importance': {k: len(v) for k, v in self.feature_importance.items()},
            'status': 'healthy' if (len(self.tier_1_models) + len(self.tier_2_models) + len(self.tier_3_models)) > 0 else 'no_models'
        }
        
        return health_info


if __name__ == "__main__":
    # Initialize and train the enhanced performance predictor
    predictor = EnhancedPerformancePredictor(model_version="v2.0", epsilon=1.0)
    
    # Train all models
    results = predictor.train_all_models()
    
    print("Enhanced Performance Predictor Training Results:")
    print(f"Tier 1 Models: {len(results.get('tier_1_models', {}))}")
    print(f"Tier 2 Models: {len(results.get('tier_2_models', {}))}")
    print(f"Tier 3 Models: {len(results.get('tier_3_models', {}))}")
    
    if 'overall_performance' in results:
        overall = results['overall_performance']
        print(f"Overall RMSE: {overall.get('mean_rmse', 'N/A'):.2f} ± {overall.get('std_rmse', 'N/A'):.2f}")
        print(f"Total Subjects: {overall.get('total_subjects', 'N/A')}")
    
    print("Training completed successfully!")
