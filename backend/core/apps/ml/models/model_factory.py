"""
Model Factory for SSAS ML Models.

This module provides a factory pattern for creating appropriate ML models
for each tier of the performance prediction system.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging
import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    """
    
    def __init__(self, model_name: str, tier_name: str):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            tier_name: Tier name (critical, science, arts)
        """
        self.model_name = model_name
        self.tier_name = tier_name
        self.model = None
        self.is_trained = False
        
        logger.info(f"Initialized {model_name} for {tier_name} tier")
    
    @abstractmethod
    def create_model(self, **kwargs) -> Any:
        """Create the specific model instance."""
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass
    
    def train(self, X, y, **kwargs):
        """Train the model."""
        if self.model is None:
            self.model = self.create_model(**kwargs)
        
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"Trained {self.model_name} for {self.tier_name} tier")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))
        return None


class CriticalTierModel(BaseModel):
    """Model for critical subjects (Mathematics, English)."""
    
    def create_model(self, **kwargs) -> Any:
        """Create ensemble model for critical subjects."""
        model_type = kwargs.get('model_type', 'gradient_boosting')
        
        if model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                **kwargs
            )
        elif model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                **kwargs
            )
        elif model_type == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'tier': 'critical',
            'complexity': 'high',
            'ensemble_methods': True,
            'model_types': ['gradient_boosting', 'random_forest', 'neural_network']
        }


class ScienceTierModel(BaseModel):
    """Model for science subjects."""
    
    def create_model(self, **kwargs) -> Any:
        """Create model for science subjects."""
        model_type = kwargs.get('model_type', 'gradient_boosting')
        
        if model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                **kwargs
            )
        elif model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=42,
                **kwargs
            )
        elif model_type == 'ridge':
            return Ridge(
                alpha=1.0,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'tier': 'science',
            'complexity': 'medium',
            'prerequisite_aware': True,
            'model_types': ['gradient_boosting', 'random_forest', 'ridge']
        }


class ArtsTierModel(BaseModel):
    """Model for arts/social science subjects."""
    
    def create_model(self, **kwargs) -> Any:
        """Create simplified model for arts subjects."""
        model_type = kwargs.get('model_type', 'random_forest')
        
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                **kwargs
            )
        elif model_type == 'linear':
            return LinearRegression(**kwargs)
        elif model_type == 'decision_tree':
            return DecisionTreeRegressor(
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'tier': 'arts',
            'complexity': 'low',
            'computational_efficient': True,
            'model_types': ['random_forest', 'linear', 'decision_tree']
        }


class ModelFactory:
    """
    Factory for creating appropriate models for each tier.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self.model_registry = {
            'critical': CriticalTierModel,
            'science': ScienceTierModel,
            'arts': ArtsTierModel
        }
        
        logger.info("Initialized Model Factory")
    
    def create_model(self, tier_name: str, model_name: str, **kwargs) -> BaseModel:
        """
        Create a model for the specified tier.
        
        Args:
            tier_name: Tier name (critical, science, arts)
            model_name: Name of the model
            **kwargs: Additional model parameters
            
        Returns:
            BaseModel instance
        """
        if tier_name not in self.model_registry:
            raise ValueError(f"Unknown tier: {tier_name}")
        
        model_class = self.model_registry[tier_name]
        model = model_class(model_name, tier_name)
        
        logger.info(f"Created {model_name} for {tier_name} tier")
        return model
    
    def get_available_models(self, tier_name: str) -> Dict[str, Any]:
        """Get available models for a tier."""
        if tier_name not in self.model_registry:
            return {}
        
        model_class = self.model_registry[tier_name]
        temp_model = model_class("temp", tier_name)
        return temp_model.get_model_params()
    
    def list_tiers(self) -> List[str]:
        """List available tiers."""
        return list(self.model_registry.keys())
    
    def register_tier(self, tier_name: str, model_class: type):
        """Register a new tier model class."""
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class must inherit from BaseModel")
        
        self.model_registry[tier_name] = model_class
        logger.info(f"Registered new tier: {tier_name}")


class EnsembleModel:
    """
    Ensemble model that combines multiple models for better predictions.
    """
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of base models
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        logger.info(f"Created ensemble with {len(models)} models")
    
    def train(self, X, y, **kwargs):
        """Train all models in the ensemble."""
        for model in self.models:
            model.train(X, y, **kwargs)
        
        logger.info("Trained all models in ensemble")
    
    def predict(self, X):
        """Make weighted ensemble prediction."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += pred * weight
        
        return weighted_pred
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get average feature importance across all models."""
        importances = []
        
        for model in self.models:
            imp = model.get_feature_importance()
            if imp:
                importances.append(imp)
        
        if not importances:
            return {}
        
        # Average importance across models
        avg_importance = {}
        for key in importances[0].keys():
            avg_importance[key] = np.mean([imp.get(key, 0) for imp in importances])
        
        return avg_importance
