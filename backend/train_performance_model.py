#!/usr/bin/env python
"""
Training script for the Performance Predictor model.
Production-ready with comprehensive logging and validation.
"""

import os
import sys
import django
import logging
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from core.apps.ml.models.performance_predictor import PerformancePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    logger.info("=== Starting Performance Predictor Training ===")
    
    try:
        # Initialize predictor
        predictor = PerformancePredictor(model_version="v1.0")
        
        # Train the model
        logger.info("Training performance predictor...")
        training_result = predictor.train()
        
        if training_result['status'] == 'success':
            logger.info("✅ Training completed successfully!")
            
            # Log detailed metrics
            metrics = training_result['metrics']
            logger.info(f"📊 Training Summary:")
            logger.info(f"   - Subjects trained: {training_result['subjects_trained']}")
            logger.info(f"   - Training time: {training_result['training_time']}")
            
            # Log performance metrics for each subject
            logger.info(f"📈 Subject Performance:")
            for subject, metric in metrics.items():
                logger.info(f"   {subject}:")
                logger.info(f"     - RMSE: {metric['rmse']:.2f}")
                logger.info(f"     - MAE: {metric['mae']:.2f}")
                logger.info(f"     - R²: {metric['r2']:.3f}")
                logger.info(f"     - CV RMSE: {metric['cv_rmse']:.2f}")
                logger.info(f"     - Samples: {metric['samples']}")
            
            # Check if models meet production criteria
            production_ready = True
            for subject, metric in metrics.items():
                if metric['rmse'] > 8.0:  # RMSE threshold
                    logger.warning(f"⚠️  {subject} RMSE ({metric['rmse']:.2f}) exceeds production threshold (8.0)")
                    production_ready = False
                
                if metric['r2'] < 0.3:  # R² threshold
                    logger.warning(f"⚠️  {subject} R² ({metric['r2']:.3f}) below production threshold (0.3)")
                    production_ready = False
            
            if production_ready:
                logger.info("✅ All models meet production criteria!")
            else:
                logger.warning("⚠️  Some models need improvement before production deployment")
            
            # Test prediction on a sample student
            logger.info("🧪 Testing prediction on sample student...")
            test_prediction = predictor.predict("STD0001")
            
            if test_prediction.get('fallback'):
                logger.warning(f"⚠️  Test prediction failed: {test_prediction.get('fallback_reason')}")
            else:
                logger.info("✅ Test prediction successful!")
                logger.info(f"   - Prediction time: {test_prediction['prediction_time']:.3f}s")
                logger.info(f"   - Subjects predicted: {len(test_prediction['predictions'])}")
            
            # Get model health
            health = predictor.get_model_health()
            logger.info(f"🏥 Model Health:")
            logger.info(f"   - Version: {health['model_version']}")
            logger.info(f"   - Subjects trained: {health['subjects_trained']}")
            logger.info(f"   - Last training: {health['last_training_time']}")
            logger.info(f"   - Prediction count: {health['prediction_count']}")
            logger.info(f"   - Error rate: {health['error_rate']:.3f}")
            
        else:
            logger.error(f"❌ Training failed: {training_result['error']}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training script failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    if success:
        logger.info("🎉 Training script completed successfully!")
    else:
        logger.error("💥 Training script failed!")
        sys.exit(1)
