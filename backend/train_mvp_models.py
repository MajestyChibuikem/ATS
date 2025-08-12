"""
Simple ML Model Training for MVP
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

def train_mvp_models():
    """Train ML models for MVP."""
    print("🚀 Training ML Models for MVP")
    print("=" * 40)
    
    # Import ML models
    try:
        from core.apps.ml.models.tier1_critical_predictor import Tier1CriticalPredictor
        from core.apps.ml.models.tier2_science_predictor import Tier2SciencePredictor
        from core.apps.ml.models.tier3_arts_predictor import Tier3ArtsPredictor
        
        print("✅ ML models imported successfully")
        
        # Initialize models
        tier1 = Tier1CriticalPredictor()
        tier2 = Tier2SciencePredictor()
        tier3 = Tier3ArtsPredictor()
        
        print("✅ ML models initialized")
        
        # Train models (this will use the existing training data generation)
        print("Training Tier 1 (Critical)...")
        tier1_success = tier1.train()
        
        print("Training Tier 2 (Science)...")
        tier2_success = tier2.train()
        
        print("Training Tier 3 (Arts)...")
        tier3_success = tier3.train()
        
        if tier1_success and tier2_success and tier3_success:
            print("🎉 All models trained successfully!")
            return True
        else:
            print("❌ Some models failed to train")
            return False
            
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return False

def main():
    """Run MVP model training."""
    success = train_mvp_models()
    
    if success:
        print("\n✅ MVP ML Models ready!")
        print("Next: Test the unified API with real predictions")
    else:
        print("\n❌ Training failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
