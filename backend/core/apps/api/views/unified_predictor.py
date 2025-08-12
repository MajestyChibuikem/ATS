"""
Unified Predictor API Views - Part 1: Basic Unified Predictor.

This module provides the basic unified API endpoint that routes requests
to the appropriate tier (Critical, Science, Arts) based on subject.
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

from core.apps.ml.models.tier1_critical_predictor import Tier1CriticalPredictor
from core.apps.ml.models.tier2_science_predictor import Tier2SciencePredictor
from core.apps.ml.models.tier3_arts_predictor import Tier3ArtsPredictor
from core.apps.students.models import Student

logger = logging.getLogger(__name__)


class UnifiedPredictorView(APIView):
    """
    Basic unified predictor that routes requests to appropriate tiers.
    
    Routes requests to:
    - Tier 1: Critical Subjects (Mathematics, English Language, Further Mathematics)
    - Tier 2: Science Subjects (Physics, Chemistry, Biology, Agricultural Science)
    - Tier 3: Arts Subjects (Government, Economics, History, Literature, Geography, Christian Religious Studies, Civic Education)
    """
    
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.critical_predictor = None
        self.science_predictor = None
        self.arts_predictor = None
        self._initialize_predictors()
    
    def _initialize_predictors(self):
        """Initialize all three tier predictors."""
        try:
            self.critical_predictor = Tier1CriticalPredictor(model_version="v2.0")
            self.science_predictor = Tier2SciencePredictor(model_version="v2.0")
            self.arts_predictor = Tier3ArtsPredictor(model_version="v2.0")
            logger.info("Basic unified predictor initialized")
        except Exception as e:
            logger.error(f"Error initializing unified predictor: {e}")
    
    def _categorize_subject(self, subject_name: str) -> str:
        """Categorize subject into appropriate tier."""
        critical_subjects = ['Mathematics', 'English Language', 'Further Mathematics']
        science_subjects = ['Physics', 'Chemistry', 'Biology', 'Agricultural Science']
        
        if subject_name in critical_subjects:
            return 'critical'
        elif subject_name in science_subjects:
            return 'science'
        else:
            return 'arts'
    
    def _get_prediction(self, student_id: str, subject_name: str):
        """Get prediction from appropriate tier."""
        tier = self._categorize_subject(subject_name)
        
        try:
            if tier == 'critical':
                return self.critical_predictor.predict(student_id, subject_name)
            elif tier == 'science':
                return self.science_predictor.predict(student_id, subject_name)
            else:
                return self.arts_predictor.predict(student_id, subject_name)
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return self._get_fallback_prediction(student_id, subject_name, tier)
    
    def _get_fallback_prediction(self, student_id: str, subject_name: str, tier: str):
        """Get fallback prediction when models are unavailable."""
        fallback_scores = {'critical': 75.0, 'science': 68.0, 'arts': 72.0}
        
        return {
            'student_id': student_id,
            'subject_name': subject_name,
            'predicted_score': fallback_scores.get(tier, 70.0),
            'confidence': 0.5,
            'tier': tier,
            'fallback': True,
            'error': 'Model unavailable'
        }
    
    def get(self, request, student_id: str, subject_name: str):
        """Get performance prediction for a student in a specific subject."""
        try:
            # Validate student exists
            if not Student.objects.filter(student_id=student_id).exists():
                return Response(
                    {'error': f'Student {student_id} not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get prediction from appropriate tier
            prediction = self._get_prediction(student_id, subject_name)
            
            # Add basic metadata
            prediction['unified_api'] = {
                'version': 'v2.0',
                'tier_used': prediction.get('tier', 'unknown'),
                'subject_categorized': self._categorize_subject(subject_name)
            }
            
            logger.info(f"Basic unified prediction completed for {student_id} in {subject_name}")
            return Response(prediction)
            
        except Exception as e:
            logger.error(f"Error in basic unified predictor: {e}")
            return Response(
                {'error': 'Internal server error'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
