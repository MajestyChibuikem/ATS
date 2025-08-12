"""
Unified Predictor API Views - Part 3: Batch Processing.

This module provides batch prediction capabilities for multiple student/subject
combinations in a single request.
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import JSONParser

from core.apps.api.views.unified_predictor_enhanced import EnhancedUnifiedPredictorView

logger = logging.getLogger(__name__)


class UnifiedBatchPredictorView(APIView):
    """
    Batch predictor for multiple student/subject predictions.
    
    Handles multiple predictions in a single request with:
    - Batch size limits and validation
    - Error handling for individual predictions
    - Batch processing optimization
    """
    
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_batch_size = 50  # Maximum predictions per request
        self.predictor = EnhancedUnifiedPredictorView()
        logger.info("Batch predictor initialized")
    
    def _validate_batch_request(self, data):
        """Validate batch request data."""
        if not isinstance(data, list):
            return False, "Request data must be a list"
        
        if len(data) > self.max_batch_size:
            return False, f"Batch size exceeds maximum of {self.max_batch_size}"
        
        if len(data) == 0:
            return False, "Batch request cannot be empty"
        
        # Validate each prediction request
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return False, f"Item {i} must be a dictionary"
            
            if 'student_id' not in item or 'subject_name' not in item:
                return False, f"Item {i} must contain 'student_id' and 'subject_name'"
            
            if not isinstance(item['student_id'], str) or not isinstance(item['subject_name'], str):
                return False, f"Item {i} must have string values for 'student_id' and 'subject_name'"
        
        return True, "Valid batch request"
    
    def _process_single_prediction(self, student_id: str, subject_name: str):
        """Process a single prediction request."""
        try:
            # Use the enhanced predictor for individual predictions
            prediction = self.predictor._get_prediction(student_id, subject_name)
            
            # Add batch metadata
            prediction['batch_metadata'] = {
                'student_id': student_id,
                'subject_name': subject_name,
                'processed': True,
                'error': None
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing prediction for {student_id} in {subject_name}: {e}")
            return {
                'student_id': student_id,
                'subject_name': subject_name,
                'predicted_score': None,
                'error': str(e),
                'batch_metadata': {
                    'student_id': student_id,
                    'subject_name': subject_name,
                    'processed': False,
                    'error': str(e)
                }
            }
    
    def post(self, request):
        """Process batch prediction request."""
        try:
            # Validate request data
            is_valid, message = self._validate_batch_request(request.data)
            if not is_valid:
                return Response(
                    {'error': f'Invalid batch request: {message}'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            batch_data = request.data
            results = []
            successful_predictions = 0
            failed_predictions = 0
            
            logger.info(f"Processing batch request with {len(batch_data)} predictions")
            
            # Process each prediction in the batch
            for item in batch_data:
                student_id = item['student_id']
                subject_name = item['subject_name']
                
                result = self._process_single_prediction(student_id, subject_name)
                results.append(result)
                
                if result.get('error'):
                    failed_predictions += 1
                else:
                    successful_predictions += 1
            
            # Prepare response
            response_data = {
                'batch_summary': {
                    'total_requests': len(batch_data),
                    'successful_predictions': successful_predictions,
                    'failed_predictions': failed_predictions,
                    'success_rate': successful_predictions / len(batch_data) if batch_data else 0
                },
                'predictions': results
            }
            
            logger.info(f"Batch processing completed: {successful_predictions} successful, {failed_predictions} failed")
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error in batch predictor: {e}")
            return Response(
                {'error': 'Internal server error', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
