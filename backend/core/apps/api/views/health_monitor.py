"""
Unified Predictor API Views - Part 4: Health Monitoring.

This module provides health monitoring capabilities for all three tiers
of the unified ML system.
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from datetime import datetime

from core.apps.api.views.unified_predictor_enhanced import EnhancedUnifiedPredictorView

logger = logging.getLogger(__name__)


class UnifiedHealthCheckView(APIView):
    """
    Health check view for monitoring system status.
    
    Provides comprehensive health monitoring for:
    - All three tier predictors (Critical, Science, Arts)
    - Model availability and status
    - System performance metrics
    - Overall system health
    """
    
    permission_classes = [IsAuthenticated]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predictor = EnhancedUnifiedPredictorView()
        logger.info("Health monitor initialized")
    
    def _check_tier_health(self, tier_name: str, tier_predictor):
        """Check health of a specific tier."""
        try:
            # Basic availability check
            is_available = tier_predictor is not None
            
            # Model status check (if models are loaded)
            models_loaded = False
            try:
                if hasattr(tier_predictor, 'load_models'):
                    tier_predictor.load_models()
                    models_loaded = True
            except Exception as e:
                logger.warning(f"Could not load models for {tier_name}: {e}")
            
            return {
                'tier': tier_name,
                'available': is_available,
                'models_loaded': models_loaded,
                'status': 'healthy' if is_available else 'unavailable',
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking health for {tier_name}: {e}")
            return {
                'tier': tier_name,
                'available': False,
                'models_loaded': False,
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now().isoformat()
            }
    
    def _check_system_health(self):
        """Check overall system health."""
        try:
            # Check each tier
            critical_health = self._check_tier_health('critical', self.predictor.critical_predictor)
            science_health = self._check_tier_health('science', self.predictor.science_predictor)
            arts_health = self._check_tier_health('arts', self.predictor.arts_predictor)
            
            # Calculate overall health
            all_tiers = [critical_health, science_health, arts_health]
            available_tiers = sum(1 for tier in all_tiers if tier['available'])
            healthy_tiers = sum(1 for tier in all_tiers if tier['status'] == 'healthy')
            
            overall_status = 'healthy' if healthy_tiers == 3 else 'degraded' if available_tiers > 0 else 'unhealthy'
            
            return {
                'overall_status': overall_status,
                'available_tiers': available_tiers,
                'healthy_tiers': healthy_tiers,
                'total_tiers': 3,
                'tiers': {
                    'critical': critical_health,
                    'science': science_health,
                    'arts': arts_health
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_performance_metrics(self):
        """Check performance metrics."""
        try:
            # Basic performance metrics
            metrics = {
                'cache_enabled': self.predictor.cache_enabled,
                'cache_timeout': self.predictor.cache_timeout,
                'model_version': 'v2.0',
                'api_version': 'v2.0'
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error checking performance metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get(self, request):
        """Get comprehensive health check results."""
        try:
            # Get system health
            system_health = self._check_system_health()
            
            # Get performance metrics
            performance_metrics = self._check_performance_metrics()
            
            # Prepare response
            response_data = {
                'health_check': system_health,
                'performance_metrics': performance_metrics,
                'status_code': status.HTTP_200_OK
            }
            
            # Set appropriate status code based on health
            if system_health.get('overall_status') == 'healthy':
                response_status = status.HTTP_200_OK
            elif system_health.get('overall_status') == 'degraded':
                response_status = status.HTTP_200_OK  # Still functional
            else:
                response_status = status.HTTP_503_SERVICE_UNAVAILABLE
            
            logger.info(f"Health check completed: {system_health.get('overall_status')}")
            
            return Response(response_data, status=response_status)
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return Response(
                {
                    'error': 'Health check failed',
                    'details': str(e),
                    'timestamp': datetime.now().isoformat()
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
