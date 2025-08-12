"""
A/B Testing Framework for SSAS ML Models.

Simplified A/B testing capabilities for comparing model configurations.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ABTestManager:
    """Manager for A/B testing of ML models."""
    
    def __init__(self):
        self.tests = {}
        self.results = []
        logger.info("A/B Test Manager initialized")
    
    def create_test(self, test_id: str, name: str, variants: Dict[str, float]) -> bool:
        """Create a new A/B test."""
        try:
            # Validate traffic split
            total_split = sum(variants.values())
            if abs(total_split - 1.0) > 0.01:
                return False
            
            self.tests[test_id] = {
                'test_id': test_id,
                'name': name,
                'variants': variants,
                'status': 'active',
                'created_at': datetime.now()
            }
            
            logger.info(f"A/B test '{name}' created with ID: {test_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            return False
    
    def assign_variant(self, test_id: str, student_id: str) -> Optional[str]:
        """Assign a variant to a student."""
        try:
            test = self.tests.get(test_id)
            if not test or test['status'] != 'active':
                return None
            
            # Use consistent hashing
            hash_input = f"{test_id}:{student_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            
            # Assign variant based on traffic split
            cumulative_prob = 0.0
            for variant, probability in test['variants'].items():
                cumulative_prob += probability
                if hash_value / (2**128) <= cumulative_prob:
                    return variant
            
            return list(test['variants'].keys())[0]
            
        except Exception as e:
            logger.error(f"Error assigning variant: {e}")
            return None
    
    def record_result(self, test_id: str, variant: str, student_id: str, 
                     prediction_score: float, latency_ms: float = None) -> bool:
        """Record a test result."""
        try:
            result = {
                'test_id': test_id,
                'variant': variant,
                'student_id': student_id,
                'prediction_score': prediction_score,
                'latency_ms': latency_ms,
                'timestamp': datetime.now()
            }
            
            self.results.append(result)
            return True
            
        except Exception as e:
            logger.error(f"Error recording result: {e}")
            return False
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze test results."""
        try:
            test_results = [r for r in self.results if r['test_id'] == test_id]
            if not test_results:
                return {'error': 'No results found'}
            
            # Group by variant
            variants = {}
            for result in test_results:
                variant = result['variant']
                if variant not in variants:
                    variants[variant] = []
                variants[variant].append(result)
            
            # Calculate metrics
            analysis = {'test_id': test_id, 'variants': {}}
            for variant, results in variants.items():
                scores = [r['prediction_score'] for r in results]
                latencies = [r['latency_ms'] for r in results if r['latency_ms']]
                
                analysis['variants'][variant] = {
                    'count': len(results),
                    'avg_score': sum(scores) / len(scores) if scores else 0,
                    'avg_latency': sum(latencies) / len(latencies) if latencies else 0
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing test: {e}")
            return {'error': str(e)}
    
    def get_active_tests(self) -> List[str]:
        """Get list of active test IDs."""
        return [test_id for test_id, test in self.tests.items() 
                if test['status'] == 'active']
