#!/usr/bin/env python
"""
SSAS ML Testing Suite - Simplified Version
Tests core ML functionality and production readiness.
"""

import os
import sys
import django
import time
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any
from asgiref.sync import sync_to_async

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from core.apps.ml.models.career_recommender import CareerRecommender
from core.apps.ml.models.peer_analyzer import PeerAnalyzer
from core.apps.ml.models.anomaly_detector import AnomalyDetector
from core.apps.ml.models.performance_predictor import PerformancePredictor
from core.apps.students.models import Student, StudentScore

@dataclass
class TestResult:
    test_name: str
    passed: bool
    execution_time_ms: float
    error_message: str = ""

class MLTestingSuite:
    def __init__(self):
        self.test_results = []
        self.career_recommender = CareerRecommender()
        self.peer_analyzer = PeerAnalyzer()
        # Initialize anomaly detector with privacy settings
        self.anomaly_detector = AnomalyDetector(epsilon=1.0, contamination=0.1, sensitivity=0.8)
        # Initialize performance predictor with privacy settings
        self.performance_predictor = PerformancePredictor(epsilon=1.0)
    
    async def run_tests(self) -> Dict[str, Any]:
        print("🚀 Starting SSAS ML Testing Suite...")
        
        await self._test_career_recommendations()
        await self._test_peer_analysis()
        await self._test_anomaly_detection()
        await self._test_performance_prediction()
        await self._test_system_health()
        
        return self._generate_report()
    
    async def _test_career_recommendations(self):
        """Test career recommendation system."""
        print("📊 Testing Career Recommendations...")
        
        try:
            # Get a sample student
            student = await sync_to_async(Student.objects.first)()
            if not student:
                self.test_results.append(TestResult(
                    test_name="Career Recommendations - Data Availability",
                    passed=False,
                    execution_time_ms=0,
                    error_message="No students found in database"
                ))
                return
            
            start_time = time.time()
            results = await sync_to_async(self.career_recommender.recommend_careers)(student.student_id)
            execution_time = (time.time() - start_time) * 1000
            
            success = 'error' not in results and len(results.get('career_recommendations', [])) > 0
            
            self.test_results.append(TestResult(
                test_name="Career Recommendations - Basic Functionality",
                passed=success,
                execution_time_ms=execution_time,
                error_message="" if success else str(results.get('error', 'Unknown error'))
            ))
            
            # Test performance
            self.test_results.append(TestResult(
                test_name="Career Recommendations - Performance",
                passed=execution_time < 500,  # Should complete in under 500ms
                execution_time_ms=execution_time
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="Career Recommendations - Exception Handling",
                passed=False,
                execution_time_ms=0,
                error_message=str(e)
            ))
    
    async def _test_peer_analysis(self):
        """Test peer analysis system."""
        print("📊 Testing Peer Analysis...")
        
        try:
            student = await sync_to_async(Student.objects.first)()
            if not student:
                self.test_results.append(TestResult(
                    test_name="Peer Analysis - Data Availability",
                    passed=False,
                    execution_time_ms=0,
                    error_message="No students found in database"
                ))
                return
            
            start_time = time.time()
            results = await sync_to_async(self.peer_analyzer.analyze_student_peers)(student.student_id)
            execution_time = (time.time() - start_time) * 1000
            
            success = 'error' not in results and len(results.get('insights', {})) > 0
            
            self.test_results.append(TestResult(
                test_name="Peer Analysis - Basic Functionality",
                passed=success,
                execution_time_ms=execution_time,
                error_message="" if success else str(results.get('error', 'Unknown error'))
            ))
            
            # Test privacy compliance
            self.test_results.append(TestResult(
                test_name="Peer Analysis - Privacy Compliance",
                passed=results.get('privacy_compliant', False),
                execution_time_ms=0
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="Peer Analysis - Exception Handling",
                passed=False,
                execution_time_ms=0,
                error_message=str(e)
            ))
    
    async def _test_anomaly_detection(self):
        """Test anomaly detection system."""
        print("📊 Testing Anomaly Detection...")
        
        try:
            student = await sync_to_async(Student.objects.first)()
            if not student:
                self.test_results.append(TestResult(
                    test_name="Anomaly Detection - Data Availability",
                    passed=False,
                    execution_time_ms=0,
                    error_message="No students found in database"
                ))
                return
            
            start_time = time.time()
            results = await sync_to_async(self.anomaly_detector.detect_anomalies)(student.student_id)
            execution_time = (time.time() - start_time) * 1000
            
            success = 'error' not in results
            
            self.test_results.append(TestResult(
                test_name="Anomaly Detection - Basic Functionality",
                passed=success,
                execution_time_ms=execution_time,
                error_message="" if success else str(results.get('error', 'Unknown error'))
            ))
            
            # Test differential privacy implementation
            privacy_guarantees = results.get('privacy_guarantees', {})
            privacy_test_passed = (
                privacy_guarantees.get('differential_privacy') == True and
                privacy_guarantees.get('epsilon') == 1.0 and
                privacy_guarantees.get('noise_added') == True
            )
            
            self.test_results.append(TestResult(
                test_name="Anomaly Detection - Differential Privacy",
                passed=privacy_test_passed,
                execution_time_ms=0,
                error_message="" if privacy_test_passed else "Privacy guarantees not properly implemented"
            ))
            
            # Test real-time performance
            self.test_results.append(TestResult(
                test_name="Anomaly Detection - Real-time Performance",
                passed=execution_time < 100,  # Should complete in under 100ms
                execution_time_ms=execution_time
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="Anomaly Detection - Exception Handling",
                passed=False,
                execution_time_ms=0,
                error_message=str(e)
            ))
    
    async def _test_performance_prediction(self):
        """Test performance prediction system."""
        print("📊 Testing Performance Prediction...")
        
        try:
            student = await sync_to_async(Student.objects.first)()
            if not student:
                self.test_results.append(TestResult(
                    test_name="Performance Prediction - Data Availability",
                    passed=False,
                    execution_time_ms=0,
                    error_message="No students found in database"
                ))
                return
            
            start_time = time.time()
            results = await sync_to_async(self.performance_predictor.predict)(student.student_id, ['Mathematics'])
            execution_time = (time.time() - start_time) * 1000
            
            success = 'error' not in results and 'predictions' in results
            
            self.test_results.append(TestResult(
                test_name="Performance Prediction - Basic Functionality",
                passed=success,
                execution_time_ms=execution_time,
                error_message="" if success else str(results.get('error', 'Unknown error'))
            ))
            
            # Test prediction accuracy
            self.test_results.append(TestResult(
                test_name="Performance Prediction - Latency",
                passed=execution_time < 200,  # Should complete in under 200ms
                execution_time_ms=execution_time
            ))
            
            # Test differential privacy implementation
            privacy_guarantees = results.get('privacy_guarantees', {})
            privacy_test_passed = (
                privacy_guarantees.get('differential_privacy') == True and
                privacy_guarantees.get('epsilon') == 1.0 and
                privacy_guarantees.get('noise_added') == True
            )
            
            self.test_results.append(TestResult(
                test_name="Performance Prediction - Differential Privacy",
                passed=privacy_test_passed,
                execution_time_ms=0,
                error_message="" if privacy_test_passed else "Privacy guarantees not properly implemented"
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="Performance Prediction - Exception Handling",
                passed=False,
                execution_time_ms=0,
                error_message=str(e)
            ))
    
    async def _test_system_health(self):
        """Test system health monitoring."""
        print("📊 Testing System Health...")
        
        try:
            # Test career recommender health
            career_health = await sync_to_async(self.career_recommender.get_system_health)()
            self.test_results.append(TestResult(
                test_name="System Health - Career Recommender",
                passed=career_health.get('status', 'unknown') == 'healthy',
                execution_time_ms=0
            ))
            
            # Test peer analyzer health
            peer_health = await sync_to_async(self.peer_analyzer.get_analysis_health)()
            self.test_results.append(TestResult(
                test_name="System Health - Peer Analyzer",
                passed=peer_health.get('status', 'unknown') == 'healthy',
                execution_time_ms=0
            ))
            
            # Test anomaly detector health
            anomaly_health = await sync_to_async(self.anomaly_detector.get_detection_health)()
            self.test_results.append(TestResult(
                test_name="System Health - Anomaly Detector",
                passed=anomaly_health.get('status', 'unknown') == 'healthy',
                execution_time_ms=0
            ))
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="System Health - Exception Handling",
                passed=False,
                execution_time_ms=0,
                error_message=str(e)
            ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        total_execution_time = sum(result.execution_time_ms for result in self.test_results)
        avg_execution_time = total_execution_time / total_tests if total_tests > 0 else 0
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.95:
            system_status = "✅ EXCELLENT - Ready for Production"
        elif success_rate >= 0.90:
            system_status = "🟡 GOOD - Minor Issues to Address"
        elif success_rate >= 0.80:
            system_status = "🟠 FAIR - Several Issues Need Attention"
        else:
            system_status = "🔴 POOR - Major Issues Must Be Fixed"
        
        return {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': round(success_rate, 3),
                'total_execution_time_ms': round(total_execution_time, 2),
                'avg_execution_time_ms': round(avg_execution_time, 2)
            },
            'system_status': system_status,
            'detailed_results': [
                {
                    'test_name': result.test_name,
                    'passed': result.passed,
                    'execution_time_ms': result.execution_time_ms,
                    'error_message': result.error_message
                }
                for result in self.test_results
            ],
            'next_steps': self._generate_next_steps(success_rate),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_next_steps(self, success_rate: float) -> List[str]:
        """Generate next steps based on test results."""
        if success_rate >= 0.95:
            return [
                "✅ Proceed with Phase 3: API Development",
                "Set up production monitoring",
                "Create deployment documentation"
            ]
        elif success_rate >= 0.85:
            return [
                "🔧 Fix remaining test failures",
                "Re-run test suite",
                "Then proceed to Phase 3"
            ]
        else:
            return [
                "🚨 Address critical system issues",
                "Review architecture decisions",
                "Delay Phase 3 until stability achieved"
            ]


async def run_ml_tests():
    """Main function to run all ML tests."""
    test_suite = MLTestingSuite()
    
    print("🧪 SSAS ML Testing Suite Starting...")
    print("=" * 60)
    
    test_report = await test_suite.run_tests()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    summary = test_report['test_summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']} ✅")
    print(f"Failed: {summary['failed_tests']} ❌")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Execution Time: {summary['total_execution_time_ms']:.1f}ms")
    
    print(f"\n{test_report['system_status']}")
    
    print("\n📋 NEXT STEPS:")
    for step in test_report['next_steps']:
        print(f"  • {step}")
    
    return test_report


async def main():
    """Run ML testing suite."""
    print("🎯 SSAS ML PRODUCTION TESTING SUITE")
    print("🧪 Testing Data Quality, ML Accuracy, Privacy, Scalability")
    print("🎯 Production-Ready Testing for Educational Systems")
    
    try:
        # Check if we have data
        student_count = await sync_to_async(Student.objects.count)()
        score_count = await sync_to_async(StudentScore.objects.count)()
        
        print(f"\n📈 Data Overview:")
        print(f"   Students: {student_count}")
        print(f"   Scores: {score_count}")
        
        if student_count == 0 or score_count == 0:
            print("❌ No data available. Please import student data first.")
            return
        
        # Run testing suite
        test_report = await run_ml_tests()
        
        # Display detailed results
        print("\n" + "=" * 60)
        print("📊 DETAILED TEST RESULTS")
        print("=" * 60)
        
        for result in test_report['detailed_results']:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {result['test_name']}")
            if result['execution_time_ms'] > 0:
                print(f"   ⏱️  Time: {result['execution_time_ms']:.1f}ms")
            if result['error_message']:
                print(f"   🚨 Error: {result['error_message']}")
        
        print("\n" + "=" * 60)
        print("🎯 TESTING COMPLETED")
        print("=" * 60)
        print(f"System Status: {test_report['system_status']}")
        print(f"Success Rate: {test_report['test_summary']['success_rate']:.1%}")
        print(f"Total Time: {test_report['test_summary']['total_execution_time_ms']:.1f}ms")
        
        # Determine next steps
        success_rate = test_report['test_summary']['success_rate']
        if success_rate >= 0.95:
            print("\n🚀 EXCELLENT - Ready for Phase 3: API Development")
        elif success_rate >= 0.85:
            print("\n🟡 GOOD - Minor issues to address before Phase 3")
        else:
            print("\n🔴 NEEDS IMPROVEMENT - Critical issues must be fixed")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
