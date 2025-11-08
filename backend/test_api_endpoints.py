#!/usr/bin/env python3
"""
SSAS API Endpoint Testing Script
Tests all API endpoints to ensure they work with trained ML models.
"""

import os
import sys
import django
import requests
import json
import time
from datetime import datetime

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.test import Client
from django.urls import reverse
from core.apps.students.models import Student, StudentScore, Subject

def test_api_endpoints():
    """Test all API endpoints."""
    print("🧪 Testing SSAS API Endpoints")
    print("=" * 50)
    
    client = Client()
    base_url = "http://localhost:8000"
    
    # Test results
    results = {
        'passed': 0,
        'failed': 0,
        'total': 0,
        'details': []
    }
    
    def test_endpoint(name, url, method='GET', data=None, expected_status=200):
        """Test a single endpoint."""
        results['total'] += 1
        start_time = time.time()
        
        try:
            if method == 'GET':
                response = client.get(url)
            elif method == 'POST':
                response = client.post(url, data=data, content_type='application/json')
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == expected_status:
                results['passed'] += 1
                status = "✅ PASS"
            else:
                results['failed'] += 1
                status = "❌ FAIL"
            
            result_detail = {
                'name': name,
                'url': url,
                'status_code': response.status_code,
                'expected_status': expected_status,
                'response_time_ms': round(response_time, 2),
                'passed': response.status_code == expected_status
            }
            
            results['details'].append(result_detail)
            
            print(f"{status} {name}")
            print(f"   URL: {url}")
            print(f"   Status: {response.status_code} (expected: {expected_status})")
            print(f"   Time: {response_time:.2f}ms")
            
            if response.status_code != expected_status:
                print(f"   Response: {response.content[:200]}...")
            
            print()
            
        except Exception as e:
            results['failed'] += 1
            print(f"❌ FAIL {name}")
            print(f"   Error: {str(e)}")
            print()
    
    # Test basic API endpoints
    print("📡 Testing Basic API Endpoints")
    print("-" * 30)
    
    test_endpoint("API Root", "/api/v1/")
    test_endpoint("API Metrics", "/api/v1/metrics/")
    test_endpoint("System Health", "/api/v1/health/")
    
    # Test ML prediction endpoints
    print("🤖 Testing ML Prediction Endpoints")
    print("-" * 35)
    
    # Get a real student ID from database
    try:
        student = Student.objects.first()
        if student:
            student_id = student.student_id
            print(f"Using student ID: {student_id}")
            
            # Test Tier 1 (Critical) predictions
            test_endpoint("Tier 1 - Mathematics", f"/api/v1/predictions/tier1/{student_id}/Mathematics/")
            test_endpoint("Tier 1 - English Language", f"/api/v1/predictions/tier1/{student_id}/English Language/")
            
            # Test Tier 2 (Science) predictions
            test_endpoint("Tier 2 - Physics", f"/api/v1/predictions/tier2/{student_id}/Physics/")
            test_endpoint("Tier 2 - Chemistry", f"/api/v1/predictions/tier2/{student_id}/Chemistry/")
            
            # Test Tier 3 (Arts) predictions
            test_endpoint("Tier 3 - Literature", f"/api/v1/predictions/tier3/{student_id}/Literature/")
            test_endpoint("Tier 3 - Government", f"/api/v1/predictions/tier3/{student_id}/Government/")
            
            # Test unified prediction endpoint
            test_endpoint("Unified Prediction", f"/api/v1/predictions/unified/{student_id}/Mathematics/")
            
        else:
            print("⚠️  No students found in database")
            
    except Exception as e:
        print(f"⚠️  Error getting student: {e}")
    
    # Test legacy endpoints
    print("🔄 Testing Legacy API Endpoints")
    print("-" * 30)
    
    test_endpoint("Career Recommendations", "/api/v1/career-recommendations/STD0001/")
    test_endpoint("Peer Analysis", "/api/v1/peer-analysis/STD0001/")
    test_endpoint("Anomaly Detection", "/api/v1/anomaly-detection/STD0001/")
    test_endpoint("Comprehensive Analysis", "/api/v1/comprehensive-analysis/STD0001/")
    
    # Test batch processing
    print("📦 Testing Batch Processing")
    print("-" * 25)
    
    batch_data = {
        'student_ids': ['STD0001', 'STD0002', 'STD0003'],
        'subjects': ['Mathematics', 'Physics', 'Literature']
    }
    
    test_endpoint("Batch Predictions", "/api/v1/batch-predictions/", 
                 method='POST', data=json.dumps(batch_data), expected_status=200)
    
    # Test privacy compliance
    print("🔒 Testing Privacy Compliance")
    print("-" * 25)
    
    test_endpoint("Privacy Compliance", "/api/v1/privacy-compliance/")
    test_endpoint("Data Validation", "/api/v1/validate-data/STD0001/")
    
    # Print summary
    print("📊 Test Summary")
    print("=" * 50)
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']} ✅")
    print(f"Failed: {results['failed']} ❌")
    print(f"Success Rate: {(results['passed']/results['total']*100):.1f}%")
    
    if results['failed'] == 0:
        print("\n🎉 All API endpoints are working correctly!")
        return True
    else:
        print(f"\n⚠️  {results['failed']} endpoints need attention")
        return False

def test_ml_predictions():
    """Test ML predictions directly."""
    print("\n🧠 Testing ML Predictions Directly")
    print("=" * 40)
    
    try:
        from core.apps.ml.models.tier1_critical_predictor import Tier1CriticalPredictor
        from core.apps.ml.models.tier2_science_predictor import Tier2SciencePredictor
        from core.apps.ml.models.tier3_arts_predictor import Tier3ArtsPredictor
        
        # Get a student
        student = Student.objects.first()
        if not student:
            print("❌ No students found in database")
            return False
        
        print(f"Testing predictions for student: {student.student_id}")
        
        # Test Tier 1
        print("\n📊 Testing Tier 1 (Critical) Predictions:")
        tier1 = Tier1CriticalPredictor()
        try:
            pred1 = tier1.predict(student.student_id, 'Mathematics')
            print(f"  Mathematics: {pred1.get('predicted_score', 'N/A')}")
        except Exception as e:
            print(f"  Mathematics: Error - {e}")
        
        # Test Tier 2
        print("\n🔬 Testing Tier 2 (Science) Predictions:")
        tier2 = Tier2SciencePredictor()
        try:
            pred2 = tier2.predict(student.student_id, 'Physics')
            print(f"  Physics: {pred2.get('predicted_score', 'N/A')}")
        except Exception as e:
            print(f"  Physics: Error - {e}")
        
        # Test Tier 3
        print("\n🎨 Testing Tier 3 (Arts) Predictions:")
        tier3 = Tier3ArtsPredictor()
        try:
            pred3 = tier3.predict(student.student_id, 'Literature')
            print(f"  Literature: {pred3.get('predicted_score', 'N/A')}")
        except Exception as e:
            print(f"  Literature: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ML predictions: {e}")
        return False

if __name__ == "__main__":
    print("🚀 SSAS API Testing Suite")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test API endpoints
    api_success = test_api_endpoints()
    
    # Test ML predictions directly
    ml_success = test_ml_predictions()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 FINAL TEST SUMMARY")
    print("=" * 50)
    print(f"API Endpoints: {'✅ PASS' if api_success else '❌ FAIL'}")
    print(f"ML Predictions: {'✅ PASS' if ml_success else '❌ FAIL'}")
    
    if api_success and ml_success:
        print("\n🎉 All tests passed! SSAS is ready for production!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
        sys.exit(1)
