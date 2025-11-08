#!/usr/bin/env python3
"""Simple API Testing Script"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

from django.test import Client
from core.apps.students.models import Student

def test_api():
    """Test basic API endpoints."""
    print("🧪 Testing SSAS API")
    print("=" * 30)
    
    client = Client()
    
    # Test basic endpoints
    endpoints = [
        ("API Root", "/api/v1/"),
        ("Health Check", "/api/v1/system/health/"),
        ("Metrics", "/api/v1/system/metrics/"),
    ]
    
    for name, url in endpoints:
        try:
            response = client.get(url)
            status = "✅ PASS" if response.status_code in [200, 401] else "❌ FAIL"
            print(f"{status} {name}: {response.status_code}")
        except Exception as e:
            print(f"❌ FAIL {name}: {e}")
    
    # Test ML predictions
    student = Student.objects.first()
    if student:
        print(f"\nTesting predictions for student: {student.student_id}")
        
        ml_endpoints = [
            f"/api/v1/students/{student.student_id}/performance-prediction/",
            f"/api/v1/students/{student.student_id}/career-recommendations/",
            f"/api/v1/students/{student.student_id}/peer-analysis/",
        ]
        
        for url in ml_endpoints:
            try:
                response = client.get(url)
                status = "✅ PASS" if response.status_code in [200, 401] else "❌ FAIL"
                print(f"{status} {url}: {response.status_code}")
            except Exception as e:
                print(f"❌ FAIL {url}: {e}")

if __name__ == "__main__":
    test_api()
