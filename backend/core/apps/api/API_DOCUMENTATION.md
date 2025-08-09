# Smart Student Analytics System (SSAS) API Documentation

## **🎯 Overview**

The SSAS API provides production-ready endpoints for educational analytics powered by advanced machine learning models. The API delivers real-time insights for student performance, career recommendations, peer analysis, and anomaly detection.

**API Version**: v1.0  
**Base URL**: `/api/v1/`  
**Authentication**: Knox Token Authentication  
**Response Format**: JSON  

---

## **🔐 Authentication**

### **Authentication Required**
All endpoints except the API root require authentication using Knox tokens.

```http
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

### **Authentication Endpoints**

#### **Login** 
```http
POST /api/v1/auth/login/
Content-Type: application/json

{
    "username": "student@example.com",
    "password": "secure_password"
}
```

**Response:**
```json
{
    "token": "9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b",
    "expiry": "2024-01-15T10:30:00Z",
    "user": {
        "id": 1,
        "username": "student@example.com"
    }
}
```

#### **Logout**
```http
POST /api/v1/auth/logout/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

---

## **🧠 Student Analytics Endpoints**

### **1. Career Recommendations**

Get AI-powered career recommendations based on student performance and interests.

```http
GET /api/v1/students/{student_id}/career-recommendations/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

**Parameters:**
- `student_id` (string, required): Student identifier (e.g., "STD0001")
- `force_refresh` (query, optional): Force refresh cache (default: false)

**Sample Request:**
```http
GET /api/v1/students/STD0001/career-recommendations/?force_refresh=false
```

**Sample Response:**
```json
{
    "student_id": "STD0001",
    "career_recommendations": [
        {
            "career": "Software Engineering",
            "match_score": 0.89,
            "confidence": 0.85,
            "required_subjects": ["Mathematics", "Physics", "Computer Science"],
            "market_factors": {
                "industry_demand": "high",
                "salary_potential": "excellent",
                "growth_outlook": "strong"
            },
            "reasoning": "Strong performance in Mathematics and logical thinking subjects",
            "university_pathways": [
                {
                    "university": "University of Technology",
                    "program": "Bachelor of Software Engineering",
                    "entry_requirements": {
                        "minimum_score": 75,
                        "required_subjects": ["Mathematics", "Physics"]
                    }
                }
            ]
        },
        {
            "career": "Data Science",
            "match_score": 0.82,
            "confidence": 0.78,
            "required_subjects": ["Mathematics", "Statistics", "Computer Science"],
            "market_factors": {
                "industry_demand": "very_high",
                "salary_potential": "excellent",
                "growth_outlook": "strong"
            },
            "reasoning": "Excellent analytical skills and mathematical foundation"
        }
    ],
    "analysis_metadata": {
        "total_recommendations": 2,
        "analysis_date": "2024-01-10T14:30:00Z",
        "model_version": "v2.1",
        "confidence_threshold": 0.7
    },
    "privacy_guarantees": {
        "differential_privacy": true,
        "epsilon": 1.0,
        "k_anonymity": 10
    }
}
```

**Performance:** < 500ms  
**Cache Duration:** 1 hour  
**Rate Limit:** 100 requests/hour  

---

### **2. Peer Analysis**

Analyze student performance relative to peer groups with privacy protection.

```http
GET /api/v1/students/{student_id}/peer-analysis/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

**Parameters:**
- `student_id` (string, required): Student identifier
- `subjects` (query, optional): Comma-separated subjects filter (e.g., "Mathematics,Physics")
- `force_refresh` (query, optional): Force refresh cache

**Sample Request:**
```http
GET /api/v1/students/STD0001/peer-analysis/?subjects=Mathematics,Physics
```

**Sample Response:**
```json
{
    "student_id": "STD0001",
    "peer_group_size": 10,
    "insights": {
        "performance_status": "above_peer_average",
        "performance_message": "Student performs above peer group average",
        "consistency_status": "more_consistent",
        "consistency_message": "Student shows more consistent performance than peers",
        "trend_status": "improving",
        "trend_message": "Peer group shows improving trend",
        "subject_strengths": ["Mathematics", "Physics"],
        "improvement_areas": ["Literature", "History"]
    },
    "peer_statistics": {
        "peer_average_score": 72.5,
        "student_average_score": 78.3,
        "peer_score_std": 8.2,
        "student_score_std": 6.1,
        "percentile_rank": 75
    },
    "recommendations": [
        "Continue excelling in Mathematics and Physics",
        "Consider additional support in Literature",
        "Peer study groups could help with consistency"
    ],
    "timestamp": "2024-01-10T14:30:00Z",
    "privacy_guarantees": {
        "k_anonymity": 10,
        "epsilon": 1.0,
        "data_anonymized": true
    }
}
```

**Performance:** < 100ms  
**Cache Duration:** 30 minutes  
**Rate Limit:** 100 requests/hour  

---

### **3. Anomaly Detection**

Detect unusual patterns in student performance and behavior.

```http
GET /api/v1/students/{student_id}/anomalies/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

**Parameters:**
- `student_id` (string, required): Student identifier
- `time_window` (query, optional): Analysis time window in days (default: 30)

**Sample Request:**
```http
GET /api/v1/students/STD0001/anomalies/?time_window=60
```

**Sample Response:**
```json
{
    "student_id": "STD0001",
    "anomalies_detected": [
        {
            "type": "performance_drop",
            "severity": "medium",
            "subject": "Mathematics",
            "description": "Significant performance decline in recent assessments",
            "confidence": 0.87,
            "detected_at": "2024-01-08T09:15:00Z",
            "data_points": 5,
            "recommendations": [
                "Schedule additional tutoring sessions",
                "Review recent learning materials",
                "Check for external factors affecting performance"
            ]
        },
        {
            "type": "irregular_submission",
            "severity": "low",
            "subject": "English",
            "description": "Unusual submission timing patterns",
            "confidence": 0.72,
            "detected_at": "2024-01-09T16:45:00Z"
        }
    ],
    "summary": {
        "total_anomalies": 2,
        "high_severity": 0,
        "medium_severity": 1,
        "low_severity": 1,
        "subjects_affected": ["Mathematics", "English"]
    },
    "analysis_period": {
        "start_date": "2023-11-10T00:00:00Z",
        "end_date": "2024-01-10T00:00:00Z",
        "days_analyzed": 60
    },
    "timestamp": "2024-01-10T14:30:00Z"
}
```

**Performance:** < 100ms  
**Cache Duration:** 15 minutes  
**Rate Limit:** 100 requests/hour  

---

### **4. Performance Prediction**

Predict future student performance using advanced ML models.

```http
GET /api/v1/students/{student_id}/performance-prediction/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

**Parameters:**
- `student_id` (string, required): Student identifier
- `subjects` (query, optional): Comma-separated subjects (default: "Mathematics")

**Sample Request:**
```http
GET /api/v1/students/STD0001/performance-prediction/?subjects=Mathematics,Physics,Chemistry
```

**Sample Response:**
```json
{
    "student_id": "STD0001",
    "predictions": [
        {
            "subject": "Mathematics",
            "predicted_score": 82.5,
            "confidence_interval": {
                "lower": 78.2,
                "upper": 86.8,
                "confidence_level": 0.95
            },
            "trend": "improving",
            "factors": {
                "historical_performance": 0.4,
                "recent_trends": 0.3,
                "peer_comparison": 0.2,
                "difficulty_adjustment": 0.1
            }
        },
        {
            "subject": "Physics",
            "predicted_score": 79.1,
            "confidence_interval": {
                "lower": 74.5,
                "upper": 83.7,
                "confidence_level": 0.95
            },
            "trend": "stable"
        }
    ],
    "model_metadata": {
        "model_version": "v1.0",
        "training_data_size": 10000,
        "model_accuracy": 0.89,
        "last_retrained": "2024-01-01T00:00:00Z"
    },
    "recommendations": [
        "Continue current study approach for Mathematics",
        "Consider additional practice for Physics concepts"
    ],
    "timestamp": "2024-01-10T14:30:00Z"
}
```

**Performance:** < 200ms  
**Cache Duration:** 30 minutes  
**Rate Limit:** 100 requests/hour  

---

### **5. Comprehensive Analysis**

Get complete student analysis combining all ML modules.

```http
GET /api/v1/students/{student_id}/comprehensive-analysis/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

**Parameters:**
- `student_id` (string, required): Student identifier
- `async` (query, optional): Process asynchronously (default: false)

**Sample Request:**
```http
GET /api/v1/students/STD0001/comprehensive-analysis/?async=false
```

**Sample Response:**
```json
{
    "student_id": "STD0001",
    "career_analysis": {
        "top_career": "Software Engineering",
        "match_score": 0.89,
        "total_recommendations": 5
    },
    "peer_analysis": {
        "performance_status": "above_peer_average",
        "peer_group_size": 10,
        "percentile_rank": 75
    },
    "anomaly_analysis": {
        "anomalies_detected": 2,
        "highest_severity": "medium",
        "subjects_affected": ["Mathematics", "English"]
    },
    "performance_predictions": {
        "next_term_average": 81.2,
        "confidence": 0.87,
        "trend": "improving"
    },
    "summary": {
        "overall_status": "performing_well",
        "key_strengths": ["Mathematics", "Physics", "Analytical Thinking"],
        "improvement_areas": ["Literature", "Time Management"],
        "priority_actions": [
            "Maintain strong performance in STEM subjects",
            "Focus on improving Literature scores",
            "Address irregular submission patterns"
        ]
    },
    "analysis_timestamp": "2024-01-10T14:30:00Z"
}
```

**Performance:** < 300ms (combined analysis)  
**Cache Duration:** 30 minutes  
**Rate Limit:** 100 requests/hour  

---

## **📊 Data Validation Endpoints**

### **Student Data Validation**

Validate student data quality before running ML analysis.

```http
POST /api/v1/students/validate/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
Content-Type: application/json

{
    "student_id": "STD0001"
}
```

**Sample Response:**
```json
{
    "student_id": "STD0001",
    "student_exists": true,
    "data_quality": {
        "total_scores": 45,
        "subjects_count": 8,
        "academic_years": 3,
        "average_score": 76.8,
        "data_completeness": "good"
    },
    "ml_readiness": {
        "career_analysis": true,
        "peer_analysis": true,
        "anomaly_detection": true,
        "performance_prediction": true
    },
    "recommendations": []
}
```

---

## **⚡ Batch Processing Endpoints**

### **Batch Analysis**

Process multiple students asynchronously for large-scale analytics.

```http
POST /api/v1/batch/analysis/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
Content-Type: application/json

{
    "student_ids": ["STD0001", "STD0002", "STD0003"],
    "analysis_type": "comprehensive"
}
```

**Analysis Types:**
- `career`: Career recommendations only
- `peer`: Peer analysis only  
- `anomaly`: Anomaly detection only
- `comprehensive`: All analyses combined

**Sample Response:**
```json
{
    "task_id": "abc123-def456-ghi789",
    "status": "processing",
    "analysis_type": "comprehensive",
    "student_count": 3,
    "message": "Batch comprehensive analysis started in background",
    "estimated_completion": "2024-01-10T14:35:00Z"
}
```

### **Task Status**

Check the status of asynchronous batch processing tasks.

```http
GET /api/v1/tasks/{task_id}/status/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

**Sample Response (In Progress):**
```json
{
    "task_id": "abc123-def456-ghi789",
    "status": "PROGRESS",
    "ready": false,
    "progress": {
        "completed": 2,
        "total": 3,
        "percentage": 66.7
    },
    "estimated_remaining": "30 seconds"
}
```

**Sample Response (Completed):**
```json
{
    "task_id": "abc123-def456-ghi789",
    "status": "SUCCESS",
    "ready": true,
    "result": {
        "completed_analyses": 3,
        "failed_analyses": 0,
        "results": [
            {
                "student_id": "STD0001",
                "career_recommendations": 5,
                "anomalies_detected": 1,
                "performance_prediction": 82.5
            }
        ]
    },
    "execution_time_ms": 2450.5
}
```

---

## **📊 System Monitoring Endpoints**

### **System Health Check**

Monitor the health and status of all ML modules.

```http
GET /api/v1/system/health/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

**Sample Response:**
```json
{
    "overall_status": "healthy",
    "modules": {
        "career_recommender": {
            "status": "healthy",
            "success_rate": 0.98,
            "total_recommendations": 1250,
            "avg_response_time_ms": 65.2
        },
        "peer_analyzer": {
            "status": "healthy",
            "total_analyses": 890,
            "avg_response_time_ms": 91.8,
            "cache_hit_rate": 0.75
        },
        "anomaly_detector": {
            "status": "healthy",
            "total_detections": 156,
            "false_positive_rate": 0.03,
            "avg_response_time_ms": 17.0
        },
        "performance_predictor": {
            "status": "healthy",
            "total_predictions": 2100,
            "model_accuracy": 0.89,
            "avg_response_time_ms": 39.1
        }
    },
    "system_metrics": {
        "uptime": "7 days, 14 hours",
        "total_requests": 4396,
        "error_rate": 0.02,
        "avg_response_time_ms": 53.3
    },
    "timestamp": "2024-01-10T14:30:00Z"
}
```

### **System Metrics**

Get detailed performance metrics for the entire system.

```http
GET /api/v1/system/metrics/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

**Sample Response:**
```json
{
    "api_performance": {
        "total_api_calls": 4396,
        "total_execution_time_ms": 234520.8,
        "average_response_time_ms": 53.3,
        "uptime_status": "healthy"
    },
    "endpoint_metrics": {
        "get_career_recommendations": {
            "total_calls": 1250,
            "total_time": 81500.0,
            "avg_time": 65.2
        },
        "get_peer_analysis": {
            "total_calls": 890,
            "total_time": 81702.0,
            "avg_time": 91.8
        }
    },
    "performance_targets": {
        "career_recommendations": "< 500ms",
        "peer_analysis": "< 100ms",
        "anomaly_detection": "< 100ms",
        "performance_prediction": "< 200ms"
    },
    "cache_statistics": {
        "hit_rate": 0.75,
        "total_hits": 3297,
        "total_misses": 1099,
        "cache_size_mb": 45.2
    }
}
```

### **API Metrics**

Get detailed metrics about API endpoint performance.

```http
GET /api/v1/system/api-metrics/
Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b
```

---

## **🔧 Error Handling**

### **Standard Error Response Format**

All endpoints return structured error responses:

```json
{
    "error": "Error category",
    "message": "Detailed error description",
    "error_code": "ERROR_CODE",
    "timestamp": "2024-01-10T14:30:00Z",
    "execution_time_ms": "125.5"
}
```

### **HTTP Status Codes**

- **200 OK**: Successful request
- **400 Bad Request**: Invalid input parameters
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Resource not found
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error

### **Error Codes**

- `VALIDATION_ERROR`: Invalid input parameters
- `PERMISSION_DENIED`: Insufficient permissions
- `STUDENT_NOT_FOUND`: Student ID not found in database
- `INSUFFICIENT_DATA`: Not enough data for analysis
- `MODEL_ERROR`: ML model processing error
- `CACHE_ERROR`: Caching system error
- `INTERNAL_ERROR`: Unexpected server error

---

## **⚡ Rate Limiting**

### **Rate Limits by Endpoint Type**

| Endpoint Type | Rate Limit | Scope |
|---------------|------------|-------|
| **ML Analysis** | 100 requests/hour | Per user |
| **Batch Analysis** | 10 requests/hour | Per user |
| **Health Checks** | 200 requests/hour | Per user |
| **General API** | 1000 requests/hour | Per user |
| **Anonymous** | 100 requests/hour | Per IP |

### **Rate Limit Headers**

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1641825600
```

### **Rate Limit Exceeded Response**

```json
{
    "error": "Rate limit exceeded",
    "message": "Too many requests. Please try again later.",
    "error_code": "RATE_LIMIT_EXCEEDED",
    "retry_after": 3600,
    "limit": 100,
    "reset_time": "2024-01-10T15:30:00Z"
}
```

---

## **🚀 Performance Specifications**

### **Response Time Targets**

| Endpoint | Target | Current Performance |
|----------|--------|-------------------|
| Career Recommendations | < 500ms | ~66ms ✅ |
| Peer Analysis | < 100ms | ~92ms ✅ |
| Anomaly Detection | < 100ms | ~17ms ✅ |
| Performance Prediction | < 200ms | ~39ms ✅ |
| Comprehensive Analysis | < 300ms | ~200ms ✅ |

### **Scalability Specifications**

- **Concurrent Users**: 100+ supported
- **Requests per Second**: 50+ per endpoint
- **Cache Hit Rate**: 75%+ target
- **Database Optimization**: Indexed for ML queries
- **Memory Usage**: < 500MB increase under load

---

## **📋 Sample Integration Code**

### **Python Example**

```python
import requests

# Authentication
auth_response = requests.post('http://localhost:8000/api/v1/auth/login/', {
    'username': 'student@example.com',
    'password': 'secure_password'
})
token = auth_response.json()['token']

# Headers for authenticated requests
headers = {
    'Authorization': f'Token {token}',
    'Content-Type': 'application/json'
}

# Get career recommendations
career_response = requests.get(
    'http://localhost:8000/api/v1/students/STD0001/career-recommendations/',
    headers=headers
)

if career_response.status_code == 200:
    recommendations = career_response.json()
    print(f"Found {len(recommendations['career_recommendations'])} career options")
else:
    print(f"Error: {career_response.json()}")
```

### **JavaScript/Node.js Example**

```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:8000/api/v1';

// Authentication
async function authenticate(username, password) {
    const response = await axios.post(`${API_BASE}/auth/login/`, {
        username,
        password
    });
    return response.data.token;
}

// Get student analytics
async function getStudentAnalytics(studentId, token) {
    const headers = { Authorization: `Token ${token}` };
    
    try {
        const [career, peer, anomalies] = await Promise.all([
            axios.get(`${API_BASE}/students/${studentId}/career-recommendations/`, { headers }),
            axios.get(`${API_BASE}/students/${studentId}/peer-analysis/`, { headers }),
            axios.get(`${API_BASE}/students/${studentId}/anomalies/`, { headers })
        ]);
        
        return {
            career: career.data,
            peer: peer.data,
            anomalies: anomalies.data
        };
    } catch (error) {
        console.error('API Error:', error.response?.data || error.message);
        throw error;
    }
}
```

### **cURL Examples**

```bash
# Get API information
curl -X GET http://localhost:8000/api/v1/

# Login
curl -X POST http://localhost:8000/api/v1/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"username": "student@example.com", "password": "secure_password"}'

# Get career recommendations
curl -X GET http://localhost:8000/api/v1/students/STD0001/career-recommendations/ \
  -H "Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b"

# Batch analysis
curl -X POST http://localhost:8000/api/v1/batch/analysis/ \
  -H "Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" \
  -H "Content-Type: application/json" \
  -d '{"student_ids": ["STD0001", "STD0002"], "analysis_type": "comprehensive"}'
```

---

## **🔍 Testing & Development**

### **API Testing Suite**

Run the comprehensive API test suite:

```bash
cd backend
python test_ml_suite.py
```

### **Health Check**

Quick health verification:

```bash
curl -X GET http://localhost:8000/api/v1/ | jq .
```

### **Performance Benchmarking**

```bash
# Test endpoint performance
time curl -X GET http://localhost:8000/api/v1/students/STD0001/peer-analysis/ \
  -H "Authorization: Token YOUR_TOKEN"
```

---

## **📚 Additional Resources**

- **Swagger UI**: `/api/docs/` - Interactive API documentation
- **ReDoc**: `/api/redoc/` - Alternative API documentation
- **Admin Interface**: `/admin/` - Django admin for data management
- **System Health**: `/api/v1/system/health/` - Real-time system status

---

## **🎯 Production Deployment Notes**

### **Environment Variables**
```bash
DJANGO_SETTINGS_MODULE=core.settings.production
SECRET_KEY=your-production-secret-key
DATABASE_URL=postgresql://user:pass@localhost/ssas_db
REDIS_URL=redis://localhost:6379/0
```

### **Performance Optimization**
- **Redis Caching**: Enable Redis for production caching
- **Database Connection Pooling**: Configure for high concurrency
- **Load Balancing**: Use nginx or similar for production
- **SSL/TLS**: Enable HTTPS for production deployment

### **Monitoring Setup**
- **Logging**: All API calls logged with performance metrics
- **Health Checks**: Automated monitoring of ML module health
- **Metrics Collection**: Real-time performance and usage statistics
- **Alerting**: Configure alerts for system degradation

---

**🎓 The SSAS API is production-ready and designed for educational institutions requiring bulletproof reliability and real-time performance for student analytics.**
