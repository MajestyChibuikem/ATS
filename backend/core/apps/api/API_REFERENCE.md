# SSAS API Reference Guide

## **📋 Quick Reference**

| Endpoint | Method | Authentication | Rate Limit | Cache Duration |
|----------|--------|----------------|-------------|----------------|
| **API Root** | GET | None | 1000/hour | None |
| **Career Recommendations** | GET | Required | 100/hour | 1 hour |
| **Peer Analysis** | GET | Required | 100/hour | 30 minutes |
| **Anomaly Detection** | GET | Required | 100/hour | 15 minutes |
| **Performance Prediction** | GET | Required | 100/hour | 30 minutes |
| **Comprehensive Analysis** | GET | Required | 100/hour | 30 minutes |
| **Data Validation** | POST | Required | 100/hour | None |
| **Batch Analysis** | POST | Required | 10/hour | None |
| **Task Status** | GET | Required | 200/hour | None |
| **System Health** | GET | Required | 200/hour | 5 minutes |
| **System Metrics** | GET | Required | 200/hour | None |
| **API Metrics** | GET | Required | 200/hour | None |

---

## **🔗 Endpoint Details**

### **1. API Root - `/api/v1/`**

**Purpose**: Get API information and available endpoints  
**Authentication**: None (Public)  
**Method**: GET  

**Response Fields:**
- `api_version` (string): Current API version
- `service` (string): Service name
- `description` (string): Service description
- `status` (string): Operational status
- `endpoints` (object): Available endpoint categories
- `rate_limits` (object): Rate limiting information
- `features` (array): Available features list

---

### **2. Career Recommendations - `/api/v1/students/{student_id}/career-recommendations/`**

**Purpose**: Get AI-powered career recommendations  
**Authentication**: Required  
**Method**: GET  
**Performance Target**: < 500ms  

**URL Parameters:**
- `student_id` (string, required): Student identifier

**Query Parameters:**
- `force_refresh` (boolean, optional): Bypass cache (default: false)

**Response Fields:**
- `student_id` (string): Student identifier
- `career_recommendations` (array): List of career recommendations
  - `career` (string): Career name
  - `match_score` (float): Match score (0-1)
  - `confidence` (float): Prediction confidence (0-1)
  - `required_subjects` (array): Required subjects
  - `market_factors` (object): Industry market information
  - `reasoning` (string): Explanation for recommendation
  - `university_pathways` (array): University program options
- `analysis_metadata` (object): Analysis information
- `privacy_guarantees` (object): Privacy protection details

---

### **3. Peer Analysis - `/api/v1/students/{student_id}/peer-analysis/`**

**Purpose**: Analyze student performance relative to peers  
**Authentication**: Required  
**Method**: GET  
**Performance Target**: < 100ms  

**URL Parameters:**
- `student_id` (string, required): Student identifier

**Query Parameters:**
- `subjects` (string, optional): Comma-separated subjects filter
- `force_refresh` (boolean, optional): Bypass cache

**Response Fields:**
- `student_id` (string): Student identifier
- `peer_group_size` (integer): Number of peers in comparison group
- `insights` (object): Performance insights
  - `performance_status` (string): Above/below peer average
  - `performance_message` (string): Human-readable performance description
  - `consistency_status` (string): Consistency compared to peers
  - `consistency_message` (string): Consistency description
  - `trend_status` (string): Performance trend
  - `trend_message` (string): Trend description
- `peer_statistics` (object): Statistical comparisons
- `recommendations` (array): Actionable recommendations
- `privacy_guarantees` (object): Privacy protection details

---

### **4. Anomaly Detection - `/api/v1/students/{student_id}/anomalies/`**

**Purpose**: Detect unusual patterns in student performance  
**Authentication**: Required  
**Method**: GET  
**Performance Target**: < 100ms  

**URL Parameters:**
- `student_id` (string, required): Student identifier

**Query Parameters:**
- `time_window` (integer, optional): Analysis window in days (default: 30)

**Response Fields:**
- `student_id` (string): Student identifier
- `anomalies_detected` (array): List of detected anomalies
  - `type` (string): Anomaly type
  - `severity` (string): Severity level (low/medium/high)
  - `subject` (string): Affected subject
  - `description` (string): Anomaly description
  - `confidence` (float): Detection confidence (0-1)
  - `detected_at` (string): Detection timestamp
  - `recommendations` (array): Suggested actions
- `summary` (object): Anomaly summary statistics
- `analysis_period` (object): Time period analyzed

---

### **5. Performance Prediction - `/api/v1/students/{student_id}/performance-prediction/`**

**Purpose**: Predict future student performance  
**Authentication**: Required  
**Method**: GET  
**Performance Target**: < 200ms  

**URL Parameters:**
- `student_id` (string, required): Student identifier

**Query Parameters:**
- `subjects` (string, optional): Comma-separated subjects (default: "Mathematics")

**Response Fields:**
- `student_id` (string): Student identifier
- `predictions` (array): Performance predictions by subject
  - `subject` (string): Subject name
  - `predicted_score` (float): Predicted score (0-100)
  - `confidence_interval` (object): Prediction confidence bounds
  - `trend` (string): Performance trend
  - `factors` (object): Prediction factors breakdown
- `model_metadata` (object): ML model information
- `recommendations` (array): Study recommendations

---

### **6. Comprehensive Analysis - `/api/v1/students/{student_id}/comprehensive-analysis/`**

**Purpose**: Complete student analysis combining all ML modules  
**Authentication**: Required  
**Method**: GET  
**Performance Target**: < 300ms  

**URL Parameters:**
- `student_id` (string, required): Student identifier

**Query Parameters:**
- `async` (boolean, optional): Process asynchronously (default: false)

**Response Fields:**
- `student_id` (string): Student identifier
- `career_analysis` (object): Career recommendation summary
- `peer_analysis` (object): Peer comparison summary
- `anomaly_analysis` (object): Anomaly detection summary
- `performance_predictions` (object): Performance prediction summary
- `summary` (object): Overall analysis summary
  - `overall_status` (string): Overall performance status
  - `key_strengths` (array): Student's key strengths
  - `improvement_areas` (array): Areas needing improvement
  - `priority_actions` (array): Recommended priority actions

---

### **7. Data Validation - `/api/v1/students/validate/`**

**Purpose**: Validate student data quality before ML analysis  
**Authentication**: Required  
**Method**: POST  

**Request Body:**
```json
{
    "student_id": "STD0001"
}
```

**Response Fields:**
- `student_id` (string): Student identifier
- `student_exists` (boolean): Whether student exists in database
- `data_quality` (object): Data quality metrics
  - `total_scores` (integer): Number of score records
  - `subjects_count` (integer): Number of distinct subjects
  - `academic_years` (integer): Number of academic years
  - `average_score` (float): Average score across all subjects
  - `data_completeness` (string): Data completeness assessment
- `ml_readiness` (object): ML analysis readiness flags
- `recommendations` (array): Data improvement recommendations

---

### **8. Batch Analysis - `/api/v1/batch/analysis/`**

**Purpose**: Process multiple students asynchronously  
**Authentication**: Required  
**Method**: POST  
**Rate Limit**: 10 requests/hour  

**Request Body:**
```json
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

**Response Fields:**
- `task_id` (string): Unique task identifier
- `status` (string): Task status
- `analysis_type` (string): Type of analysis requested
- `student_count` (integer): Number of students to process
- `message` (string): Status message

---

### **9. Task Status - `/api/v1/tasks/{task_id}/status/`**

**Purpose**: Check asynchronous task status  
**Authentication**: Required  
**Method**: GET  

**URL Parameters:**
- `task_id` (string, required): Task identifier from batch analysis

**Response Fields:**
- `task_id` (string): Task identifier
- `status` (string): Task status (PENDING/PROGRESS/SUCCESS/FAILURE)
- `ready` (boolean): Whether task is complete
- `result` (object): Task results (if completed)
- `error` (string): Error message (if failed)

---

### **10. System Health - `/api/v1/system/health/`**

**Purpose**: Monitor ML system health  
**Authentication**: Required  
**Method**: GET  
**Cache Duration**: 5 minutes  

**Response Fields:**
- `overall_status` (string): Overall system health
- `modules` (object): Individual module health status
- `system_metrics` (object): System-wide metrics
- `timestamp` (string): Health check timestamp

---

### **11. System Metrics - `/api/v1/system/metrics/`**

**Purpose**: Get detailed system performance metrics  
**Authentication**: Required  
**Method**: GET  

**Response Fields:**
- `career_recommender` (object): Career recommender metrics
- `peer_analyzer` (object): Peer analyzer metrics
- `anomaly_detector` (object): Anomaly detector metrics
- `cache_stats` (object): Cache performance statistics

---

### **12. API Metrics - `/api/v1/system/api-metrics/`**

**Purpose**: Get API endpoint performance metrics  
**Authentication**: Required  
**Method**: GET  

**Response Fields:**
- `api_performance` (object): Overall API performance
- `endpoint_metrics` (object): Per-endpoint performance metrics
- `performance_targets` (object): Target performance thresholds
- `timestamp` (float): Metrics collection timestamp

---

## **🚨 Error Reference**

### **Common Error Scenarios**

#### **Student Not Found (404)**
```json
{
    "error": "Student not found",
    "message": "Student with ID 'STD9999' does not exist",
    "error_code": "STUDENT_NOT_FOUND",
    "student_id": "STD9999"
}
```

#### **Insufficient Data (400)**
```json
{
    "error": "Insufficient data",
    "message": "Student needs at least 5 score records for career analysis",
    "error_code": "INSUFFICIENT_DATA",
    "required_records": 5,
    "current_records": 2
}
```

#### **Rate Limit Exceeded (429)**
```json
{
    "error": "Rate limit exceeded",
    "message": "Too many requests for ML analysis endpoints",
    "error_code": "RATE_LIMIT_EXCEEDED",
    "retry_after": 3600,
    "limit": 100,
    "current_usage": 101
}
```

#### **Authentication Required (401)**
```json
{
    "error": "Authentication required",
    "message": "Valid authentication token required",
    "error_code": "AUTHENTICATION_REQUIRED"
}
```

#### **Internal Server Error (500)**
```json
{
    "error": "Internal server error",
    "message": "An unexpected error occurred. Please try again later.",
    "error_code": "INTERNAL_ERROR",
    "execution_time_ms": "125.5"
}
```

---

## **⚡ Performance Optimization Tips**

### **1. Caching Strategy**
- Use `force_refresh=false` to leverage caching
- Career recommendations: 1-hour cache
- Peer analysis: 30-minute cache  
- Anomaly detection: 15-minute cache

### **2. Batch Processing**
- Use batch analysis for > 5 students
- Process asynchronously for > 10 students
- Monitor task status for completion

### **3. Query Optimization**
- Filter subjects when possible
- Use appropriate time windows for anomaly detection
- Validate data quality before expensive operations

### **4. Error Handling**
- Implement retry logic with exponential backoff
- Handle rate limiting gracefully
- Cache successful responses locally

---

## **📊 API Health Monitoring**

### **Monitoring Endpoints**

```bash
# Quick health check
curl -H "Authorization: Token YOUR_TOKEN" \
  http://localhost:8000/api/v1/system/health/

# Detailed metrics
curl -H "Authorization: Token YOUR_TOKEN" \
  http://localhost:8000/api/v1/system/api-metrics/

# System performance
curl -H "Authorization: Token YOUR_TOKEN" \
  http://localhost:8000/api/v1/system/metrics/
```

### **Health Status Indicators**

- **healthy**: All systems operational
- **degraded**: Some performance issues
- **unhealthy**: Critical issues requiring attention

### **Performance Thresholds**

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| **Response Time** | < 200ms | > 500ms |
| **Success Rate** | > 95% | < 90% |
| **Cache Hit Rate** | > 75% | < 50% |
| **Error Rate** | < 5% | > 10% |

---

**🎯 This reference guide provides complete technical details for integrating and monitoring the SSAS API in production educational environments.**
