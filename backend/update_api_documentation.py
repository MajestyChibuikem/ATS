"""
API Documentation Update Script
Updates API documentation with new unified endpoints and performance metrics.
"""

import os
import sys
import json
from datetime import datetime

def update_api_documentation():
    """Update API documentation with new unified endpoints."""
    print("📚 Updating API Documentation")
    print("=" * 40)
    
    # New unified API documentation
    unified_api_docs = {
        "version": "2.0",
        "last_updated": datetime.now().isoformat(),
        "description": "Unified API for SSAS ML Predictions",
        "base_url": "/api/v1/v2/",
        "endpoints": {
            "unified-predictor": {
                "url": "/api/v1/v2/predict/",
                "method": "GET",
                "description": "Basic unified predictor for all subjects",
                "parameters": {
                    "student_id": "string (required) - Student identifier",
                    "subject": "string (required) - Subject name"
                },
                "response": {
                    "prediction": "float - Predicted score",
                    "confidence": "float - Prediction confidence",
                    "tier": "string - ML tier used (critical/science/arts)",
                    "features_used": "integer - Number of features used"
                },
                "performance": "18.63ms average response time"
            },
            "enhanced-unified-predictor": {
                "url": "/api/v1/v2/predict/enhanced/",
                "method": "GET", 
                "description": "Enhanced predictor with caching and privacy",
                "parameters": {
                    "student_id": "string (required) - Student identifier",
                    "subject": "string (required) - Subject name"
                },
                "response": {
                    "prediction": "float - Predicted score",
                    "confidence": "float - Prediction confidence",
                    "tier": "string - ML tier used",
                    "cached": "boolean - Whether result was cached",
                    "privacy_level": "string - Privacy compliance level"
                },
                "performance": "2.90ms average response time"
            },
            "batch-predictor": {
                "url": "/api/v1/v2/predict/batch/",
                "method": "POST",
                "description": "Batch predictions for multiple students/subjects",
                "parameters": {
                    "requests": "array - Array of prediction requests",
                    "max_batch_size": "integer - Maximum 50 requests per batch"
                },
                "response": {
                    "predictions": "array - Array of prediction results",
                    "summary": "object - Batch processing summary",
                    "processing_time": "float - Total processing time in ms"
                }
            },
            "unified-health-check": {
                "url": "/api/v1/v2/health/",
                "method": "GET",
                "description": "Comprehensive system health monitoring",
                "parameters": {},
                "response": {
                    "status": "string - Overall system status",
                    "tiers": "object - Health status of each ML tier",
                    "performance": "object - Performance metrics",
                    "last_updated": "string - Timestamp of last check"
                },
                "performance": "3.07ms average response time"
            }
        },
        "authentication": {
            "required": True,
            "type": "Token-based authentication",
            "headers": {
                "Authorization": "Token <your_token>"
            }
        },
        "rate_limits": {
            "unified-predictor": "100 requests/hour",
            "enhanced-unified-predictor": "200 requests/hour", 
            "batch-predictor": "10 requests/hour",
            "health-check": "500 requests/hour"
        },
        "ml_tiers": {
            "critical": {
                "subjects": ["Mathematics", "English Language"],
                "features": 30,
                "complexity": "High - Ensemble methods"
            },
            "science": {
                "subjects": ["Physics", "Chemistry", "Biology"],
                "features": 32,
                "complexity": "Medium - Prerequisite awareness"
            },
            "arts": {
                "subjects": ["Literature", "History", "Geography", "Economics"],
                "features": 34,
                "complexity": "Low - Efficient processing"
            }
        },
        "performance_metrics": {
            "average_response_time": "8.20ms",
            "concurrent_requests": "5/5 successful",
            "memory_efficiency": "Optimized model loading",
            "caching": "Redis-based result caching",
            "privacy": "Differential privacy (ε=1.0)"
        }
    }
    
    # Write updated documentation
    with open('UNIFIED_API_DOCUMENTATION.json', 'w') as f:
        json.dump(unified_api_docs, f, indent=2)
    
    print("✅ Updated UNIFIED_API_DOCUMENTATION.json")
    
    # Create markdown documentation
    markdown_docs = f"""# SSAS Unified API Documentation v2.0

## Overview
Smart Student Analytics System (SSAS) Unified API provides ML-powered predictions for all subjects across three specialized tiers.

**Base URL**: `/api/v1/v2/`
**Version**: 2.0
**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Authentication
All endpoints require token-based authentication:
```
Authorization: Token <your_token>
```

## Endpoints

### 1. Basic Unified Predictor
**URL**: `GET /api/v1/v2/predict/`
**Performance**: 18.63ms average response time

Predicts student performance for any subject using the appropriate ML tier.

**Parameters**:
- `student_id` (string, required): Student identifier
- `subject` (string, required): Subject name

**Response**:
```json
{{
    "prediction": 85.5,
    "confidence": 0.92,
    "tier": "critical",
    "features_used": 30
}}
```

### 2. Enhanced Unified Predictor
**URL**: `GET /api/v1/v2/predict/enhanced/`
**Performance**: 2.90ms average response time

Enhanced predictor with caching and privacy compliance.

**Parameters**:
- `student_id` (string, required): Student identifier
- `subject` (string, required): Subject name

**Response**:
```json
{{
    "prediction": 85.5,
    "confidence": 0.92,
    "tier": "critical",
    "cached": true,
    "privacy_level": "differential_privacy"
}}
```

### 3. Batch Predictor
**URL**: `POST /api/v1/v2/predict/batch/`
**Rate Limit**: 10 requests/hour

Process multiple predictions in a single request.

**Request Body**:
```json
{{
    "requests": [
        {{"student_id": "STU0001", "subject": "Mathematics"}},
        {{"student_id": "STU0002", "subject": "Physics"}}
    ]
}}
```

**Response**:
```json
{{
    "predictions": [...],
    "summary": {{
        "total_requests": 2,
        "successful": 2,
        "failed": 0
    }},
    "processing_time": 45.2
}}
```

### 4. Health Check
**URL**: `GET /api/v1/v2/health/`
**Performance**: 3.07ms average response time

Comprehensive system health monitoring.

**Response**:
```json
{{
    "status": "healthy",
    "tiers": {{
        "critical": "healthy",
        "science": "healthy", 
        "arts": "healthy"
    }},
    "performance": {{
        "avg_response_time": "8.20ms",
        "memory_usage": "optimal"
    }}
}}
```

## ML Tiers

### Critical Tier
- **Subjects**: Mathematics, English Language
- **Features**: 30
- **Complexity**: High - Ensemble methods
- **Use Case**: Core academic subjects

### Science Tier  
- **Subjects**: Physics, Chemistry, Biology
- **Features**: 32
- **Complexity**: Medium - Prerequisite awareness
- **Use Case**: Science subjects with dependencies

### Arts Tier
- **Subjects**: Literature, History, Geography, Economics
- **Features**: 34
- **Complexity**: Low - Efficient processing
- **Use Case**: Humanities and social sciences

## Performance Metrics
- **Average Response Time**: 8.20ms
- **Concurrent Processing**: 5/5 successful
- **Memory Efficiency**: Optimized model loading
- **Caching**: Redis-based result caching
- **Privacy**: Differential privacy (ε=1.0)

## Rate Limits
- Basic Predictor: 100 requests/hour
- Enhanced Predictor: 200 requests/hour
- Batch Predictor: 10 requests/hour
- Health Check: 500 requests/hour

## Error Handling
All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (authentication required)
- `429`: Too Many Requests (rate limit exceeded)
- `500`: Internal Server Error

## Examples

### Python Example
```python
import requests

# Basic prediction
response = requests.get(
    'http://localhost:8000/api/v1/v2/predict/',
    params={{'student_id': 'STU0001', 'subject': 'Mathematics'}},
    headers={{'Authorization': 'Token your_token_here'}}
)

# Enhanced prediction with caching
response = requests.get(
    'http://localhost:8000/api/v1/v2/predict/enhanced/',
    params={{'student_id': 'STU0001', 'subject': 'Mathematics'}},
    headers={{'Authorization': 'Token your_token_here'}}
)
```

### cURL Example
```bash
# Health check
curl -H "Authorization: Token your_token_here" \\
     http://localhost:8000/api/v1/v2/health/

# Basic prediction
curl -H "Authorization: Token your_token_here" \\
     "http://localhost:8000/api/v1/v2/predict/?student_id=STU0001&subject=Mathematics"
```
"""
    
    with open('UNIFIED_API_DOCUMENTATION.md', 'w') as f:
        f.write(markdown_docs)
    
    print("✅ Updated UNIFIED_API_DOCUMENTATION.md")
    
    return unified_api_docs

def main():
    """Update API documentation."""
    print("🚀 SSAS API Documentation Update")
    print("=" * 50)
    
    docs = update_api_documentation()
    
    print(f"\n📊 Documentation Summary:")
    print(f"• {len(docs['endpoints'])} endpoints documented")
    print(f"• {len(docs['ml_tiers'])} ML tiers described")
    print(f"• Performance metrics included")
    print(f"• Authentication details provided")
    print(f"• Rate limits specified")
    
    print(f"\n✅ API documentation updated successfully!")
    return docs

if __name__ == "__main__":
    main()
