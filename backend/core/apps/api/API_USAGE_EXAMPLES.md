# SSAS API Usage Examples

## **🎯 Real-World Usage Scenarios**

This document provides practical examples of how to use the SSAS API in real educational scenarios.

---

## **📚 Scenario 1: Teacher Dashboard Integration**

### **Get Complete Student Overview**

A teacher wants to see a comprehensive overview of student "STD0001" performance.

```python
import requests
import json

class SSASTeacherDashboard:
    def __init__(self, api_base_url, auth_token):
        self.api_base = api_base_url
        self.headers = {
            'Authorization': f'Token {auth_token}',
            'Content-Type': 'application/json'
        }
    
    async def get_student_overview(self, student_id):
        """Get comprehensive student analysis for teacher dashboard."""
        
        # Get comprehensive analysis
        response = requests.get(
            f'{self.api_base}/students/{student_id}/comprehensive-analysis/',
            headers=self.headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract key insights
            overview = {
                'student_id': student_id,
                'performance_status': data['peer_analysis']['performance_status'],
                'top_career_match': data['career_analysis']['top_career'],
                'career_confidence': data['career_analysis']['match_score'],
                'anomalies_count': data['anomaly_analysis']['anomalies_detected'],
                'predicted_next_score': data['performance_predictions']['next_term_average'],
                'key_strengths': data['summary']['key_strengths'],
                'improvement_areas': data['summary']['improvement_areas'],
                'priority_actions': data['summary']['priority_actions']
            }
            
            return overview
        else:
            raise Exception(f"API Error: {response.json()}")

# Usage
dashboard = SSASTeacherDashboard('http://localhost:8000/api/v1', 'your_token_here')
student_overview = dashboard.get_student_overview('STD0001')
print(json.dumps(student_overview, indent=2))
```

**Expected Output:**
```json
{
  "student_id": "STD0001",
  "performance_status": "above_peer_average",
  "top_career_match": "Software Engineering",
  "career_confidence": 0.89,
  "anomalies_count": 2,
  "predicted_next_score": 81.2,
  "key_strengths": ["Mathematics", "Physics", "Analytical Thinking"],
  "improvement_areas": ["Literature", "Time Management"],
  "priority_actions": [
    "Maintain strong performance in STEM subjects",
    "Focus on improving Literature scores",
    "Address irregular submission patterns"
  ]
}
```

---

## **🎓 Scenario 2: Student Self-Assessment Portal**

### **Student Career Exploration**

A student wants to explore career options based on their performance.

```javascript
class StudentCareerExplorer {
    constructor(apiBase, authToken) {
        this.apiBase = apiBase;
        this.headers = {
            'Authorization': `Token ${authToken}`,
            'Content-Type': 'application/json'
        };
    }
    
    async exploreCareerOptions(studentId) {
        try {
            // Get career recommendations
            const careerResponse = await fetch(
                `${this.apiBase}/students/${studentId}/career-recommendations/`,
                { headers: this.headers }
            );
            
            if (!careerResponse.ok) {
                throw new Error(`HTTP ${careerResponse.status}`);
            }
            
            const careerData = await careerResponse.json();
            
            // Format for student-friendly display
            const careerOptions = careerData.career_recommendations.map(career => ({
                title: career.career,
                matchPercentage: Math.round(career.match_score * 100),
                confidence: Math.round(career.confidence * 100),
                marketDemand: career.market_factors.industry_demand,
                salaryPotential: career.market_factors.salary_potential,
                requiredSubjects: career.required_subjects,
                reasoning: career.reasoning,
                universityOptions: career.university_pathways?.length || 0
            }));
            
            return {
                studentId,
                totalOptions: careerOptions.length,
                topMatch: careerOptions[0],
                allOptions: careerOptions,
                privacyProtected: careerData.privacy_guarantees.differential_privacy
            };
            
        } catch (error) {
            console.error('Career exploration failed:', error);
            throw error;
        }
    }
}

// Usage
const explorer = new StudentCareerExplorer('http://localhost:8000/api/v1', 'student_token');
const careerOptions = await explorer.exploreCareerOptions('STD0001');

console.log(`Found ${careerOptions.totalOptions} career matches!`);
console.log(`Top match: ${careerOptions.topMatch.title} (${careerOptions.topMatch.matchPercentage}% match)`);
```

---

## **📊 Scenario 3: School Administrator Analytics**

### **Batch Analysis for Class Performance**

An administrator wants to analyze an entire class of students.

```python
import asyncio
import aiohttp
import json

class SchoolAdminAnalytics:
    def __init__(self, api_base_url, auth_token):
        self.api_base = api_base_url
        self.headers = {
            'Authorization': f'Token {auth_token}',
            'Content-Type': 'application/json'
        }
    
    async def analyze_class_performance(self, student_ids):
        """Analyze performance for entire class."""
        
        async with aiohttp.ClientSession() as session:
            # Start batch analysis
            batch_payload = {
                'student_ids': student_ids,
                'analysis_type': 'comprehensive'
            }
            
            async with session.post(
                f'{self.api_base}/batch/analysis/',
                headers=self.headers,
                json=batch_payload
            ) as response:
                batch_result = await response.json()
                task_id = batch_result['task_id']
                
            print(f"Started batch analysis for {len(student_ids)} students...")
            print(f"Task ID: {task_id}")
            
            # Poll for completion
            while True:
                async with session.get(
                    f'{self.api_base}/tasks/{task_id}/status/',
                    headers=self.headers
                ) as response:
                    status_data = await response.json()
                    
                    if status_data['ready']:
                        if status_data['status'] == 'SUCCESS':
                            return status_data['result']
                        else:
                            raise Exception(f"Batch analysis failed: {status_data.get('error')}")
                    
                    # Wait before next poll
                    await asyncio.sleep(2)
    
    def generate_class_report(self, batch_results):
        """Generate summary report for class."""
        
        students_data = batch_results['results']
        
        # Calculate class statistics
        total_students = len(students_data)
        avg_performance = sum(s['performance_prediction'] for s in students_data) / total_students
        total_anomalies = sum(s['anomalies_detected'] for s in students_data)
        
        # Career distribution
        career_distribution = {}
        for student in students_data:
            career = student.get('top_career', 'Unknown')
            career_distribution[career] = career_distribution.get(career, 0) + 1
        
        report = {
            'class_summary': {
                'total_students': total_students,
                'average_predicted_performance': round(avg_performance, 1),
                'total_anomalies_detected': total_anomalies,
                'students_at_risk': sum(1 for s in students_data if s['anomalies_detected'] > 2)
            },
            'career_insights': {
                'career_distribution': career_distribution,
                'most_popular_career': max(career_distribution, key=career_distribution.get),
                'career_diversity_score': len(career_distribution) / total_students
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if report['class_summary']['students_at_risk'] > total_students * 0.2:
            report['recommendations'].append("High number of at-risk students - consider intervention programs")
        
        if avg_performance < 70:
            report['recommendations'].append("Class average below target - review curriculum delivery")
        
        return report

# Usage
admin = SchoolAdminAnalytics('http://localhost:8000/api/v1', 'admin_token')

# Analyze Class 10A
class_10a_students = ['STD0001', 'STD0002', 'STD0003', 'STD0004', 'STD0005']
batch_results = await admin.analyze_class_performance(class_10a_students)
class_report = admin.generate_class_report(batch_results)

print("Class 10A Performance Report:")
print(json.dumps(class_report, indent=2))
```

---

## **🔔 Scenario 4: Real-Time Monitoring System**

### **Anomaly Alert System**

Monitor students in real-time and alert when anomalies are detected.

```python
import time
import requests
from datetime import datetime

class StudentMonitoringSystem:
    def __init__(self, api_base_url, auth_token):
        self.api_base = api_base_url
        self.headers = {
            'Authorization': f'Token {auth_token}',
            'Content-Type': 'application/json'
        }
        self.alert_thresholds = {
            'high_severity_anomalies': 1,
            'medium_severity_anomalies': 3,
            'performance_drop_threshold': 15  # percentage
        }
    
    def monitor_student(self, student_id):
        """Monitor single student for anomalies."""
        
        try:
            # Get anomaly detection results
            response = requests.get(
                f'{self.api_base}/students/{student_id}/anomalies/',
                headers=self.headers,
                params={'time_window': 7}  # Last 7 days
            )
            
            if response.status_code == 200:
                anomaly_data = response.json()
                return self.process_anomalies(student_id, anomaly_data)
            else:
                print(f"Failed to get anomalies for {student_id}: {response.json()}")
                return None
                
        except Exception as e:
            print(f"Monitoring error for {student_id}: {e}")
            return None
    
    def process_anomalies(self, student_id, anomaly_data):
        """Process anomaly data and generate alerts."""
        
        anomalies = anomaly_data.get('anomalies_detected', [])
        summary = anomaly_data.get('summary', {})
        
        alerts = []
        
        # Check for high severity anomalies
        high_severity = summary.get('high_severity', 0)
        if high_severity >= self.alert_thresholds['high_severity_anomalies']:
            alerts.append({
                'type': 'URGENT',
                'message': f"Student {student_id} has {high_severity} high-severity anomalies",
                'action_required': 'immediate_intervention',
                'anomalies': [a for a in anomalies if a['severity'] == 'high']
            })
        
        # Check for medium severity anomalies
        medium_severity = summary.get('medium_severity', 0)
        if medium_severity >= self.alert_thresholds['medium_severity_anomalies']:
            alerts.append({
                'type': 'WARNING',
                'message': f"Student {student_id} has {medium_severity} medium-severity anomalies",
                'action_required': 'review_recommended',
                'anomalies': [a for a in anomalies if a['severity'] == 'medium']
            })
        
        return {
            'student_id': student_id,
            'monitoring_timestamp': datetime.now().isoformat(),
            'alerts': alerts,
            'total_anomalies': len(anomalies),
            'requires_attention': len(alerts) > 0
        }
    
    def monitor_class(self, student_ids, interval_seconds=300):
        """Monitor entire class with periodic checks."""
        
        print(f"Starting monitoring for {len(student_ids)} students...")
        print(f"Check interval: {interval_seconds} seconds")
        
        while True:
            print(f"\n🔍 Monitoring check at {datetime.now().strftime('%H:%M:%S')}")
            
            alerts_generated = 0
            
            for student_id in student_ids:
                monitoring_result = self.monitor_student(student_id)
                
                if monitoring_result and monitoring_result['requires_attention']:
                    alerts_generated += 1
                    self.send_alert(monitoring_result)
            
            print(f"✅ Monitoring complete - {alerts_generated} alerts generated")
            
            # Wait for next check
            time.sleep(interval_seconds)
    
    def send_alert(self, monitoring_result):
        """Send alert for student requiring attention."""
        
        student_id = monitoring_result['student_id']
        alerts = monitoring_result['alerts']
        
        for alert in alerts:
            print(f"\n🚨 {alert['type']} ALERT for {student_id}")
            print(f"📋 {alert['message']}")
            print(f"⚡ Action: {alert['action_required']}")
            
            # In production, this would send email/SMS/push notifications
            # self.send_notification(alert)

# Usage
monitor = StudentMonitoringSystem('http://localhost:8000/api/v1', 'teacher_token')

# Monitor specific students
class_students = ['STD0001', 'STD0002', 'STD0003']
monitor.monitor_class(class_students, interval_seconds=300)  # Check every 5 minutes
```

---

## **📈 Scenario 5: Academic Performance Analytics**

### **Trend Analysis and Predictions**

Analyze performance trends and make predictions for academic planning.

```python
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class AcademicAnalytics:
    def __init__(self, api_base_url, auth_token):
        self.api_base = api_base_url
        self.headers = {
            'Authorization': f'Token {auth_token}',
            'Content-Type': 'application/json'
        }
    
    def get_class_performance_trends(self, student_ids, subjects=None):
        """Analyze performance trends for entire class."""
        
        class_data = []
        
        for student_id in student_ids:
            try:
                # Get peer analysis for context
                peer_response = requests.get(
                    f'{self.api_base}/students/{student_id}/peer-analysis/',
                    headers=self.headers,
                    params={'subjects': ','.join(subjects)} if subjects else None
                )
                
                # Get performance prediction
                pred_response = requests.get(
                    f'{self.api_base}/students/{student_id}/performance-prediction/',
                    headers=self.headers,
                    params={'subjects': ','.join(subjects)} if subjects else None
                )
                
                if peer_response.status_code == 200 and pred_response.status_code == 200:
                    peer_data = peer_response.json()
                    pred_data = pred_response.json()
                    
                    student_record = {
                        'student_id': student_id,
                        'current_average': peer_data['peer_statistics']['student_average_score'],
                        'peer_average': peer_data['peer_statistics']['peer_average_score'],
                        'percentile_rank': peer_data['peer_statistics']['percentile_rank'],
                        'predicted_score': pred_data['predictions'][0]['predicted_score'],
                        'performance_trend': peer_data['insights']['trend_status'],
                        'consistency_status': peer_data['insights']['consistency_status']
                    }
                    
                    class_data.append(student_record)
                    
            except Exception as e:
                print(f"Error processing {student_id}: {e}")
        
        return pd.DataFrame(class_data)
    
    def generate_performance_insights(self, class_df):
        """Generate insights from class performance data."""
        
        insights = {
            'class_statistics': {
                'total_students': len(class_df),
                'average_current_score': class_df['current_average'].mean(),
                'average_predicted_score': class_df['predicted_score'].mean(),
                'predicted_improvement': class_df['predicted_score'].mean() - class_df['current_average'].mean(),
                'top_performers': len(class_df[class_df['percentile_rank'] >= 80]),
                'at_risk_students': len(class_df[class_df['percentile_rank'] <= 20])
            },
            'trends': {
                'improving_students': len(class_df[class_df['performance_trend'] == 'improving']),
                'declining_students': len(class_df[class_df['performance_trend'] == 'declining']),
                'stable_students': len(class_df[class_df['performance_trend'] == 'stable'])
            },
            'consistency': {
                'consistent_performers': len(class_df[class_df['consistency_status'] == 'more_consistent']),
                'inconsistent_performers': len(class_df[class_df['consistency_status'] == 'less_consistent'])
            }
        }
        
        # Generate recommendations
        recommendations = []
        
        if insights['class_statistics']['at_risk_students'] > len(class_df) * 0.2:
            recommendations.append("High number of at-risk students - implement intervention program")
        
        if insights['trends']['declining_students'] > insights['trends']['improving_students']:
            recommendations.append("More students declining than improving - review teaching methods")
        
        if insights['class_statistics']['predicted_improvement'] < 0:
            recommendations.append("Class predicted to decline - urgent curriculum review needed")
        
        insights['recommendations'] = recommendations
        
        return insights

# Usage Example
analytics = AcademicAnalytics('http://localhost:8000/api/v1', 'admin_token')

# Analyze Mathematics performance for Class 10A
class_10a = ['STD0001', 'STD0002', 'STD0003', 'STD0004', 'STD0005']
performance_df = analytics.get_class_performance_trends(class_10a, ['Mathematics'])
insights = analytics.generate_performance_insights(performance_df)

print("Class 10A Mathematics Performance Analysis:")
print(json.dumps(insights, indent=2))
```

---

## **🔍 Scenario 6: Data Quality Validation**

### **Pre-Analysis Data Validation**

Validate student data quality before running expensive ML analysis.

```python
class DataQualityValidator:
    def __init__(self, api_base_url, auth_token):
        self.api_base = api_base_url
        self.headers = {
            'Authorization': f'Token {auth_token}',
            'Content-Type': 'application/json'
        }
    
    def validate_student_readiness(self, student_id):
        """Validate if student data is ready for ML analysis."""
        
        try:
            response = requests.post(
                f'{self.api_base}/students/validate/',
                headers=self.headers,
                json={'student_id': student_id}
            )
            
            if response.status_code == 200:
                validation_data = response.json()
                
                # Assess readiness
                readiness_score = 0
                max_score = 4
                
                ml_readiness = validation_data['ml_readiness']
                if ml_readiness['career_analysis']:
                    readiness_score += 1
                if ml_readiness['peer_analysis']:
                    readiness_score += 1
                if ml_readiness['anomaly_detection']:
                    readiness_score += 1
                if ml_readiness['performance_prediction']:
                    readiness_score += 1
                
                readiness_percentage = (readiness_score / max_score) * 100
                
                return {
                    'student_id': student_id,
                    'data_exists': validation_data['student_exists'],
                    'readiness_percentage': readiness_percentage,
                    'data_quality': validation_data['data_quality'],
                    'ml_capabilities': ml_readiness,
                    'recommendations': validation_data['recommendations'],
                    'ready_for_analysis': readiness_percentage >= 75
                }
                
            elif response.status_code == 404:
                return {
                    'student_id': student_id,
                    'data_exists': False,
                    'error': 'Student not found in database',
                    'ready_for_analysis': False
                }
            else:
                raise Exception(f"Validation failed: {response.json()}")
                
        except Exception as e:
            return {
                'student_id': student_id,
                'error': str(e),
                'ready_for_analysis': False
            }
    
    def batch_validate_class(self, student_ids):
        """Validate entire class data quality."""
        
        validation_results = []
        
        for student_id in student_ids:
            result = self.validate_student_readiness(student_id)
            validation_results.append(result)
        
        # Generate class validation summary
        ready_count = sum(1 for r in validation_results if r.get('ready_for_analysis', False))
        
        summary = {
            'total_students': len(student_ids),
            'ready_for_analysis': ready_count,
            'data_quality_percentage': (ready_count / len(student_ids)) * 100,
            'students_needing_data': [
                r['student_id'] for r in validation_results 
                if not r.get('ready_for_analysis', False)
            ],
            'validation_details': validation_results
        }
        
        return summary

# Usage
validator = DataQualityValidator('http://localhost:8000/api/v1', 'teacher_token')

# Validate before running expensive analysis
class_validation = validator.batch_validate_class(['STD0001', 'STD0002', 'STD0003'])

print(f"Class Data Quality: {class_validation['data_quality_percentage']:.1f}%")
print(f"Students ready for analysis: {class_validation['ready_for_analysis']}/{class_validation['total_students']}")

if class_validation['students_needing_data']:
    print("Students needing more data:", class_validation['students_needing_data'])
```

---

## **⚡ Scenario 7: Performance Monitoring Dashboard**

### **Real-Time System Health Monitoring**

Monitor API and ML system performance in real-time.

```python
import requests
import time
from datetime import datetime

class SystemHealthMonitor:
    def __init__(self, api_base_url, auth_token):
        self.api_base = api_base_url
        self.headers = {
            'Authorization': f'Token {auth_token}',
            'Content-Type': 'application/json'
        }
        self.health_thresholds = {
            'response_time_ms': 200,
            'success_rate': 0.95,
            'error_rate': 0.05
        }
    
    def get_system_health(self):
        """Get comprehensive system health status."""
        
        try:
            # Get health check
            health_response = requests.get(
                f'{self.api_base}/system/health/',
                headers=self.headers
            )
            
            # Get API metrics
            metrics_response = requests.get(
                f'{self.api_base}/system/api-metrics/',
                headers=self.headers
            )
            
            if health_response.status_code == 200 and metrics_response.status_code == 200:
                health_data = health_response.json()
                metrics_data = metrics_response.json()
                
                return self.analyze_system_status(health_data, metrics_data)
            else:
                return {'status': 'error', 'message': 'Failed to get system data'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def analyze_system_status(self, health_data, metrics_data):
        """Analyze system status and generate alerts."""
        
        # Check ML module health
        ml_modules_healthy = all(
            module['status'] == 'healthy' 
            for module in health_data['modules'].values()
        )
        
        # Check API performance
        avg_response_time = metrics_data['api_performance']['average_response_time_ms']
        performance_acceptable = avg_response_time < self.health_thresholds['response_time_ms']
        
        # Overall system status
        if ml_modules_healthy and performance_acceptable:
            overall_status = 'healthy'
            status_message = '✅ All systems operational'
        elif ml_modules_healthy:
            overall_status = 'degraded'
            status_message = '⚠️ Performance issues detected'
        else:
            overall_status = 'unhealthy'
            status_message = '🚨 ML modules experiencing issues'
        
        return {
            'overall_status': overall_status,
            'status_message': status_message,
            'ml_modules_healthy': ml_modules_healthy,
            'performance_acceptable': performance_acceptable,
            'metrics': {
                'avg_response_time_ms': avg_response_time,
                'total_api_calls': metrics_data['api_performance']['total_api_calls'],
                'cache_hit_rate': metrics_data.get('cache_statistics', {}).get('hit_rate', 0)
            },
            'alerts': self.generate_health_alerts(health_data, metrics_data),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_health_alerts(self, health_data, metrics_data):
        """Generate health alerts based on system status."""
        
        alerts = []
        
        # Check individual module performance
        for module_name, module_data in health_data['modules'].items():
            if module_data['status'] != 'healthy':
                alerts.append({
                    'type': 'MODULE_UNHEALTHY',
                    'module': module_name,
                    'message': f'{module_name} is not healthy',
                    'details': module_data
                })
            
            # Check response times
            avg_time = module_data.get('avg_response_time_ms', 0)
            if avg_time > self.health_thresholds['response_time_ms']:
                alerts.append({
                    'type': 'SLOW_RESPONSE',
                    'module': module_name,
                    'message': f'{module_name} response time ({avg_time:.1f}ms) exceeds threshold',
                    'threshold': self.health_thresholds['response_time_ms']
                })
        
        return alerts
    
    def continuous_monitoring(self, check_interval=60):
        """Run continuous system monitoring."""
        
        print("🔍 Starting continuous system monitoring...")
        print(f"Check interval: {check_interval} seconds")
        
        while True:
            health_status = self.get_system_health()
            
            print(f"\n📊 Health Check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Status: {health_status['status_message']}")
            print(f"Avg Response Time: {health_status['metrics']['avg_response_time_ms']:.1f}ms")
            print(f"Cache Hit Rate: {health_status['metrics']['cache_hit_rate']:.1%}")
            
            if health_status['alerts']:
                print(f"🚨 {len(health_status['alerts'])} alerts:")
                for alert in health_status['alerts']:
                    print(f"   - {alert['type']}: {alert['message']}")
            
            time.sleep(check_interval)

# Usage
monitor = SystemHealthMonitor('http://localhost:8000/api/v1', 'admin_token')

# Get current system status
current_health = monitor.get_system_health()
print("Current System Health:")
print(json.dumps(current_health, indent=2))

# Start continuous monitoring
# monitor.continuous_monitoring(check_interval=300)  # Every 5 minutes
```

---

## **🎯 API Response Time Benchmarks**

### **Performance Testing Script**

```python
import requests
import time
import statistics

def benchmark_api_performance(api_base, auth_token, student_id, iterations=10):
    """Benchmark API endpoint performance."""
    
    headers = {'Authorization': f'Token {auth_token}'}
    
    endpoints = {
        'career_recommendations': f'/students/{student_id}/career-recommendations/',
        'peer_analysis': f'/students/{student_id}/peer-analysis/',
        'anomaly_detection': f'/students/{student_id}/anomalies/',
        'performance_prediction': f'/students/{student_id}/performance-prediction/',
        'comprehensive_analysis': f'/students/{student_id}/comprehensive-analysis/'
    }
    
    results = {}
    
    for endpoint_name, endpoint_path in endpoints.items():
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            response = requests.get(f'{api_base}{endpoint_path}', headers=headers)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                times.append(response_time)
            
            # Small delay between requests
            time.sleep(0.1)
        
        if times:
            results[endpoint_name] = {
                'avg_response_time_ms': statistics.mean(times),
                'min_response_time_ms': min(times),
                'max_response_time_ms': max(times),
                'median_response_time_ms': statistics.median(times),
                'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0,
                'success_rate': len(times) / iterations,
                'iterations': iterations
            }
    
    return results

# Usage
benchmark_results = benchmark_api_performance(
    'http://localhost:8000/api/v1',
    'your_token_here',
    'STD0001',
    iterations=20
)

print("API Performance Benchmark Results:")
for endpoint, metrics in benchmark_results.items():
    print(f"\n{endpoint}:")
    print(f"  Average: {metrics['avg_response_time_ms']:.1f}ms")
    print(f"  Range: {metrics['min_response_time_ms']:.1f}ms - {metrics['max_response_time_ms']:.1f}ms")
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
```

---

## **🔒 Security Best Practices**

### **Token Management**

```python
import requests
from datetime import datetime, timedelta

class SSASTokenManager:
    def __init__(self, api_base_url):
        self.api_base = api_base_url
        self.token = None
        self.token_expiry = None
    
    def login(self, username, password):
        """Login and store token."""
        
        response = requests.post(f'{self.api_base}/auth/login/', {
            'username': username,
            'password': password
        })
        
        if response.status_code == 200:
            data = response.json()
            self.token = data['token']
            self.token_expiry = datetime.fromisoformat(data['expiry'].replace('Z', '+00:00'))
            return True
        else:
            raise Exception(f"Login failed: {response.json()}")
    
    def get_headers(self):
        """Get authentication headers, refreshing token if needed."""
        
        if not self.token:
            raise Exception("Not authenticated - call login() first")
        
        if self.token_expiry and datetime.now() >= self.token_expiry:
            raise Exception("Token expired - please login again")
        
        return {
            'Authorization': f'Token {self.token}',
            'Content-Type': 'application/json'
        }
    
    def logout(self):
        """Logout and clear token."""
        
        if self.token:
            requests.post(
                f'{self.api_base}/auth/logout/',
                headers=self.get_headers()
            )
            self.token = None
            self.token_expiry = None

# Usage
token_manager = SSASTokenManager('http://localhost:8000/api/v1')
token_manager.login('teacher@school.edu', 'secure_password')

# Use for API calls
headers = token_manager.get_headers()
response = requests.get(
    'http://localhost:8000/api/v1/students/STD0001/career-recommendations/',
    headers=headers
)
```

---

## **📱 Frontend Integration Examples**

### **React.js Integration**

```javascript
// SSAS API Client for React
import axios from 'axios';

class SSASApiClient {
    constructor(baseURL = 'http://localhost:8000/api/v1') {
        this.client = axios.create({
            baseURL,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        // Add request interceptor for auth
        this.client.interceptors.request.use(
            config => {
                const token = localStorage.getItem('ssas_token');
                if (token) {
                    config.headers.Authorization = `Token ${token}`;
                }
                return config;
            },
            error => Promise.reject(error)
        );
        
        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            response => response,
            error => {
                if (error.response?.status === 401) {
                    // Token expired, redirect to login
                    localStorage.removeItem('ssas_token');
                    window.location.href = '/login';
                }
                return Promise.reject(error);
            }
        );
    }
    
    // Authentication
    async login(username, password) {
        const response = await this.client.post('/auth/login/', {
            username,
            password
        });
        
        localStorage.setItem('ssas_token', response.data.token);
        return response.data;
    }
    
    // Student Analytics
    async getCareerRecommendations(studentId) {
        const response = await this.client.get(`/students/${studentId}/career-recommendations/`);
        return response.data;
    }
    
    async getPeerAnalysis(studentId, subjects = null) {
        const params = subjects ? { subjects: subjects.join(',') } : {};
        const response = await this.client.get(`/students/${studentId}/peer-analysis/`, { params });
        return response.data;
    }
    
    async getAnomalies(studentId, timeWindow = 30) {
        const response = await this.client.get(`/students/${studentId}/anomalies/`, {
            params: { time_window: timeWindow }
        });
        return response.data;
    }
    
    // System monitoring
    async getSystemHealth() {
        const response = await this.client.get('/system/health/');
        return response.data;
    }
}

// React Component Example
import React, { useState, useEffect } from 'react';

const StudentDashboard = ({ studentId }) => {
    const [analytics, setAnalytics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    const apiClient = new SSASApiClient();
    
    useEffect(() => {
        const loadStudentAnalytics = async () => {
            try {
                setLoading(true);
                
                const [career, peer, anomalies] = await Promise.all([
                    apiClient.getCareerRecommendations(studentId),
                    apiClient.getPeerAnalysis(studentId),
                    apiClient.getAnomalies(studentId)
                ]);
                
                setAnalytics({ career, peer, anomalies });
                setError(null);
                
            } catch (err) {
                setError(err.response?.data?.message || 'Failed to load analytics');
            } finally {
                setLoading(false);
            }
        };
        
        loadStudentAnalytics();
    }, [studentId]);
    
    if (loading) return <div>Loading student analytics...</div>;
    if (error) return <div>Error: {error}</div>;
    
    return (
        <div className="student-dashboard">
            <h2>Student Analytics Dashboard</h2>
            
            {/* Career Recommendations */}
            <div className="career-section">
                <h3>Career Recommendations</h3>
                {analytics.career.career_recommendations.map((career, index) => (
                    <div key={index} className="career-card">
                        <h4>{career.career}</h4>
                        <p>Match: {Math.round(career.match_score * 100)}%</p>
                        <p>{career.reasoning}</p>
                    </div>
                ))}
            </div>
            
            {/* Peer Analysis */}
            <div className="peer-section">
                <h3>Peer Comparison</h3>
                <p>Status: {analytics.peer.insights.performance_message}</p>
                <p>Percentile: {analytics.peer.peer_statistics.percentile_rank}th</p>
            </div>
            
            {/* Anomalies */}
            <div className="anomaly-section">
                <h3>Performance Alerts</h3>
                {analytics.anomalies.anomalies_detected.length > 0 ? (
                    analytics.anomalies.anomalies_detected.map((anomaly, index) => (
                        <div key={index} className={`alert alert-${anomaly.severity}`}>
                            <strong>{anomaly.type}</strong>: {anomaly.description}
                        </div>
                    ))
                ) : (
                    <p>No anomalies detected ✅</p>
                )}
            </div>
        </div>
    );
};

export default StudentDashboard;
```

---

## **🎯 Production Deployment Checklist**

### **API Configuration**

- [ ] Configure Redis for production caching
- [ ] Set up proper database connection pooling
- [ ] Configure SSL/TLS certificates
- [ ] Set production-appropriate rate limits
- [ ] Configure logging and monitoring
- [ ] Set up backup and recovery procedures

### **Security Configuration**

- [ ] Generate secure SECRET_KEY
- [ ] Configure CORS settings for frontend
- [ ] Set up proper firewall rules
- [ ] Configure user authentication system
- [ ] Implement API key management
- [ ] Set up audit logging

### **Performance Configuration**

- [ ] Enable Redis caching
- [ ] Configure database indexes
- [ ] Set up CDN for static files
- [ ] Configure load balancing
- [ ] Set up monitoring and alerting
- [ ] Configure auto-scaling

---

**🎓 This API documentation provides everything needed to integrate the SSAS system into educational institutions with confidence and reliability.**
