"""
Anomaly Detection - Phase 2
Identifies sudden performance changes, behavioral patterns, and stress indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from core.apps.students.models import Student, StudentScore, Subject

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detection for student performance and behavioral patterns.
    
    Features:
    - Sudden performance changes identification
    - Behavioral pattern analysis
    - Stress indicators through grade volatility
    - Early intervention triggers
    - Multi-dimensional anomaly detection
    """
    
    def __init__(self, contamination: float = 0.1, sensitivity: float = 0.8):
        self.contamination = contamination  # Expected proportion of anomalies
        self.sensitivity = sensitivity  # Detection sensitivity
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.anomaly_thresholds = {}
        self.behavioral_patterns = {}
        
        # Monitoring
        self.detection_count = 0
        self.false_positives = 0
        self.last_detection_time = None
        
    def detect_anomalies(self, student_id: str, time_window: int = 30) -> Dict[str, Any]:
        """
        Detect anomalies in student performance and behavior.
        
        Args:
            student_id: Target student ID
            time_window: Days to look back for analysis
            
        Returns:
            Comprehensive anomaly analysis results
        """
        try:
            # Get student data
            student_data = self._get_student_data(student_id, time_window)
            if student_data is None or student_data.empty:
                return self._fallback_detection(student_id, "No data available")
            
            # Detect different types of anomalies
            performance_anomalies = self._detect_performance_anomalies(student_data)
            behavioral_anomalies = self._detect_behavioral_anomalies(student_data)
            stress_indicators = self._detect_stress_indicators(student_data)
            
            # Combine and prioritize anomalies
            combined_anomalies = self._combine_anomalies(
                performance_anomalies, 
                behavioral_anomalies, 
                stress_indicators
            )
            
            # Generate intervention recommendations
            recommendations = self._generate_interventions(combined_anomalies)
            
            # Log detection
            self._log_detection(student_id, combined_anomalies)
            
            return {
                'student_id': student_id,
                'analysis_window': f"{time_window} days",
                'anomalies_detected': len(combined_anomalies),
                'performance_anomalies': performance_anomalies,
                'behavioral_anomalies': behavioral_anomalies,
                'stress_indicators': stress_indicators,
                'combined_analysis': combined_anomalies,
                'intervention_recommendations': recommendations,
                'detection_confidence': self._calculate_confidence(combined_anomalies),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed for student {student_id}: {e}")
            return self._fallback_detection(student_id, str(e))
    
    def _get_student_data(self, student_id: str, time_window: int) -> Optional[pd.DataFrame]:
        """Get student data within time window."""
        try:
            student = Student.objects.get(student_id=student_id)
            cutoff_date = datetime.now() - timedelta(days=time_window)
            
            scores = StudentScore.objects.filter(
                student=student,
                created_at__gte=cutoff_date
            ).select_related('subject').order_by('created_at')
            
            if not scores.exists():
                return None
            
            data = []
            for score in scores:
                data.append({
                    'student_id': student.student_id,
                    'subject': score.subject.name,
                    'total_score': float(score.total_score),
                    'class_average': float(score.class_average),
                    'grade': score.grade,
                    'term': score.term,
                    'academic_year': score.academic_year,
                    'created_at': score.created_at,
                    'score_difference': float(score.total_score - score.class_average)
                })
            
            return pd.DataFrame(data)
            
        except Student.DoesNotExist:
            return None
    
    def _detect_performance_anomalies(self, student_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect sudden changes in performance patterns."""
        anomalies = []
        
        if len(student_data) < 3:
            return anomalies
        
        # Calculate performance metrics
        scores = student_data['total_score'].values
        dates = pd.to_datetime(student_data['created_at'])
        
        # 1. Sudden drops in performance
        score_changes = np.diff(scores)
        mean_change = np.mean(score_changes)
        std_change = np.std(score_changes)
        
        for i, change in enumerate(score_changes):
            if change < (mean_change - 2 * std_change):  # Significant drop
                anomalies.append({
                    'type': 'performance_drop',
                    'severity': 'high' if change < (mean_change - 3 * std_change) else 'medium',
                    'description': f'Sudden performance drop of {abs(change):.2f} points',
                    'date': dates[i + 1].strftime('%Y-%m-%d'),
                    'subject': student_data.iloc[i + 1]['subject'],
                    'score_before': scores[i],
                    'score_after': scores[i + 1],
                    'change': change
                })
        
        # 2. Unusual score volatility
        score_std = np.std(scores)
        if score_std > np.mean(scores) * 0.3:  # High volatility
            anomalies.append({
                'type': 'high_volatility',
                'severity': 'medium',
                'description': f'Unusually high score volatility (std: {score_std:.2f})',
                'date': dates[-1].strftime('%Y-%m-%d'),
                'volatility': score_std,
                'mean_score': np.mean(scores)
            })
        
        # 3. Consistent underperformance
        class_averages = student_data['class_average'].values
        underperformance_count = np.sum(scores < class_averages)
        underperformance_rate = underperformance_count / len(scores)
        
        if underperformance_rate > 0.7:  # 70% of scores below class average
            anomalies.append({
                'type': 'consistent_underperformance',
                'severity': 'high',
                'description': f'Consistently performing below class average ({underperformance_rate:.1%})',
                'date': dates[-1].strftime('%Y-%m-%d'),
                'underperformance_rate': underperformance_rate,
                'total_scores': len(scores)
            })
        
        return anomalies
    
    def _detect_behavioral_anomalies(self, student_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect behavioral pattern anomalies."""
        anomalies = []
        
        if len(student_data) < 5:
            return anomalies
        
        # 1. Irregular submission patterns
        dates = pd.to_datetime(student_data['created_at'])
        time_diffs = np.diff((dates - dates.min()).dt.days)
        
        if len(time_diffs) > 0:
            mean_interval = np.mean(time_diffs)
            std_interval = np.std(time_diffs)
            
            # Detect irregular intervals
            irregular_intervals = time_diffs[np.abs(time_diffs - mean_interval) > 2 * std_interval]
            if len(irregular_intervals) > len(time_diffs) * 0.3:  # 30% irregular
                anomalies.append({
                    'type': 'irregular_submission',
                    'severity': 'medium',
                    'description': 'Irregular submission patterns detected',
                    'date': dates[-1].strftime('%Y-%m-%d'),
                    'irregular_rate': len(irregular_intervals) / len(time_diffs)
                })
        
        # 2. Subject-specific patterns
        subject_counts = student_data['subject'].value_counts()
        total_scores = len(student_data)
        
        for subject, count in subject_counts.items():
            subject_rate = count / total_scores
            
            # Detect over-focus on specific subjects
            if subject_rate > 0.4:  # 40% of scores in one subject
                anomalies.append({
                    'type': 'subject_overfocus',
                    'severity': 'low',
                    'description': f'Over-focus on {subject} ({subject_rate:.1%} of scores)',
                    'date': dates[-1].strftime('%Y-%m-%d'),
                    'subject': subject,
                    'focus_rate': subject_rate
                })
        
        # 3. Performance consistency across subjects
        subject_performance = student_data.groupby('subject')['total_score'].agg(['mean', 'std'])
        
        # Detect subjects with unusually high/low performance
        overall_mean = student_data['total_score'].mean()
        overall_std = student_data['total_score'].std()
        
        for subject, stats in subject_performance.iterrows():
            subject_mean = stats['mean']
            
            if subject_mean < (overall_mean - 2 * overall_std):
                anomalies.append({
                    'type': 'subject_weakness',
                    'severity': 'medium',
                    'description': f'Significantly lower performance in {subject}',
                    'date': dates[-1].strftime('%Y-%m-%d'),
                    'subject': subject,
                    'subject_mean': subject_mean,
                    'overall_mean': overall_mean
                })
        
        return anomalies
    
    def _detect_stress_indicators(self, student_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect stress indicators through grade volatility and patterns."""
        indicators = []
        
        if len(student_data) < 3:
            return indicators
        
        scores = student_data['total_score'].values
        dates = pd.to_datetime(student_data['created_at'])
        
        # 1. Increasing volatility over time (stress indicator)
        if len(scores) >= 6:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            first_volatility = np.std(first_half)
            second_volatility = np.std(second_half)
            
            if second_volatility > first_volatility * 1.5:  # 50% increase in volatility
                indicators.append({
                    'type': 'increasing_volatility',
                    'severity': 'high',
                    'description': 'Increasing score volatility over time (potential stress indicator)',
                    'date': dates[-1].strftime('%Y-%m-%d'),
                    'early_volatility': first_volatility,
                    'recent_volatility': second_volatility,
                    'increase_factor': second_volatility / first_volatility
                })
        
        # 2. Performance decline trend
        if len(scores) >= 4:
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]
            
            if slope < -2:  # Declining trend
                indicators.append({
                    'type': 'performance_decline',
                    'severity': 'high',
                    'description': 'Consistent performance decline over time',
                    'date': dates[-1].strftime('%Y-%m-%d'),
                    'decline_rate': abs(slope),
                    'trend_direction': 'declining'
                })
        
        # 3. Unusual score patterns (potential stress)
        score_diffs = np.diff(scores)
        extreme_changes = score_diffs[np.abs(score_diffs) > np.std(score_diffs) * 2]
        
        if len(extreme_changes) > len(score_diffs) * 0.2:  # 20% extreme changes
            indicators.append({
                'type': 'extreme_score_changes',
                'severity': 'medium',
                'description': 'Unusual number of extreme score changes',
                'date': dates[-1].strftime('%Y-%m-%d'),
                'extreme_change_rate': len(extreme_changes) / len(score_diffs),
                'total_extreme_changes': len(extreme_changes)
            })
        
        # 4. Performance vs. class average gap widening
        score_diffs_from_class = student_data['score_difference'].values
        
        if len(score_diffs_from_class) >= 3:
            early_gap = np.mean(score_diffs_from_class[:len(score_diffs_from_class)//2])
            recent_gap = np.mean(score_diffs_from_class[len(score_diffs_from_class)//2:])
            
            if recent_gap < early_gap - 5:  # Widening negative gap
                indicators.append({
                    'type': 'widening_performance_gap',
                    'severity': 'medium',
                    'description': 'Performance gap with class average is widening',
                    'date': dates[-1].strftime('%Y-%m-%d'),
                    'early_gap': early_gap,
                    'recent_gap': recent_gap,
                    'gap_change': recent_gap - early_gap
                })
        
        return indicators
    
    def _combine_anomalies(self, performance: List[Dict], behavioral: List[Dict], stress: List[Dict]) -> List[Dict[str, Any]]:
        """Combine and prioritize all detected anomalies."""
        all_anomalies = []
        
        # Add performance anomalies
        for anomaly in performance:
            anomaly['category'] = 'performance'
            all_anomalies.append(anomaly)
        
        # Add behavioral anomalies
        for anomaly in behavioral:
            anomaly['category'] = 'behavioral'
            all_anomalies.append(anomaly)
        
        # Add stress indicators
        for indicator in stress:
            indicator['category'] = 'stress'
            all_anomalies.append(indicator)
        
        # Sort by severity and type
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        all_anomalies.sort(key=lambda x: (severity_order.get(x.get('severity', 'low'), 0), x['type']), reverse=True)
        
        return all_anomalies
    
    def _generate_interventions(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate intervention recommendations based on anomalies."""
        interventions = []
        
        for anomaly in anomalies:
            intervention = self._get_intervention_for_anomaly(anomaly)
            if intervention:
                interventions.append(intervention)
        
        return interventions
    
    def _get_intervention_for_anomaly(self, anomaly: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get specific intervention for anomaly type."""
        anomaly_type = anomaly.get('type', '')
        severity = anomaly.get('severity', 'low')
        
        interventions = {
            'performance_drop': {
                'title': 'Address Performance Drop',
                'description': 'Schedule one-on-one meeting to understand recent performance decline',
                'urgency': 'high' if severity == 'high' else 'medium',
                'actions': [
                    'Review recent assignments and tests',
                    'Identify specific areas of difficulty',
                    'Provide additional support resources',
                    'Monitor progress over next 2 weeks'
                ]
            },
            'high_volatility': {
                'title': 'Stabilize Performance',
                'description': 'High score volatility indicates inconsistent performance',
                'urgency': 'medium',
                'actions': [
                    'Implement regular study schedule',
                    'Focus on foundational concepts',
                    'Provide consistent feedback',
                    'Consider stress management techniques'
                ]
            },
            'consistent_underperformance': {
                'title': 'Improve Academic Performance',
                'description': 'Student consistently performs below class average',
                'urgency': 'high',
                'actions': [
                    'Assess learning gaps',
                    'Provide remedial support',
                    'Consider tutoring or study groups',
                    'Regular progress monitoring'
                ]
            },
            'irregular_submission': {
                'title': 'Improve Study Habits',
                'description': 'Irregular submission patterns detected',
                'urgency': 'medium',
                'actions': [
                    'Establish regular study routine',
                    'Set clear deadlines and expectations',
                    'Provide organizational support',
                    'Monitor submission patterns'
                ]
            },
            'subject_overfocus': {
                'title': 'Balance Subject Focus',
                'description': 'Student over-focusing on specific subjects',
                'urgency': 'low',
                'actions': [
                    'Encourage balanced study approach',
                    'Highlight importance of all subjects',
                    'Provide motivation for weaker subjects',
                    'Monitor subject balance'
                ]
            },
            'subject_weakness': {
                'title': 'Strengthen Weak Subject',
                'description': f'Significant weakness in {anomaly.get("subject", "specific subject")}',
                'urgency': 'medium',
                'actions': [
                    'Provide targeted support for weak subject',
                    'Consider additional resources or tutoring',
                    'Break down complex concepts',
                    'Regular practice and assessment'
                ]
            },
            'increasing_volatility': {
                'title': 'Address Stress Indicators',
                'description': 'Increasing volatility may indicate stress or anxiety',
                'urgency': 'high',
                'actions': [
                    'Schedule counseling session',
                    'Assess personal circumstances',
                    'Provide stress management resources',
                    'Consider academic accommodations'
                ]
            },
            'performance_decline': {
                'title': 'Reverse Performance Decline',
                'description': 'Consistent performance decline detected',
                'urgency': 'high',
                'actions': [
                    'Immediate intervention required',
                    'Assess personal and academic factors',
                    'Provide intensive support',
                    'Regular monitoring and feedback'
                ]
            },
            'extreme_score_changes': {
                'title': 'Stabilize Performance',
                'description': 'Unusual extreme score changes detected',
                'urgency': 'medium',
                'actions': [
                    'Investigate causes of extreme changes',
                    'Provide consistent study environment',
                    'Implement regular assessment schedule',
                    'Monitor for external factors'
                ]
            },
            'widening_performance_gap': {
                'title': 'Close Performance Gap',
                'description': 'Performance gap with class average is widening',
                'urgency': 'medium',
                'actions': [
                    'Identify specific learning gaps',
                    'Provide targeted remediation',
                    'Consider peer tutoring',
                    'Regular progress assessment'
                ]
            }
        }
        
        if anomaly_type in interventions:
            intervention = interventions[anomaly_type].copy()
            intervention['anomaly_type'] = anomaly_type
            intervention['anomaly_severity'] = severity
            intervention['anomaly_description'] = anomaly.get('description', '')
            return intervention
        
        return None
    
    def _calculate_confidence(self, anomalies: List[Dict[str, Any]]) -> float:
        """Calculate confidence in anomaly detection."""
        if not anomalies:
            return 0.0
        
        # Base confidence on number and severity of anomalies
        severity_scores = {'high': 0.9, 'medium': 0.7, 'low': 0.5}
        total_score = sum(severity_scores.get(a.get('severity', 'low'), 0.5) for a in anomalies)
        
        # Normalize by number of anomalies
        confidence = min(total_score / len(anomalies), 0.95)
        
        return round(confidence, 2)
    
    def _fallback_detection(self, student_id: str, reason: str) -> Dict[str, Any]:
        """Fallback detection when analysis fails."""
        return {
            'student_id': student_id,
            'error': reason,
            'anomalies_detected': 0,
            'performance_anomalies': [],
            'behavioral_anomalies': [],
            'stress_indicators': [],
            'combined_analysis': [],
            'intervention_recommendations': [],
            'detection_confidence': 0.0,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _log_detection(self, student_id: str, anomalies: List[Dict[str, Any]]):
        """Log anomaly detection for audit trail."""
        self.detection_count += 1
        self.last_detection_time = datetime.now()
        
        log_entry = {
            'student_id': student_id,
            'anomalies_count': len(anomalies),
            'high_severity_count': len([a for a in anomalies if a.get('severity') == 'high']),
            'timestamp': self.last_detection_time.isoformat()
        }
        
        logger.info(f"Anomaly detection completed: {log_entry}")
    
    def get_detection_health(self) -> Dict[str, Any]:
        """Get health metrics for anomaly detection."""
        return {
            'status': 'healthy' if self.detection_count > 0 else 'unknown',
            'total_detections': self.detection_count,
            'false_positives': self.false_positives,
            'last_detection_time': self.last_detection_time.isoformat() if self.last_detection_time else None,
            'detection_settings': {
                'contamination': self.contamination,
                'sensitivity': self.sensitivity
            }
        }
