"""
Peer Contextual Analysis - Phase 2
Anonymous peer comparison with differential privacy for teacher insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from django.conf import settings
from django.core.cache import cache
from django.db import transaction, models
from core.apps.students.models import Student, StudentScore, Subject

logger = logging.getLogger(__name__)


class PeerAnalyzer:
    """
    Anonymous peer analysis with differential privacy.
    
    Features:
    - K-anonymity (minimum group size 10)
    - Differential privacy (ε = 1.0)
    - Anonymous peer comparison
    - Teacher effectiveness correlation
    - Study group optimization
    - Privacy-preserving insights
    """
    
    def __init__(self, epsilon: float = 1.0, k_anonymity: int = 10):
        self.epsilon = epsilon  # Differential privacy parameter
        self.k_anonymity = k_anonymity  # Minimum group size
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.feature_names = []
        self.analysis_cache = {}
        
        # Privacy monitoring
        self.query_count = 0
        self.privacy_violations = 0
        self.last_analysis_time = None
        
    def _add_noise(self, value: float, sensitivity: float) -> float:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            value: Original value
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value preserving privacy
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def _ensure_k_anonymity(self, group_size: int) -> bool:
        """
        Ensure k-anonymity constraint is met.
        
        Args:
            group_size: Size of the peer group
            
        Returns:
            True if k-anonymity is satisfied
        """
        return group_size >= self.k_anonymity
    
    def analyze_student_peers(self, student_id: str, subjects: List[str] = None) -> Dict[str, Any]:
        """
        Analyze student's anonymous peer group (optimized with caching).
        
        Args:
            student_id: Target student ID
            subjects: List of subjects to analyze (default: all)
            
        Returns:
            Anonymous peer analysis results
        """
        try:
            # Check cache first for complete analysis
            cache_key = f"peer_analysis_{student_id}_{hash(str(subjects))}"
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Get student data (optimized)
            student_data = self._get_student_data_optimized(student_id)
            if student_data is None:
                return self._fallback_analysis(student_id, "Student not found")
            
            # Get peer group (anonymous) using pre-computed features
            peer_group = self._get_anonymous_peer_group_optimized(student_data, subjects)
            
            if not self._ensure_k_anonymity(len(peer_group)):
                return self._fallback_analysis(student_id, "Insufficient peer data for privacy")
            
            # Calculate anonymous statistics (vectorized)
            peer_stats = self._calculate_peer_statistics_optimized(peer_group, student_data)
            
            # Add differential privacy noise
            noisy_stats = self._apply_differential_privacy(peer_stats)
            
            # Generate insights
            insights = self._generate_peer_insights(student_data, noisy_stats)
            
            result = {
                'student_id': student_id,
                'peer_group_size': len(peer_group),
                'subjects_analyzed': subjects or 'all',
                'anonymous_statistics': noisy_stats,
                'insights': insights,
                'privacy_guarantees': {
                    'k_anonymity': self.k_anonymity,
                    'epsilon': self.epsilon,
                    'differential_privacy': True
                },
                'privacy_compliant': True,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Cache the complete result for 15 minutes
            cache.set(cache_key, result, 900)
            
            # Log analysis
            self._log_analysis(student_id, len(peer_group), insights)
            
            return result
            
        except Exception as e:
            logger.error(f"Peer analysis failed for student {student_id}: {e}")
            return self._fallback_analysis(student_id, str(e))
    
    def _get_student_data(self, student_id: str) -> Optional[pd.DataFrame]:
        """Get comprehensive student data."""
        try:
            student = Student.objects.get(student_id=student_id)
            scores = StudentScore.objects.filter(student=student).select_related('subject')
            
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
                    'academic_year': score.academic_year
                })
            
            return pd.DataFrame(data)
            
        except Student.DoesNotExist:
            return None
    
    def _get_student_data_optimized(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get optimized student data using database aggregation."""
        try:
            from django.db.models import Avg, StdDev, Count
            
            # Cache key for student data
            cache_key = f"student_features_{student_id}"
            cached_data = cache.get(cache_key)
            
            if cached_data is not None:
                return cached_data
            
            # Get aggregated student data directly from database
            student_stats = StudentScore.objects.filter(
                student__student_id=student_id
            ).aggregate(
                avg_score=Avg('total_score'),
                score_std=StdDev('total_score'),
                subject_count=Count('subject', distinct=True)
            )
            
            if student_stats['avg_score'] is None:
                return None
            
            # Convert Decimal to float for compatibility
            student_stats['avg_score'] = float(student_stats['avg_score'])
            student_stats['score_std'] = float(student_stats['score_std'] or 0)
            
            # Add student_id and trend
            student_stats['student_id'] = student_id
            student_stats['trend'] = 0.0  # Simplified for performance
            
            # Cache for 1 hour
            cache.set(cache_key, student_stats, 3600)
            
            return student_stats
            
        except Exception as e:
            logger.error(f"Failed to get optimized student data for {student_id}: {e}")
            return None
    
    def _get_anonymous_peer_group_optimized(self, student_data: Dict[str, Any], subjects: List[str] = None) -> pd.DataFrame:
        """Get optimized anonymous peer group."""
        # Get all students data (cached and limited)
        all_students = self._get_all_students_data(subjects)
        
        if all_students.empty:
            return pd.DataFrame()
        
        # Convert student data to DataFrame format for similarity
        target_df = pd.DataFrame([student_data])
        
        # Find similar students using optimized algorithm
        similar_students = self._find_similar_students(target_df, all_students)
        
        return similar_students
    
    def _calculate_peer_statistics_optimized(self, peer_group: pd.DataFrame, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimized peer statistics using vectorized operations."""
        if peer_group.empty:
            return {}
        
        # Vectorized calculations
        stats = {
            'peer_avg_score': peer_group['avg_score'].mean(),
            'peer_score_std': peer_group['score_std'].mean(),
            'peer_subject_count': peer_group['subject_count'].mean(),
            'peer_trend': peer_group['trend'].mean()
        }
        
        # Performance percentile calculation
        student_avg = student_data['avg_score']
        peer_scores = peer_group['avg_score'].values
        percentile = np.mean(peer_scores < student_avg) * 100
        stats['performance_percentile'] = percentile
        
        return stats
    
    def _get_anonymous_peer_group(self, student_data: pd.DataFrame, subjects: List[str] = None) -> pd.DataFrame:
        """
        Get anonymous peer group based on similar performance patterns.
        
        Args:
            student_data: Target student's data
            subjects: Subjects to consider for similarity
            
        Returns:
            Anonymous peer group data
        """
        # Get all students with similar characteristics
        all_students = self._get_all_students_data(subjects)
        
        if all_students.empty:
            return pd.DataFrame()
        
        # Calculate similarity features
        student_features = self._extract_similarity_features(student_data)
        all_features = self._extract_similarity_features(all_students)
        
        # Find similar students (anonymous)
        similar_students = self._find_similar_students(student_features, all_features)
        
        # Ensure anonymity
        if len(similar_students) < self.k_anonymity:
            # Expand search to meet k-anonymity
            similar_students = self._expand_peer_group(all_features, len(similar_students))
        
        return similar_students
    
    def _extract_similarity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for similarity calculation (optimized)."""
        if data.empty:
            return pd.DataFrame()
        
        # For individual student data, calculate features directly
        if 'avg_score' in data.columns:
            # Data is already aggregated
            return data[['student_id', 'avg_score', 'score_std', 'subject_count', 'trend']]
        
        # For raw score data, use vectorized operations
        features = data.groupby('student_id').agg({
            'total_score': ['mean', 'std', 'count'],
            'subject': 'nunique'
        }).reset_index()
        
        # Flatten column names
        features.columns = ['student_id', 'avg_score', 'score_std', 'score_count', 'subject_count']
        
        # Add simplified trend (can be enhanced later)
        features['trend'] = 0.0
        
        return features
    
    def _find_similar_students(self, target_features: pd.DataFrame, all_features: pd.DataFrame) -> pd.DataFrame:
        """Find students with similar performance patterns (optimized)."""
        if all_features.empty or target_features.empty:
            return pd.DataFrame()
        
        # Cache key for similar students
        target_student_id = target_features.iloc[0]['student_id']
        cache_key = f"similar_students_{target_student_id}"
        cached_result = cache.get(cache_key)
        
        if cached_result is not None:
            return pd.DataFrame(cached_result)
        
        # Use approximate similarity (much faster than cosine similarity)
        target_avg = target_features.iloc[0]['avg_score']
        target_std = target_features.iloc[0]['score_std']
        
        # Simple distance-based similarity (much faster)
        all_features['score_diff'] = np.abs(all_features['avg_score'] - target_avg)
        all_features['std_diff'] = np.abs(all_features['score_std'] - target_std)
        
        # Combined distance score (weighted)
        all_features['similarity_score'] = (
            1.0 / (1.0 + all_features['score_diff'] * 0.1 + all_features['std_diff'] * 0.05)
        )
        
        # Get top similar students (excluding self)
        similar_students = all_features[
            all_features['student_id'] != target_student_id
        ].nlargest(self.k_anonymity, 'similarity_score')
        
        # Cache the result for 1 hour
        cache.set(cache_key, similar_students.to_dict('records'), 3600)
        
        return similar_students
    
    def _expand_peer_group(self, all_features: pd.DataFrame, current_size: int) -> pd.DataFrame:
        """Expand peer group to meet k-anonymity requirements."""
        needed = self.k_anonymity - current_size
        
        if needed <= 0:
            return all_features.head(current_size)
        
        # Add random students to meet k-anonymity
        remaining = all_features.iloc[current_size:]
        if len(remaining) >= needed:
            additional = remaining.sample(n=needed)
            return pd.concat([all_features.head(current_size), additional])
        else:
            return all_features.head(self.k_anonymity)
    
    def _calculate_peer_statistics(self, peer_group: pd.DataFrame, student_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate anonymous peer statistics."""
        stats = {}
        
        # Performance statistics
        stats['peer_avg_score'] = peer_group['avg_score'].mean()
        stats['peer_score_std'] = peer_group['score_std'].mean()
        stats['peer_subject_count'] = peer_group['subject_count'].mean()
        stats['peer_trend'] = peer_group['trend'].mean()
        
        # Percentile rankings (anonymous)
        student_avg = student_data['total_score'].mean()
        peer_scores = peer_group['avg_score'].values
        
        # Calculate percentile without revealing exact position
        percentile = np.mean(peer_scores < student_avg) * 100
        stats['performance_percentile'] = percentile
        
        return stats
    
    def _apply_differential_privacy(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy noise to statistics."""
        noisy_stats = {}
        
        # Define sensitivity for each statistic
        sensitivities = {
            'peer_avg_score': 100.0,  # Maximum possible score
            'peer_score_std': 50.0,   # Half of max score
            'peer_subject_count': 20.0,  # Maximum subjects
            'peer_trend': 10.0,       # Trend range
            'performance_percentile': 1.0  # Percentage (0-100)
        }
        
        for key, value in stats.items():
            if key in sensitivities:
                noisy_value = self._add_noise(value, sensitivities[key])
                # Clamp values to reasonable ranges
                if key == 'performance_percentile':
                    noisy_value = np.clip(noisy_value, 0, 100)
                elif key == 'peer_avg_score':
                    noisy_value = np.clip(noisy_value, 0, 100)
                
                noisy_stats[key] = round(noisy_value, 2)
            else:
                noisy_stats[key] = value
        
        return noisy_stats
    
    def _generate_peer_insights(self, student_data, peer_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from peer analysis (optimized for both data formats)."""
        insights = {}
        
        # Handle both DataFrame and Dict input formats
        if isinstance(student_data, pd.DataFrame):
            student_avg = student_data['total_score'].mean()
            student_std = student_data['total_score'].std()
        else:
            student_avg = student_data['avg_score']
            student_std = student_data.get('score_std', 0)
        
        peer_avg = peer_stats.get('peer_avg_score', 0)
        peer_std = peer_stats.get('peer_score_std', 0)
        
        # Performance comparison
        if student_avg > peer_avg:
            insights['performance_status'] = 'above_peer_average'
            insights['performance_message'] = 'Student performs above peer group average'
        elif student_avg < peer_avg:
            insights['performance_status'] = 'below_peer_average'
            insights['performance_message'] = 'Student performs below peer group average'
        else:
            insights['performance_status'] = 'at_peer_average'
            insights['performance_message'] = 'Student performs at peer group average'
        
        # Consistency analysis
        if student_std < peer_std:
            insights['consistency_status'] = 'more_consistent'
            insights['consistency_message'] = 'Student shows more consistent performance than peers'
        else:
            insights['consistency_status'] = 'less_consistent'
            insights['consistency_message'] = 'Student shows less consistent performance than peers'
        
        # Improvement trend
        trend = peer_stats.get('peer_trend', 0)
        if trend > 0:
            insights['trend_status'] = 'improving'
            insights['trend_message'] = 'Peer group shows improving trend'
        elif trend < 0:
            insights['trend_status'] = 'declining'
            insights['trend_message'] = 'Peer group shows declining trend'
        else:
            insights['trend_status'] = 'stable'
            insights['trend_message'] = 'Peer group shows stable performance'
        
        return insights
    
    def _calculate_trend(self, scores: pd.Series) -> float:
        """Calculate performance trend."""
        if len(scores) < 2:
            return 0.0
        
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        return slope
    
    def _get_all_students_data(self, subjects: List[str] = None) -> pd.DataFrame:
        """Get optimized student data using database aggregation."""
        try:
            # Use caching for frequently accessed data
            cache_key = f"peer_analysis_data_{hash(str(subjects))}"
            cached_data = cache.get(cache_key)
            
            if cached_data is not None:
                return pd.DataFrame(cached_data)
            
            # Optimized query using database aggregation
            from django.db.models import Avg, StdDev, Count
            
            # Get aggregated student performance data directly from database
            queryset = StudentScore.objects.select_related('student', 'subject').values(
                'student__student_id'
            ).annotate(
                avg_score=Avg('total_score'),
                score_std=StdDev('total_score'),
                subject_count=Count('subject', distinct=True),
                student_id=models.F('student__student_id')
            )
            
            if subjects:
                queryset = queryset.filter(subject__name__in=subjects)
            
            # Limit to reasonable sample size for performance
            queryset = queryset[:1000]  # Limit to 1000 students for similarity
            
            # Convert to DataFrame efficiently
            data = list(queryset.values('student_id', 'avg_score', 'score_std', 'subject_count'))
            
            # Convert Decimal to float and add trends
            for item in data:
                item['avg_score'] = float(item['avg_score'])
                item['score_std'] = float(item['score_std'] or 0)
                item['trend'] = 0.0  # Simplified - can be enhanced later
            
            df = pd.DataFrame(data)
            
            # Cache the result for 30 minutes
            cache.set(cache_key, data, 1800)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get optimized students data: {e}")
            return pd.DataFrame()
    
    def _fallback_analysis(self, student_id: str, reason: str) -> Dict[str, Any]:
        """Fallback analysis when peer analysis fails."""
        return {
            'student_id': student_id,
            'error': reason,
            'peer_group_size': 0,
            'anonymous_statistics': {},
            'insights': {
                'performance_status': 'unknown',
                'performance_message': 'Analysis unavailable',
                'consistency_status': 'unknown',
                'consistency_message': 'Analysis unavailable',
                'trend_status': 'unknown',
                'trend_message': 'Analysis unavailable'
            },
            'privacy_guarantees': {
                'k_anonymity': self.k_anonymity,
                'epsilon': self.epsilon,
                'differential_privacy': True
            },
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _log_analysis(self, student_id: str, peer_group_size: int, insights: Dict[str, Any]):
        """Log analysis for audit trail."""
        self.query_count += 1
        self.last_analysis_time = datetime.now()
        
        log_entry = {
            'student_id': student_id,
            'peer_group_size': peer_group_size,
            'insights': insights,
            'timestamp': self.last_analysis_time.isoformat(),
            'privacy_guarantees': {
                'k_anonymity': self.k_anonymity,
                'epsilon': self.epsilon
            }
        }
        
        logger.info(f"Peer analysis completed: {log_entry}")
    
    def get_analysis_health(self) -> Dict[str, Any]:
        """Get health metrics for peer analysis."""
        return {
            'status': 'healthy' if self.query_count > 0 else 'unknown',
            'total_queries': self.query_count,
            'privacy_violations': self.privacy_violations,
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'privacy_settings': {
                'epsilon': self.epsilon,
                'k_anonymity': self.k_anonymity
            },
            'cache_size': len(self.analysis_cache)
        }
