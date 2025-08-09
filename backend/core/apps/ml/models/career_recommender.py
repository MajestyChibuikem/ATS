"""
Career Recommendation Engine - Phase 2 (Enhanced Production-Ready)
Strength-based career guidance using performance patterns and subject correlations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
from functools import lru_cache
warnings.filterwarnings('ignore')

from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from core.apps.students.models import Student, StudentScore, Subject

logger = logging.getLogger(__name__)


class MatchLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class GapLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CareerMatch:
    """Data class for career match results."""
    career: str
    match_score: float
    match_level: MatchLevel
    requirements_met: Dict[str, Any]
    core_subjects: List[str]
    preferred_subjects: List[str]
    min_score_requirement: float
    career_prospects: List[str]
    skill_requirements: List[str]
    recommendation_reason: str
    confidence_score: float


@dataclass
class SkillGap:
    """Data class for skill gap analysis."""
    skill: str
    gap_level: GapLevel
    reason: str
    performance_score: float
    relevant_subjects: List[str]
    improvement_suggestions: List[str]


class CareerRecommendationError(Exception):
    """Custom exception for career recommendation errors."""
    pass


class CareerRecommender:
    """
    Enhanced Career recommendation engine with production-ready features.
    
    Features:
    - Production-ready architecture with comprehensive error handling
    - Caching and performance optimization
    - Enhanced matching algorithm with market factors
    - Data quality assessment and validation
    - Monitoring and health checks
    - Async processing capabilities
    """
    
    # Configuration from settings
    CACHE_TIMEOUT = getattr(settings, 'CAREER_RECOMMENDATION_SETTINGS', {}).get('CACHE_TIMEOUT', 3600)
    MIN_MATCH_THRESHOLD = getattr(settings, 'CAREER_RECOMMENDATION_SETTINGS', {}).get('MIN_MATCH_THRESHOLD', 0.3)
    MAX_RECOMMENDATIONS = getattr(settings, 'CAREER_RECOMMENDATION_SETTINGS', {}).get('MAX_RECOMMENDATIONS', 5)
    MIN_DATA_POINTS = getattr(settings, 'CAREER_RECOMMENDATION_SETTINGS', {}).get('MIN_DATA_POINTS', 3)
    CACHE_PREFIX = "career_rec_"
    
    def __init__(self):
        self.career_pathways = self._load_career_pathways()
        self.subject_correlations = self._load_subject_correlations()
        self.university_requirements = self._load_university_requirements()
        self.scaler = StandardScaler()
        self.classifier = None  # Lazy loading
        
        # Enhanced career clusters with weights
        self.career_clusters = {
            'stem': {
                'careers': ['Engineering', 'Computer Science', 'Medicine', 'Physics', 'Chemistry'],
                'weight': 1.0
            },
            'business': {
                'careers': ['Business Administration', 'Economics', 'Finance', 'Marketing', 'Accounting'],
                'weight': 0.9
            },
            'arts_humanities': {
                'careers': ['Literature', 'History', 'Philosophy', 'Languages', 'Arts'],
                'weight': 0.8
            },
            'social_sciences': {
                'careers': ['Psychology', 'Sociology', 'Political Science', 'Education', 'Social Work'],
                'weight': 0.85
            },
            'health_sciences': {
                'careers': ['Nursing', 'Pharmacy', 'Public Health', 'Physiotherapy', 'Nutrition'],
                'weight': 0.95
            }
        }
        
        # Monitoring with better structure
        self._metrics = {
            'total_recommendations': 0,
            'successful_recommendations': 0,
            'failed_recommendations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_recommendation_time': None,
            'average_processing_time': 0.0,
            'error_count': 0,
            'validation_errors': 0
        }
        
    def _load_career_pathways(self) -> Dict[str, Dict[str, Any]]:
        """Load career pathway definitions with enhanced data."""
        return {
            'Engineering': {
                'core_subjects': ['Mathematics', 'Physics', 'Chemistry'],
                'preferred_subjects': ['Further Mathematics', 'Computer Science'],
                'min_score': 70,
                'skill_requirements': ['analytical_thinking', 'problem_solving', 'technical_skills'],
                'career_prospects': ['Software Engineer', 'Civil Engineer', 'Mechanical Engineer', 'Electrical Engineer'],
                'industry_demand': 0.9,  # High demand
                'salary_potential': 0.95,  # High salary potential
                'growth_outlook': 0.85  # Good growth
            },
            'Medicine': {
                'core_subjects': ['Biology', 'Chemistry', 'Physics'],
                'preferred_subjects': ['Mathematics'],
                'min_score': 85,
                'skill_requirements': ['scientific_thinking', 'attention_to_detail', 'empathy'],
                'career_prospects': ['Medical Doctor', 'Surgeon', 'Researcher', 'Public Health Specialist'],
                'industry_demand': 0.95,
                'salary_potential': 1.0,
                'growth_outlook': 0.9
            },
            'Computer Science': {
                'core_subjects': ['Mathematics', 'Computer Science'],
                'preferred_subjects': ['Physics', 'Further Mathematics'],
                'min_score': 75,
                'skill_requirements': ['logical_thinking', 'programming', 'problem_solving'],
                'career_prospects': ['Software Developer', 'Data Scientist', 'Systems Analyst', 'AI Engineer'],
                'industry_demand': 1.0,
                'salary_potential': 0.9,
                'growth_outlook': 1.0
            },
            'Business Administration': {
                'core_subjects': ['Mathematics', 'Economics'],
                'preferred_subjects': ['Accounting', 'Commerce'],
                'min_score': 65,
                'skill_requirements': ['leadership', 'communication', 'analytical_thinking'],
                'career_prospects': ['Business Manager', 'Entrepreneur', 'Consultant', 'Marketing Manager'],
                'industry_demand': 0.8,
                'salary_potential': 0.85,
                'growth_outlook': 0.8
            },
            'Law': {
                'core_subjects': ['English Language', 'Literature'],
                'preferred_subjects': ['Government', 'History'],
                'min_score': 75,
                'skill_requirements': ['critical_thinking', 'communication', 'research_skills'],
                'career_prospects': ['Lawyer', 'Legal Consultant', 'Judge', 'Legal Researcher'],
                'industry_demand': 0.75,
                'salary_potential': 0.9,
                'growth_outlook': 0.7
            },
            'Psychology': {
                'core_subjects': ['Biology', 'English Language'],
                'preferred_subjects': ['Mathematics', 'Literature'],
                'min_score': 70,
                'skill_requirements': ['empathy', 'research_skills', 'communication'],
                'career_prospects': ['Clinical Psychologist', 'Counselor', 'Researcher', 'HR Specialist'],
                'industry_demand': 0.85,
                'salary_potential': 0.8,
                'growth_outlook': 0.85
            },
            'Education': {
                'core_subjects': ['English Language', 'Mathematics'],
                'preferred_subjects': ['Literature', 'History'],
                'min_score': 65,
                'skill_requirements': ['communication', 'patience', 'leadership'],
                'career_prospects': ['Teacher', 'Educational Administrator', 'Curriculum Developer', 'Educational Consultant'],
                'industry_demand': 0.9,
                'salary_potential': 0.7,
                'growth_outlook': 0.8
            },
            'Agriculture': {
                'core_subjects': ['Biology', 'Chemistry'],
                'preferred_subjects': ['Agricultural Science', 'Geography'],
                'min_score': 60,
                'skill_requirements': ['practical_skills', 'scientific_thinking', 'management'],
                'career_prospects': ['Agricultural Engineer', 'Farm Manager', 'Agronomist', 'Food Scientist'],
                'industry_demand': 0.7,
                'salary_potential': 0.75,
                'growth_outlook': 0.8
            }
        }
    
    def _load_subject_correlations(self) -> Dict[str, Dict[str, float]]:
        """Load enhanced subject correlation matrix."""
        correlations = {
            'Mathematics': {
                'Physics': 0.85, 'Chemistry': 0.70, 'Computer Science': 0.80,
                'Economics': 0.65, 'Accounting': 0.60, 'Engineering': 0.90
            },
            'Physics': {
                'Mathematics': 0.85, 'Chemistry': 0.75, 'Computer Science': 0.70,
                'Engineering': 0.90, 'Medicine': 0.80
            },
            'Chemistry': {
                'Mathematics': 0.70, 'Physics': 0.75, 'Biology': 0.80,
                'Medicine': 0.90, 'Agriculture': 0.75
            },
            'Biology': {
                'Chemistry': 0.80, 'Physics': 0.60, 'Medicine': 0.90,
                'Agriculture': 0.85, 'Psychology': 0.70
            },
            'English Language': {
                'Literature': 0.90, 'Law': 0.85, 'Psychology': 0.70,
                'Education': 0.80, 'Business': 0.65
            },
            'Literature': {
                'English Language': 0.90, 'Law': 0.80, 'Psychology': 0.65,
                'Education': 0.85, 'Arts': 0.90
            }
        }
        
        # Validate correlation matrix
        self._validate_correlations(correlations)
        return correlations
    
    def _validate_correlations(self, correlations: Dict[str, Dict[str, float]]):
        """Validate correlation matrix for consistency."""
        for subject1, corr_dict in correlations.items():
            for subject2, correlation in corr_dict.items():
                if not 0 <= correlation <= 1:
                    raise ValueError(f"Invalid correlation {correlation} between {subject1} and {subject2}")
                
                # Check symmetry where possible
                if (subject2 in correlations and 
                    subject1 in correlations[subject2] and 
                    abs(correlations[subject2][subject1] - correlation) > 0.1):
                    logger.warning(f"Asymmetric correlation between {subject1} and {subject2}")
    
    def _load_university_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Load enhanced university admission requirements."""
        return {
            'University of Lagos': {
                'Engineering': {
                    'min_score': 75, 
                    'core_subjects': ['Mathematics', 'Physics'],
                    'cutoff_trend': [75, 76, 78],  # Last 3 years
                    'competition_level': 0.9  # High competition
                },
                'Medicine': {
                    'min_score': 85, 
                    'core_subjects': ['Biology', 'Chemistry'],
                    'cutoff_trend': [85, 87, 88],
                    'competition_level': 0.95
                },
                'Law': {
                    'min_score': 75, 
                    'core_subjects': ['English Language', 'Literature'],
                    'cutoff_trend': [75, 75, 76],
                    'competition_level': 0.85
                },
                'Business': {
                    'min_score': 65, 
                    'core_subjects': ['Mathematics', 'Economics'],
                    'cutoff_trend': [65, 66, 67],
                    'competition_level': 0.75
                }
            },
            'University of Ibadan': {
                'Engineering': {
                    'min_score': 78, 
                    'core_subjects': ['Mathematics', 'Physics'],
                    'cutoff_trend': [78, 79, 80],
                    'competition_level': 0.9
                },
                'Medicine': {
                    'min_score': 88, 
                    'core_subjects': ['Biology', 'Chemistry'],
                    'cutoff_trend': [88, 89, 90],
                    'competition_level': 0.95
                }
            },
            'Obafemi Awolowo University': {
                'Engineering': {
                    'min_score': 76, 
                    'core_subjects': ['Mathematics', 'Physics'],
                    'cutoff_trend': [76, 77, 78],
                    'competition_level': 0.85
                },
                'Medicine': {
                    'min_score': 86, 
                    'core_subjects': ['Biology', 'Chemistry'],
                    'cutoff_trend': [86, 87, 88],
                    'competition_level': 0.9
                }
            }
        }
    
    @lru_cache(maxsize=128)
    def _get_cached_student_data(self, student_id: str, cache_key: str) -> Optional[str]:
        """Get cached student data with LRU cache."""
        cache_full_key = f"{self.CACHE_PREFIX}student_{student_id}_{cache_key}"
        return cache.get(cache_full_key)
    
    def _set_cached_student_data(self, student_id: str, cache_key: str, data: str):
        """Set cached student data."""
        cache_full_key = f"{self.CACHE_PREFIX}student_{student_id}_{cache_key}"
        cache.set(cache_full_key, data, self.CACHE_TIMEOUT)
    
    def recommend_careers(self, student_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive career recommendations with caching and error handling.
        
        Args:
            student_id: Target student ID
            force_refresh: Skip cache and generate fresh recommendations
            
        Returns:
            Career recommendation analysis
        """
        start_time = datetime.now()
        
        try:
            # Validate input
            if not student_id or not isinstance(student_id, str):
                raise CareerRecommendationError("Invalid student ID provided")
            
            # Check cache first unless forced refresh
            if not force_refresh:
                cached_result = self._get_cached_recommendation(student_id)
                if cached_result:
                    self._metrics['cache_hits'] += 1
                    return cached_result
            
            self._metrics['cache_misses'] += 1
            
            # Validate student exists
            if not self._validate_student(student_id):
                raise CareerRecommendationError(f"Student {student_id} not found")
            
            # Get student performance data
            student_data = self._get_student_data(student_id)
            if student_data is None or len(student_data) < self.MIN_DATA_POINTS:
                return self._fallback_recommendation(student_id, "Insufficient performance data")
            
            # Perform comprehensive analysis
            analysis_result = self._perform_comprehensive_analysis(student_data)
            
            # Add metadata
            analysis_result.update({
                'student_id': student_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality_score': self._calculate_data_quality(student_data),
                'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000)
            })
            
            # Cache the result
            self._cache_recommendation(student_id, analysis_result)
            
            # Update metrics
            self._update_metrics('success', start_time)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Career recommendation failed for student {student_id}: {e}", exc_info=True)
            self._update_metrics('failure', start_time)
            return self._fallback_recommendation(student_id, str(e))
    
    def _validate_student(self, student_id: str) -> bool:
        """Validate that student exists."""
        try:
            Student.objects.get(student_id=student_id)
            return True
        except Student.DoesNotExist:
            return False
    
    def _get_cached_recommendation(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get cached recommendation if available and fresh."""
        cache_key = f"{self.CACHE_PREFIX}recommendation_{student_id}"
        cached_data = cache.get(cache_key)
        
        if cached_data:
            # Check if cache is still fresh (within 6 hours for recommendations)
            cached_time = datetime.fromisoformat(cached_data.get('analysis_timestamp', ''))
            if datetime.now() - cached_time < timedelta(hours=6):
                return cached_data
        
        return None
    
    def _cache_recommendation(self, student_id: str, recommendation: Dict[str, Any]):
        """Cache recommendation result."""
        cache_key = f"{self.CACHE_PREFIX}recommendation_{student_id}"
        cache.set(cache_key, recommendation, self.CACHE_TIMEOUT)
    
    def _perform_comprehensive_analysis(self, student_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive career analysis."""
        # Analyze student strengths and weaknesses
        strengths_analysis = self._analyze_strengths(student_data)
        weaknesses_analysis = self._analyze_weaknesses(student_data)
        
        # Generate career recommendations with enhanced matching
        career_recommendations = self._generate_enhanced_career_recommendations(
            student_data, strengths_analysis
        )
        
        # Calculate university admission probabilities with trends
        university_probabilities = self._calculate_enhanced_university_probabilities(student_data)
        
        # Generate detailed skill gap analysis
        skill_gaps = self._analyze_detailed_skill_gaps(student_data, career_recommendations)
        
        # Generate improvement recommendations
        improvement_plan = self._generate_improvement_plan(weaknesses_analysis, skill_gaps)
        
        return {
            'strengths_analysis': strengths_analysis,
            'weaknesses_analysis': weaknesses_analysis,
            'career_recommendations': career_recommendations,
            'university_probabilities': university_probabilities,
            'skill_gaps': skill_gaps,
            'improvement_plan': improvement_plan,
            'recommendation_confidence': self._calculate_enhanced_confidence(student_data, career_recommendations)
        }
    
    def _calculate_data_quality(self, student_data: pd.DataFrame) -> float:
        """Calculate data quality score."""
        if student_data.empty:
            return 0.0
        
        # Factors affecting data quality
        data_points = len(student_data)
        subjects_count = student_data['subject'].nunique()
        terms_count = student_data['term'].nunique()
        completeness = 1 - (student_data.isnull().sum().sum() / (len(student_data) * len(student_data.columns)))
        
        # Calculate quality score
        quality_score = (
            min(data_points / 20, 1.0) * 0.3 +  # More data points
            min(subjects_count / 8, 1.0) * 0.3 +  # More subjects
            min(terms_count / 6, 1.0) * 0.2 +  # Multiple terms
            completeness * 0.2  # Data completeness
        )
        
        return round(quality_score, 2)
    
    def _generate_enhanced_career_recommendations(self, student_data: pd.DataFrame, strengths: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate enhanced career recommendations with market factors."""
        recommendations = []
        strong_subjects = [s['subject'] for s in strengths.get('strong_subjects', [])]
        
        for career, requirements in self.career_pathways.items():
            match_result = self._calculate_enhanced_career_match(
                student_data, career, requirements, strong_subjects
            )
            
            if match_result.match_score > self.MIN_MATCH_THRESHOLD:
                # Convert to dict for serialization
                recommendation_dict = {
                    'career': match_result.career,
                    'match_score': match_result.match_score,
                    'match_level': match_result.match_level.value,
                    'requirements_met': match_result.requirements_met,
                    'core_subjects': match_result.core_subjects,
                    'preferred_subjects': match_result.preferred_subjects,
                    'min_score_requirement': match_result.min_score_requirement,
                    'career_prospects': match_result.career_prospects,
                    'skill_requirements': match_result.skill_requirements,
                    'recommendation_reason': match_result.recommendation_reason,
                    'confidence_score': match_result.confidence_score,
                    'market_factors': {
                        'industry_demand': requirements.get('industry_demand', 0.5),
                        'salary_potential': requirements.get('salary_potential', 0.5),
                        'growth_outlook': requirements.get('growth_outlook', 0.5)
                    }
                }
                
                recommendations.append(recommendation_dict)
        
        # Sort by enhanced scoring (match score + market factors)
        recommendations.sort(
            key=lambda x: x['match_score'] * 0.7 + 
                         (x['market_factors']['industry_demand'] + 
                          x['market_factors']['growth_outlook']) * 0.15,
            reverse=True
        )
        
        return recommendations[:self.MAX_RECOMMENDATIONS]
    
    def _calculate_enhanced_career_match(self, student_data: pd.DataFrame, career: str, 
                                       requirements: Dict[str, Any], strong_subjects: List[str]) -> CareerMatch:
        """Calculate enhanced career match with confidence scoring."""
        # Core matching logic (similar to original but enhanced)
        match_score = self._calculate_career_match(student_data, career, requirements, strong_subjects)
        
        # Calculate confidence based on data quality and match factors
        confidence_score = self._calculate_match_confidence(student_data, requirements, match_score)
        
        # Determine match level
        if match_score >= 0.8:
            match_level = MatchLevel.EXCELLENT
        elif match_score >= 0.6:
            match_level = MatchLevel.GOOD
        elif match_score >= 0.4:
            match_level = MatchLevel.FAIR
        else:
            match_level = MatchLevel.POOR
        
        return CareerMatch(
            career=career,
            match_score=match_score,
            match_level=match_level,
            requirements_met=self._check_requirements(student_data, requirements),
            core_subjects=requirements['core_subjects'],
            preferred_subjects=requirements.get('preferred_subjects', []),
            min_score_requirement=requirements.get('min_score', 0),
            career_prospects=requirements.get('career_prospects', []),
            skill_requirements=requirements.get('skill_requirements', []),
            recommendation_reason=self._get_recommendation_reason(student_data, career, requirements),
            confidence_score=confidence_score
        )
    
    def _calculate_match_confidence(self, student_data: pd.DataFrame, 
                                  requirements: Dict[str, Any], match_score: float) -> float:
        """Calculate confidence in career match."""
        # Base confidence on data quality
        data_quality = self._calculate_data_quality(student_data)
        
        # Adjust based on match score consistency
        score_consistency = self._calculate_score_consistency(student_data)
        
        # Adjust based on requirement coverage
        requirement_coverage = len([
            s for s in requirements.get('core_subjects', [])
            if self._has_subject_performance(student_data, s)
        ]) / max(len(requirements.get('core_subjects', [])), 1)
        
        confidence = (
            data_quality * 0.4 +
            score_consistency * 0.3 +
            requirement_coverage * 0.3
        )
        
        return round(min(confidence, 0.95), 2)
    
    def _calculate_score_consistency(self, student_data: pd.DataFrame) -> float:
        """Calculate consistency of student performance."""
        if len(student_data) < 2:
            return 0.5
        
        # Calculate coefficient of variation for each subject
        subject_consistency = []
        for subject in student_data['subject'].unique():
            subject_scores = student_data[student_data['subject'] == subject]['total_score']
            if len(subject_scores) > 1 and subject_scores.mean() > 0:
                cv = subject_scores.std() / subject_scores.mean()
                consistency = max(0, 1 - cv)  # Lower CV = higher consistency
                subject_consistency.append(consistency)
        
        return np.mean(subject_consistency) if subject_consistency else 0.5
    
    def _generate_improvement_plan(self, weaknesses: Dict[str, Any], 
                                 skill_gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate personalized improvement plan."""
        improvement_plan = {
            'priority_areas': [],
            'short_term_goals': [],
            'long_term_goals': [],
            'resource_recommendations': []
        }
        
        # Priority areas from weaknesses
        weak_subjects = weaknesses.get('weak_subjects', [])[:3]  # Top 3 weaknesses
        for weakness in weak_subjects:
            improvement_plan['priority_areas'].append({
                'area': weakness['subject'],
                'current_score': weakness['student_average'],
                'target_improvement': weakness['class_average'] + 5,
                'priority_level': 'high' if weakness['disadvantage'] > 15 else 'medium'
            })
        
        # Short-term goals (next 3-6 months)
        improvement_plan['short_term_goals'] = [
            f"Improve {weakness['subject']} by {min(10, weakness['disadvantage'])} points"
            for weakness in weak_subjects[:2]
        ]
        
        # Long-term goals (6-12 months)
        improvement_plan['long_term_goals'] = [
            "Achieve consistent performance above class average",
            "Strengthen top 3 subjects for career preparation",
            "Develop identified skill gaps"
        ]
        
        return improvement_plan
    
    def _update_metrics(self, result_type: str, start_time: datetime):
        """Update performance metrics."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self._metrics['total_recommendations'] += 1
        if result_type == 'success':
            self._metrics['successful_recommendations'] += 1
        else:
            self._metrics['failed_recommendations'] += 1
            self._metrics['error_count'] += 1
        
        # Update average processing time
        total_recs = self._metrics['total_recommendations']
        current_avg = self._metrics['average_processing_time']
        self._metrics['average_processing_time'] = (
            (current_avg * (total_recs - 1) + processing_time) / total_recs
        )
        
        self._metrics['last_recommendation_time'] = datetime.now()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        success_rate = (
            self._metrics['successful_recommendations'] / 
            max(self._metrics['total_recommendations'], 1)
        )
        
        cache_hit_rate = (
            self._metrics['cache_hits'] / 
            max(self._metrics['cache_hits'] + self._metrics['cache_misses'], 1)
        )
        
        return {
            'status': 'healthy' if self._metrics['total_recommendations'] == 0 or success_rate > 0.9 else 'degraded',
            'metrics': self._metrics.copy(),
            'success_rate': round(success_rate, 3),
            'cache_hit_rate': round(cache_hit_rate, 3),
            'career_pathways_count': len(self.career_pathways),
            'university_count': len(self.university_requirements),
            'system_status': 'healthy' if self._metrics['total_recommendations'] == 0 or success_rate > 0.9 else 'degraded',
            'error_rate': round(self._metrics['error_count'] / max(self._metrics['total_recommendations'], 1), 3)
        }
    
    # Keep existing methods but add error handling and validation
    def _calculate_career_match(self, student_data: pd.DataFrame, career: str, requirements: Dict[str, Any], strong_subjects: List[str]) -> float:
        """Calculate career match with enhanced validation."""
        try:
            match_score = 0.0
            
            # Validate inputs
            if student_data.empty:
                return 0.0
            
            core_subjects = requirements.get('core_subjects', [])
            if not core_subjects:
                logger.warning(f"No core subjects defined for career: {career}")
                return 0.0
            
            # Core subject requirements (weighted 50%)
            core_match = sum(
                1.0 if subject in strong_subjects else 
                0.5 if self._has_subject_performance(student_data, subject) else 0.0
                for subject in core_subjects
            )
            core_score = core_match / len(core_subjects)
            
            # Preferred subjects (weighted 30%)
            preferred_subjects = requirements.get('preferred_subjects', [])
            preferred_score = 0.0
            if preferred_subjects:
                preferred_match = sum(
                    1.0 if subject in strong_subjects else 
                    0.5 if self._has_subject_performance(student_data, subject) else 0.0
                    for subject in preferred_subjects
                )
                preferred_score = preferred_match / len(preferred_subjects)
            
            # Minimum score requirement (weighted 20%)
            overall_score = student_data['total_score'].mean()
            min_score = requirements.get('min_score', 0)
            score_match = min(overall_score / max(min_score, 1), 1.0)
            
            # Calculate weighted match score
            match_score = (core_score * 0.5) + (preferred_score * 0.3) + (score_match * 0.2)
            
            return round(match_score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating career match for {career}: {e}")
            return 0.0
    
    def _get_student_data(self, student_id: str) -> Optional[pd.DataFrame]:
        """Get comprehensive student performance data with database optimization."""
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
    
    def _analyze_strengths(self, student_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze student's academic strengths."""
        strengths = {}
        
        # Calculate subject performance
        subject_performance = student_data.groupby('subject').agg({
            'total_score': ['mean', 'std', 'count'],
            'class_average': 'mean'
        }).round(2)
        
        # Identify strong subjects (above class average)
        strong_subjects = []
        for subject in subject_performance.index:
            student_avg = subject_performance.loc[subject, ('total_score', 'mean')]
            class_avg = subject_performance.loc[subject, ('class_average', 'mean')]
            
            if student_avg > class_avg + 5:  # 5 points above class average
                strong_subjects.append({
                    'subject': subject,
                    'student_average': student_avg,
                    'class_average': class_avg,
                    'advantage': student_avg - class_avg,
                    'performance_level': 'excellent' if student_avg > class_avg + 10 else 'good'
                })
        
        # Sort by advantage
        strong_subjects.sort(key=lambda x: x['advantage'], reverse=True)
        
        strengths['strong_subjects'] = strong_subjects
        strengths['top_strength'] = strong_subjects[0] if strong_subjects else None
        strengths['strength_count'] = len(strong_subjects)
        
        return strengths
    
    def _analyze_weaknesses(self, student_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze student's academic weaknesses."""
        weaknesses = {}
        
        # Calculate subject performance
        subject_performance = student_data.groupby('subject').agg({
            'total_score': ['mean', 'std', 'count'],
            'class_average': 'mean'
        }).round(2)
        
        # Identify weak subjects (below class average)
        weak_subjects = []
        for subject in subject_performance.index:
            student_avg = subject_performance.loc[subject, ('total_score', 'mean')]
            class_avg = subject_performance.loc[subject, ('class_average', 'mean')]
            
            if student_avg < class_avg - 5:  # 5 points below class average
                weak_subjects.append({
                    'subject': subject,
                    'student_average': student_avg,
                    'class_average': class_avg,
                    'disadvantage': class_avg - student_avg,
                    'performance_level': 'poor' if student_avg < class_avg - 10 else 'below_average'
                })
        
        # Sort by disadvantage
        weak_subjects.sort(key=lambda x: x['disadvantage'], reverse=True)
        
        weaknesses['weak_subjects'] = weak_subjects
        weaknesses['biggest_weakness'] = weak_subjects[0] if weak_subjects else None
        weaknesses['weakness_count'] = len(weak_subjects)
        
        return weaknesses
    
    def _calculate_enhanced_university_probabilities(self, student_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate university admission probabilities with trends."""
        probabilities = {}
        overall_score = student_data['total_score'].mean()
        
        for university, programs in self.university_requirements.items():
            probabilities[university] = {}
            
            for program, requirements in programs.items():
                min_score = requirements['min_score']
                core_subjects = requirements['core_subjects']
                cutoff_trend = requirements.get('cutoff_trend', [min_score])
                competition_level = requirements.get('competition_level', 0.8)
                
                # Check if student has required subjects
                has_core_subjects = all(
                    self._has_subject_performance(student_data, subject) 
                    for subject in core_subjects
                )
                
                if has_core_subjects and overall_score >= min_score:
                    # Calculate probability based on score margin and trends
                    score_margin = overall_score - min_score
                    trend_factor = self._calculate_trend_factor(cutoff_trend)
                    competition_factor = 1 - competition_level
                    
                    probability = min(0.95, 0.7 + (score_margin / 20) + trend_factor + competition_factor)
                elif has_core_subjects:
                    # Has subjects but below minimum score
                    probability = max(0.1, 0.3 - ((min_score - overall_score) / 10))
                else:
                    # Missing core subjects
                    probability = 0.05
                
                probabilities[university][program] = round(probability, 3)
        
        return probabilities
    
    def _calculate_trend_factor(self, cutoff_trend: List[float]) -> float:
        """Calculate trend factor based on cutoff history."""
        if len(cutoff_trend) < 2:
            return 0.0
        
        # Calculate trend direction
        recent_trend = cutoff_trend[-1] - cutoff_trend[0]
        trend_factor = min(0.1, max(-0.1, recent_trend / 10))
        
        return trend_factor
    
    def _analyze_detailed_skill_gaps(self, student_data: pd.DataFrame, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze detailed skill gaps for recommended careers."""
        skill_gaps = []
        
        for recommendation in recommendations[:3]:  # Top 3 recommendations
            career = recommendation['career']
            required_skills = recommendation.get('skill_requirements', [])
            
            # Analyze skill gaps based on subject performance
            gaps = []
            for skill in required_skills:
                gap_analysis = self._analyze_skill_gap(student_data, skill)
                if gap_analysis['gap_level'] != 'none':
                    gaps.append(gap_analysis)
            
            if gaps:
                skill_gaps.append({
                    'career': career,
                    'match_score': recommendation['match_score'],
                    'skill_gaps': gaps,
                    'development_priority': 'high' if len(gaps) > 2 else 'medium'
                })
        
        return skill_gaps
    
    def _analyze_skill_gap(self, student_data: pd.DataFrame, skill: str) -> Dict[str, Any]:
        """Analyze gap for specific skill."""
        skill_subject_mapping = {
            'analytical_thinking': ['Mathematics', 'Physics'],
            'problem_solving': ['Mathematics', 'Computer Science'],
            'technical_skills': ['Computer Science', 'Physics'],
            'scientific_thinking': ['Biology', 'Chemistry', 'Physics'],
            'attention_to_detail': ['Chemistry', 'Biology'],
            'empathy': ['Literature', 'Psychology'],
            'logical_thinking': ['Mathematics', 'Computer Science'],
            'programming': ['Computer Science', 'Mathematics'],
            'leadership': ['English Language', 'Literature'],
            'communication': ['English Language', 'Literature'],
            'critical_thinking': ['Literature', 'History'],
            'research_skills': ['Biology', 'Chemistry'],
            'patience': ['Literature', 'History'],
            'practical_skills': ['Agricultural Science', 'Chemistry'],
            'management': ['Economics', 'Accounting']
        }
        
        relevant_subjects = skill_subject_mapping.get(skill, [])
        if not relevant_subjects:
            return {'skill': skill, 'gap_level': 'none', 'reason': 'No relevant subjects found'}
        
        # Check performance in relevant subjects
        subject_performance = []
        for subject in relevant_subjects:
            if self._has_subject_performance(student_data, subject):
                subject_data = student_data[student_data['subject'] == subject]
                avg_score = subject_data['total_score'].mean()
                class_avg = subject_data['class_average'].mean()
                performance = (avg_score - class_avg) / class_avg
                subject_performance.append(performance)
        
        if not subject_performance:
            return {'skill': skill, 'gap_level': 'high', 'reason': 'No relevant subject data'}
        
        avg_performance = np.mean(subject_performance)
        
        if avg_performance > 0.1:
            gap_level = 'none'
            reason = 'Strong performance in relevant subjects'
        elif avg_performance > -0.1:
            gap_level = 'low'
            reason = 'Slight improvement needed'
        elif avg_performance > -0.3:
            gap_level = 'medium'
            reason = 'Moderate improvement needed'
        else:
            gap_level = 'high'
            reason = 'Significant improvement needed'
        
        return {
            'skill': skill,
            'gap_level': gap_level,
            'reason': reason,
            'performance_score': round(avg_performance, 3),
            'relevant_subjects': relevant_subjects
        }
    
    def _calculate_enhanced_confidence(self, student_data: pd.DataFrame, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate enhanced confidence in career recommendations."""
        # Base confidence on data quality
        data_quality = self._calculate_data_quality(student_data)
        
        # Adjust based on recommendation quality
        if not recommendations:
            return 0.0
        
        avg_match_score = np.mean([r['match_score'] for r in recommendations])
        recommendation_quality = min(avg_match_score, 0.95)
        
        # Adjust based on data consistency
        consistency = self._calculate_score_consistency(student_data)
        
        confidence = (
            data_quality * 0.4 +
            recommendation_quality * 0.4 +
            consistency * 0.2
        )
        
        return round(min(confidence, 0.95), 2)
    
    def _has_subject_performance(self, student_data: pd.DataFrame, subject: str) -> bool:
        """Check if student has performance data for subject."""
        return subject in student_data['subject'].values
    
    def _check_requirements(self, student_data: pd.DataFrame, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Check which requirements are met."""
        met_requirements = []
        unmet_requirements = []
        
        # Check core subjects
        for subject in requirements.get('core_subjects', []):
            if self._has_subject_performance(student_data, subject):
                met_requirements.append(f"Core subject: {subject}")
            else:
                unmet_requirements.append(f"Core subject: {subject}")
        
        # Check minimum score
        overall_score = student_data['total_score'].mean()
        min_score = requirements.get('min_score', 0)
        
        if overall_score >= min_score:
            met_requirements.append(f"Minimum score: {min_score}")
        else:
            unmet_requirements.append(f"Minimum score: {min_score} (current: {overall_score:.1f})")
        
        return {
            'met': met_requirements,
            'unmet': unmet_requirements,
            'met_count': len(met_requirements),
            'unmet_count': len(unmet_requirements)
        }
    
    def _get_recommendation_reason(self, student_data: pd.DataFrame, career: str, requirements: Dict[str, Any]) -> str:
        """Generate reason for career recommendation."""
        strong_subjects = []
        for subject in requirements.get('core_subjects', []):
            if self._has_subject_performance(student_data, subject):
                subject_data = student_data[student_data['subject'] == subject]
                if subject_data['total_score'].mean() > subject_data['class_average'].mean():
                    strong_subjects.append(subject)
        
        if strong_subjects:
            return f"Strong performance in core subjects: {', '.join(strong_subjects)}"
        else:
            return f"Good overall academic performance and potential for {career}"
    
    def _fallback_recommendation(self, student_id: str, reason: str) -> Dict[str, Any]:
        """Fallback recommendation when analysis fails."""
        return {
            'student_id': student_id,
            'error': reason,
            'strengths_analysis': {},
            'weaknesses_analysis': {},
            'career_recommendations': [],
            'university_probabilities': {},
            'skill_gaps': [],
            'improvement_plan': {},
            'recommendation_confidence': 0.0,
            'analysis_timestamp': datetime.now().isoformat()
        }
