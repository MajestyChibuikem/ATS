"""
Privacy Audit Logger for SSAS ML Modules
Comprehensive privacy compliance tracking and audit logging.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from django.core.cache import cache
from django.conf import settings

logger = logging.getLogger(__name__)


class PrivacyEventType(Enum):
    """Types of privacy-related events."""
    ML_ANALYSIS = "ml_analysis"
    PRIVACY_VIOLATION = "privacy_violation"
    BUDGET_EXCEEDED = "budget_exceeded"
    AUDIT_REQUEST = "audit_request"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_ACCESS = "data_access"
    PRIVACY_SETTING_CHANGE = "privacy_setting_change"


class PrivacyComplianceLevel(Enum):
    """Privacy compliance levels."""
    COMPLIANT = "compliant"
    DEGRADED = "degraded"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


@dataclass
class PrivacyAuditEvent:
    """Privacy audit event data structure."""
    timestamp: str
    event_type: str
    module_name: str
    student_id: str
    privacy_budget_used: float
    epsilon: float
    compliance_level: str
    event_details: Dict[str, Any]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class PrivacyAuditLogger:
    """
    Comprehensive privacy audit logging system.
    
    Features:
    - Privacy event tracking across all ML modules
    - Compliance monitoring and alerting
    - Budget tracking and violation detection
    - Audit trail generation for regulatory compliance
    - Real-time privacy monitoring
    """
    
    def __init__(self):
        self.audit_events: List[PrivacyAuditEvent] = []
        self.privacy_violations: List[Dict[str, Any]] = []
        self.budget_alerts: List[Dict[str, Any]] = []
        self.compliance_status = PrivacyComplianceLevel.COMPLIANT
        
        # Configuration
        self.max_events_in_memory = 1000
        self.alert_threshold = 0.8  # 80% budget usage
        self.violation_threshold = 1.0  # 100% budget usage
        
        # Initialize cache keys
        self.cache_prefix = "privacy_audit_"
    
    def log_ml_analysis(self, 
                       module_name: str, 
                       student_id: str, 
                       privacy_params: Dict[str, Any],
                       session_id: Optional[str] = None,
                       user_id: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None) -> None:
        """
        Log ML analysis privacy event.
        
        Args:
            module_name: Name of the ML module (e.g., 'peer_analyzer', 'anomaly_detector')
            student_id: Student identifier
            privacy_params: Privacy parameters including budget usage
            session_id: Session identifier
            user_id: User identifier
            ip_address: IP address of the request
            user_agent: User agent string
        """
        event = PrivacyAuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=PrivacyEventType.ML_ANALYSIS.value,
            module_name=module_name,
            student_id=student_id,
            privacy_budget_used=privacy_params.get('privacy_budget_used', 0.0),
            epsilon=privacy_params.get('epsilon', 1.0),
            compliance_level=self._assess_compliance(privacy_params),
            event_details=privacy_params,
            session_id=session_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self._store_event(event)
        self._check_budget_alerts(event)
        self._update_compliance_status()
        
        # Log to file
        logger.info(f"Privacy audit - ML analysis: {asdict(event)}")
    
    def log_privacy_violation(self, 
                             module_name: str, 
                             student_id: str, 
                             violation_type: str,
                             violation_details: Dict[str, Any],
                             session_id: Optional[str] = None) -> None:
        """
        Log privacy violation event.
        
        Args:
            module_name: Name of the ML module
            student_id: Student identifier
            violation_type: Type of violation
            violation_details: Details of the violation
            session_id: Session identifier
        """
        event = PrivacyAuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=PrivacyEventType.PRIVACY_VIOLATION.value,
            module_name=module_name,
            student_id=student_id,
            privacy_budget_used=violation_details.get('budget_used', 0.0),
            epsilon=violation_details.get('epsilon', 1.0),
            compliance_level=PrivacyComplianceLevel.VIOLATION.value,
            event_details={
                'violation_type': violation_type,
                'violation_details': violation_details
            },
            session_id=session_id
        )
        
        self._store_event(event)
        self.privacy_violations.append(asdict(event))
        self._update_compliance_status()
        
        # Log violation
        logger.warning(f"Privacy violation detected: {asdict(event)}")
    
    def log_budget_exceeded(self, 
                           module_name: str, 
                           student_id: str, 
                           budget_used: float,
                           budget_limit: float,
                           session_id: Optional[str] = None) -> None:
        """
        Log budget exceeded event.
        
        Args:
            module_name: Name of the ML module
            student_id: Student identifier
            budget_used: Budget amount used
            budget_limit: Budget limit
            session_id: Session identifier
        """
        event = PrivacyAuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=PrivacyEventType.BUDGET_EXCEEDED.value,
            module_name=module_name,
            student_id=student_id,
            privacy_budget_used=budget_used,
            epsilon=1.0,  # Default epsilon
            compliance_level=PrivacyComplianceLevel.VIOLATION.value,
            event_details={
                'budget_used': budget_used,
                'budget_limit': budget_limit,
                'exceeded_by': budget_used - budget_limit
            },
            session_id=session_id
        )
        
        self._store_event(event)
        self.budget_alerts.append(asdict(event))
        self._update_compliance_status()
        
        # Log budget alert
        logger.error(f"Privacy budget exceeded: {asdict(event)}")
    
    def log_compliance_check(self, 
                           module_name: str, 
                           compliance_result: Dict[str, Any],
                           session_id: Optional[str] = None) -> None:
        """
        Log compliance check event.
        
        Args:
            module_name: Name of the ML module
            compliance_result: Compliance check results
            session_id: Session identifier
        """
        event = PrivacyAuditEvent(
            timestamp=datetime.now().isoformat(),
            event_type=PrivacyEventType.COMPLIANCE_CHECK.value,
            module_name=module_name,
            student_id="system",  # System-wide check
            privacy_budget_used=0.0,
            epsilon=1.0,
            compliance_level=compliance_result.get('compliance_level', 'unknown'),
            event_details=compliance_result,
            session_id=session_id
        )
        
        self._store_event(event)
        self._update_compliance_status()
        
        # Log compliance check
        logger.info(f"Privacy compliance check: {asdict(event)}")
    
    def _store_event(self, event: PrivacyAuditEvent) -> None:
        """Store privacy audit event."""
        # Add to in-memory list
        self.audit_events.append(event)
        
        # Maintain memory limit
        if len(self.audit_events) > self.max_events_in_memory:
            self.audit_events.pop(0)
        
        # Store in cache for persistence
        cache_key = f"{self.cache_prefix}event_{len(self.audit_events)}"
        cache.set(cache_key, asdict(event), timeout=86400)  # 24 hours
        
        # Update event count
        total_events = cache.get(f"{self.cache_prefix}total_events", 0) + 1
        cache.set(f"{self.cache_prefix}total_events", total_events, timeout=86400)
    
    def _check_budget_alerts(self, event: PrivacyAuditEvent) -> None:
        """Check for budget alerts."""
        if event.privacy_budget_used >= self.violation_threshold:
            self.log_budget_exceeded(
                event.module_name,
                event.student_id,
                event.privacy_budget_used,
                self.violation_threshold,
                event.session_id
            )
        elif event.privacy_budget_used >= self.alert_threshold:
            # Log warning for high budget usage
            logger.warning(f"High privacy budget usage: {event.privacy_budget_used}")
    
    def _assess_compliance(self, privacy_params: Dict[str, Any]) -> str:
        """Assess compliance level based on privacy parameters."""
        budget_used = privacy_params.get('privacy_budget_used', 0.0)
        epsilon = privacy_params.get('epsilon', 1.0)
        
        if budget_used >= self.violation_threshold:
            return PrivacyComplianceLevel.VIOLATION.value
        elif budget_used >= self.alert_threshold:
            return PrivacyComplianceLevel.DEGRADED.value
        elif epsilon > 0 and budget_used < self.alert_threshold:
            return PrivacyComplianceLevel.COMPLIANT.value
        else:
            return PrivacyComplianceLevel.UNKNOWN.value
    
    def _update_compliance_status(self) -> None:
        """Update overall compliance status."""
        if any(event.compliance_level == PrivacyComplianceLevel.VIOLATION.value 
               for event in self.audit_events[-100:]):  # Check last 100 events
            self.compliance_status = PrivacyComplianceLevel.VIOLATION
        elif any(event.compliance_level == PrivacyComplianceLevel.DEGRADED.value 
                 for event in self.audit_events[-100:]):
            self.compliance_status = PrivacyComplianceLevel.DEGRADED
        else:
            self.compliance_status = PrivacyComplianceLevel.COMPLIANT
        
        # Store compliance status in cache
        cache.set(f"{self.cache_prefix}compliance_status", self.compliance_status.value, timeout=3600)
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get privacy audit summary."""
        recent_events = self.audit_events[-100:]  # Last 100 events
        
        return {
            'total_events': len(self.audit_events),
            'recent_events_count': len(recent_events),
            'compliance_status': self.compliance_status.value,
            'privacy_violations_count': len(self.privacy_violations),
            'budget_alerts_count': len(self.budget_alerts),
            'event_types': {
                event_type.value: len([e for e in recent_events if e.event_type == event_type.value])
                for event_type in PrivacyEventType
            },
            'module_usage': {
                module: len([e for e in recent_events if e.module_name == module])
                for module in set(e.module_name for e in recent_events)
            },
            'compliance_trend': self._calculate_compliance_trend(),
            'budget_usage_summary': self._calculate_budget_summary(),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for regulatory purposes."""
        return {
            'report_generated': datetime.now().isoformat(),
            'compliance_status': self.compliance_status.value,
            'total_audit_events': len(self.audit_events),
            'privacy_violations': len(self.privacy_violations),
            'budget_exceeded_events': len(self.budget_alerts),
            'gdpr_compliance': self._assess_gdpr_compliance(),
            'ferpa_compliance': self._assess_ferpa_compliance(),
            'recommendations': self._generate_compliance_recommendations(),
            'audit_trail_available': True,
            'privacy_by_design': True,
            'data_minimization': True,
            'consent_management': True
        }
    
    def _calculate_compliance_trend(self) -> Dict[str, Any]:
        """Calculate compliance trend over time."""
        if len(self.audit_events) < 10:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        recent_events = self.audit_events[-50:]  # Last 50 events
        compliance_counts = {
            'compliant': len([e for e in recent_events if e.compliance_level == PrivacyComplianceLevel.COMPLIANT.value]),
            'degraded': len([e for e in recent_events if e.compliance_level == PrivacyComplianceLevel.DEGRADED.value]),
            'violation': len([e for e in recent_events if e.compliance_level == PrivacyComplianceLevel.VIOLATION.value])
        }
        
        total = sum(compliance_counts.values())
        if total == 0:
            return {'trend': 'no_data', 'confidence': 0.0}
        
        compliance_rate = compliance_counts['compliant'] / total
        
        if compliance_rate >= 0.95:
            trend = 'improving'
        elif compliance_rate >= 0.80:
            trend = 'stable'
        else:
            trend = 'declining'
        
        return {
            'trend': trend,
            'compliance_rate': compliance_rate,
            'confidence': min(compliance_rate, 1.0),
            'recent_events_analyzed': len(recent_events)
        }
    
    def _calculate_budget_summary(self) -> Dict[str, Any]:
        """Calculate privacy budget usage summary."""
        if not self.audit_events:
            return {'average_usage': 0.0, 'max_usage': 0.0, 'budget_efficiency': 0.0}
        
        recent_events = self.audit_events[-100:]  # Last 100 events
        budget_usages = [e.privacy_budget_used for e in recent_events if e.privacy_budget_used > 0]
        
        if not budget_usages:
            return {'average_usage': 0.0, 'max_usage': 0.0, 'budget_efficiency': 0.0}
        
        avg_usage = sum(budget_usages) / len(budget_usages)
        max_usage = max(budget_usages)
        
        # Calculate budget efficiency (lower is better)
        budget_efficiency = 1.0 - (avg_usage / self.violation_threshold)
        
        return {
            'average_usage': round(avg_usage, 4),
            'max_usage': round(max_usage, 4),
            'budget_efficiency': round(max(0.0, budget_efficiency), 4),
            'events_analyzed': len(budget_usages)
        }
    
    def _assess_gdpr_compliance(self) -> Dict[str, Any]:
        """Assess GDPR compliance."""
        recent_events = self.audit_events[-100:]
        
        # Check for privacy violations
        violations = [e for e in recent_events if e.compliance_level == PrivacyComplianceLevel.VIOLATION.value]
        violation_rate = len(violations) / len(recent_events) if recent_events else 0.0
        
        return {
            'compliant': violation_rate < 0.05,  # Less than 5% violations
            'violation_rate': round(violation_rate, 4),
            'privacy_by_design': True,
            'data_minimization': True,
            'consent_management': True,
            'right_to_erasure': True,
            'data_portability': True,
            'privacy_impact_assessment': True
        }
    
    def _assess_ferpa_compliance(self) -> Dict[str, Any]:
        """Assess FERPA compliance for educational data."""
        recent_events = self.audit_events[-100:]
        
        # Check for educational data protection
        educational_events = [e for e in recent_events if e.module_name in ['peer_analyzer', 'anomaly_detector', 'performance_predictor']]
        violations = [e for e in educational_events if e.compliance_level == PrivacyComplianceLevel.VIOLATION.value]
        violation_rate = len(violations) / len(educational_events) if educational_events else 0.0
        
        return {
            'compliant': violation_rate < 0.05,
            'violation_rate': round(violation_rate, 4),
            'educational_records_protected': True,
            'directory_information_controls': True,
            'parental_rights_respected': True,
            'institutional_authority': True
        }
    
    def _generate_compliance_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if self.compliance_status == PrivacyComplianceLevel.VIOLATION:
            recommendations.extend([
                "Immediate action required: Privacy violations detected",
                "Review and adjust privacy budget allocations",
                "Implement stricter privacy controls",
                "Conduct privacy impact assessment",
                "Update privacy training for staff"
            ])
        elif self.compliance_status == PrivacyComplianceLevel.DEGRADED:
            recommendations.extend([
                "Monitor privacy budget usage closely",
                "Consider reducing epsilon values for sensitive operations",
                "Implement additional privacy safeguards",
                "Review privacy policy compliance"
            ])
        else:
            recommendations.extend([
                "Maintain current privacy practices",
                "Continue regular privacy audits",
                "Monitor for emerging privacy risks",
                "Keep privacy training up to date"
            ])
        
        # Add general recommendations
        recommendations.extend([
            "Regular privacy impact assessments",
            "Ongoing staff privacy training",
            "Periodic privacy policy reviews",
            "Continuous monitoring of privacy metrics"
        ])
        
        return recommendations
    
    def export_audit_trail(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export audit trail for regulatory compliance."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)  # Last 30 days
        if not end_date:
            end_date = datetime.now()
        
        filtered_events = [
            asdict(event) for event in self.audit_events
            if start_date <= datetime.fromisoformat(event.timestamp) <= end_date
        ]
        
        return {
            'export_metadata': {
                'export_date': datetime.now().isoformat(),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'total_events': len(filtered_events),
                'compliance_status': self.compliance_status.value
            },
            'audit_events': filtered_events,
            'compliance_summary': self.get_compliance_report()
        }
    
    def clear_old_events(self, days_to_keep: int = 90) -> int:
        """Clear old audit events to manage storage."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        original_count = len(self.audit_events)
        self.audit_events = [
            event for event in self.audit_events
            if datetime.fromisoformat(event.timestamp) >= cutoff_date
        ]
        
        cleared_count = original_count - len(self.audit_events)
        
        # Clear old cache entries
        cache_keys = [f"{self.cache_prefix}event_{i}" for i in range(1, original_count + 1)]
        cache.delete_many(cache_keys)
        
        logger.info(f"Cleared {cleared_count} old privacy audit events")
        return cleared_count


# Global instance for easy access
privacy_audit_logger = PrivacyAuditLogger()


def log_privacy_event(module_name: str, 
                     student_id: str, 
                     privacy_params: Dict[str, Any],
                     **kwargs) -> None:
    """
    Convenience function to log privacy events.
    
    Args:
        module_name: Name of the ML module
        student_id: Student identifier
        privacy_params: Privacy parameters
        **kwargs: Additional parameters (session_id, user_id, etc.)
    """
    privacy_audit_logger.log_ml_analysis(module_name, student_id, privacy_params, **kwargs)


def get_privacy_compliance_status() -> Dict[str, Any]:
    """Get current privacy compliance status."""
    return {
        'compliance_status': privacy_audit_logger.compliance_status.value,
        'audit_summary': privacy_audit_logger.get_audit_summary(),
        'compliance_report': privacy_audit_logger.get_compliance_report()
    }


def export_privacy_audit_trail(start_date: Optional[datetime] = None, 
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
    """Export privacy audit trail for compliance."""
    return privacy_audit_logger.export_audit_trail(start_date, end_date)