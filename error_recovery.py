"""
Error recovery and graceful degradation for the chatbot.
Handles API failures, timeouts, and provides fallback responses.
Ensures fast response times even under failure conditions.
"""

import time
import logging
from typing import Dict, Optional, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"      # Requires user intervention
    HIGH = "high"              # Degraded functionality
    MEDIUM = "medium"          # Minor issue
    LOW = "low"                # Can be retried silently


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"            # Retry with exponential backoff
    FALLBACK = "fallback"      # Use fallback response
    CLARIFY = "clarify"        # Ask user for clarification
    ESCALATE = "escalate"      # Escalate to human
    TIMEOUT = "timeout"        # Operation timed out


class ErrorRecovery:
    """Handles errors gracefully with recovery strategies."""
    
    def __init__(self, max_retries: int = 3, base_timeout: float = 5.0):
        """
        Initialize error recovery handler.
        
        Args:
            max_retries (int): Maximum retry attempts
            base_timeout (float): Base timeout in seconds
        """
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.logger = logging.getLogger(__name__)
        self.error_history = []
    
    # Fallback responses for different scenarios
    FALLBACK_RESPONSES = {
        "api_failure": {
            "message": "I'm having trouble accessing that right now, but I can try a different approach. What's your question?",
            "severity": ErrorSeverity.HIGH
        },
        "timeout": {
            "message": "That's taking longer than expected. Can you rephrase your question more concisely?",
            "severity": ErrorSeverity.MEDIUM
        },
        "out_of_scope": {
            "message": "That's outside my college domain expertise. Can I help with admissions, courses, or other college info instead?",
            "severity": ErrorSeverity.LOW
        },
        "no_intent": {
            "message": "I'm not quite sure what you're asking. Could you give me more details?",
            "severity": ErrorSeverity.MEDIUM
        },
        "low_confidence": {
            "message": "I'm uncertain about that. Could you rephrase or give me more context?",
            "severity": ErrorSeverity.LOW
        },
        "generic": {
            "message": "Let me reconsider that. What specifically would you like to know?",
            "severity": ErrorSeverity.MEDIUM
        }
    }
    
    def handle_api_error(self, 
                        error: Exception, 
                        operation: str,
                        context: Dict = None) -> Dict:
        """
        Handle API errors with retry strategy.
        
        Args:
            error (Exception): The error that occurred
            operation (str): Name of failed operation
            context (Dict): Operation context
            
        Returns:
            Dict: Recovery strategy and fallback response
        """
        error_name = type(error).__name__
        self.logger.error(f"API Error in {operation}: {error_name}: {str(error)}")
        
        # Log error
        self.error_history.append({
            "timestamp": time.time(),
            "operation": operation,
            "error_type": error_name,
            "message": str(error),
            "context": context or {}
        })
        
        # Determine strategy based on error type
        if "timeout" in str(error).lower():
            return self._create_recovery({
                "strategy": ErrorRecoveryStrategy.TIMEOUT,
                "severity": ErrorSeverity.MEDIUM,
                "fallback_key": "timeout"
            })
        
        elif "connection" in str(error).lower():
            return self._create_recovery({
                "strategy": ErrorRecoveryStrategy.RETRY,
                "severity": ErrorSeverity.HIGH,
                "fallback_key": "api_failure"
            })
        
        else:
            return self._create_recovery({
                "strategy": ErrorRecoveryStrategy.FALLBACK,
                "severity": ErrorSeverity.HIGH,
                "fallback_key": "api_failure"
            })
    
    def handle_confidence_error(self, 
                               confidence: float,
                               intent: str) -> Dict:
        """
        Handle low confidence predictions.
        
        Args:
            confidence (float): Prediction confidence (0-1)
            intent (str): Predicted intent
            
        Returns:
            Dict: Recovery strategy
        """
        if confidence < 0.3:
            return self._create_recovery({
                "strategy": ErrorRecoveryStrategy.CLARIFY,
                "severity": ErrorSeverity.MEDIUM,
                "fallback_key": "no_intent",
                "confidence": confidence,
                "intent": intent
            })
        elif confidence < 0.6:
            return self._create_recovery({
                "strategy": ErrorRecoveryStrategy.CLARIFY,
                "severity": ErrorSeverity.LOW,
                "fallback_key": "low_confidence",
                "confidence": confidence,
                "intent": intent
            })
        
        return {"strategy": ErrorRecoveryStrategy.RETRY, "should_proceed": True}
    
    def retry_with_backoff(self, 
                          func, 
                          *args, 
                          **kwargs) -> Optional[any]:
        """
        Retry operation with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result or None if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                wait_time = (2 ** attempt) * 0.1  # Exponential backoff
                
                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"All {self.max_retries} retry attempts failed")
                    return None
    
    def get_fallback_response(self, fallback_key: str) -> Dict:
        """
        Get a fallback response for a specific error scenario.
        
        Args:
            fallback_key (str): Type of fallback needed
            
        Returns:
            Dict: Fallback response with message
        """
        fallback = self.FALLBACK_RESPONSES.get(
            fallback_key, 
            self.FALLBACK_RESPONSES["generic"]
        )
        return {
            "message": fallback["message"],
            "severity": fallback["severity"],
            "is_fallback": True
        }
    
    def _create_recovery(self, strategy_info: Dict) -> Dict:
        """Create recovery response object."""
        recovery = {
            "strategy": strategy_info.get("strategy"),
            "severity": strategy_info.get("severity"),
            "is_error": True,
            "should_proceed": False,
        }
        
        fallback_key = strategy_info.get("fallback_key")
        if fallback_key:
            recovery.update(self.get_fallback_response(fallback_key))
        
        return recovery
    
    def get_error_stats(self) -> Dict:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_types = {}
        for error in self.error_history:
            error_type = error.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": self.error_history[-5:]
        }
