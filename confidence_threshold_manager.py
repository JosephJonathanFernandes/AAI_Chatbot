"""
Dynamic Confidence Threshold Management for Intent Classification

Implements smart confidence thresholds that vary based on:
1. Intent type (some intents are harder to classify than others)
2. Query characteristics (length, clarity)
3. Model agreement (when models agree, lower threshold needed)

Goal: Reduce clarifications for high-confidence queries while triggering
smart clarifications for ambiguous queries.
"""

from typing import Dict, Tuple
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level categories."""
    VERY_HIGH = "very_high"      # > 0.85 - very confident
    HIGH = "high"                # 0.70 - 0.85 - confident
    MODERATE = "moderate"        # 0.50 - 0.70 - reasonable confidence
    LOW = "low"                  # 0.35 - 0.50 - needs clarification
    VERY_LOW = "very_low"        # < 0.35 - unclear


class ConfidenceThresholdManager:
    """
    Manages dynamic confidence thresholds based on intent and context.
    
    Key improvements:
    - Intent-specific thresholds (some intents are naturally harder)
    - Query characteristics influence threshold
    - Model agreement boosts confidence
    - Adaptive thresholds based on conversation history
    """
    
    # Base thresholds per intent (tuned from analysis)
    INTENT_THRESHOLDS = {
        # High-precision intents (clear keywords)
        "fees": 0.45,                    # Clear keywords: fees, cost, payment
        "exams": 0.45,                   # Clear keywords: exam, test, schedule
        "placements": 0.48,              # Clear keywords: placement, job, company
        "admission": 0.47,               # Clear keywords: admission, apply
        
        # Medium-precision intents
        "hostel": 0.52,                  # Some ambiguity with campus
        "faculty": 0.50,                 # Can be confused with general_info
        "library": 0.50,                 # Can be confused with campus facilities
        "timetable": 0.48,               # Clear but some variation in phrasing
        
        # Harder-to-classify intents (more contextual)
        "campus_life": 0.55,             # Broad category, many variations
        "general_info": 0.58,            # Very broad, can overlap with others
        "comparison": 0.60,              # Requires understanding of comparison context
        "eligibility": 0.52,             # Overlaps with admission
        
        # Social intents (easier to classify)
        "greetings": 0.35,               # Very easy to identify
        "gratitude": 0.40,               # Clear patterns
        "affirmation": 0.40,             # Clear patterns
        "negation": 0.40,                # Clear patterns
        
        # Catch-all
        "unknown": 0.30,                 # Default threshold
        "out_of_scope": 0.50,            # Should be reasonably confident
    }
    
    # Minimum thresholds (never go below this)
    MIN_THRESHOLD = 0.30
    # Maximum thresholds (never go above this)
    MAX_THRESHOLD = 0.80
    
    def __init__(self):
        """Initialize confidence threshold manager."""
        self.thresholds = self.INTENT_THRESHOLDS.copy()
        self.adjustment_history = []
    
    def get_threshold(self, intent: str, confidence: float = 0.0, 
                     models_agree: bool = False, query_length: int = 0) -> float:
        """
        Get dynamic threshold for an intent.
        
        Args:
            intent (str): Intent type
            confidence (float): Current confidence score (for learning)
            models_agree (bool): Whether different models agree on intent
            query_length (int): Length of user query in characters
        
        Returns:
            float: Adjusted threshold for this intent
        """
        # Get base threshold for intent
        base_threshold = self.thresholds.get(intent, 0.50)
        
        # Adjustment 1: Models agree - lower threshold (can be more lenient)
        if models_agree:
            adjusted_threshold = base_threshold * 0.90  # 10% lower
        else:
            adjusted_threshold = base_threshold
        
        # Adjustment 2: Query characteristics
        # Very short queries (1-2 words) might be less ambiguous
        if query_length < 10:
            adjusted_threshold = adjusted_threshold * 0.95  # 5% lower
        # Very long queries (50+ chars) might be more complex
        elif query_length > 50:
            adjusted_threshold = adjusted_threshold * 1.05  # 5% higher
        
        # Clamp to valid range
        adjusted_threshold = max(self.MIN_THRESHOLD, 
                                min(self.MAX_THRESHOLD, adjusted_threshold))
        
        return adjusted_threshold
    
    def should_clarify(self, intent: str, confidence: float, 
                      models_agree: bool = False, query_length: int = 0) -> bool:
        """
        Determine if clarification is needed for a prediction.
        
        Args:
            intent (str): Predicted intent
            confidence (float): Prediction confidence
            models_agree (bool): Whether models agree
            query_length (int): Query length
        
        Returns:
            bool: True if clarification should be requested
        """
        threshold = self.get_threshold(intent, confidence, models_agree, query_length)
        should_clarify = confidence < threshold
        
        # Log for learning/analysis
        self.adjustment_history.append({
            "intent": intent,
            "confidence": confidence,
            "threshold": threshold,
            "should_clarify": should_clarify,
            "models_agree": models_agree,
            "query_length": query_length
        })
        
        return should_clarify
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """
        Categorize confidence score into levels.
        
        Args:
            confidence (float): Confidence score 0-1
        
        Returns:
            ConfidenceLevel: Category of confidence
        """
        if confidence >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.70:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.50:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.35:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def adjust_threshold(self, intent: str, new_threshold: float) -> None:
        """
        Manually adjust threshold for an intent.
        Used for A/B testing and tuning.
        
        Args:
            intent (str): Intent to adjust
            new_threshold (float): New threshold value
        """
        new_threshold = max(self.MIN_THRESHOLD, 
                           min(self.MAX_THRESHOLD, new_threshold))
        self.thresholds[intent] = new_threshold
    
    def get_stats(self) -> Dict:
        """
        Get statistics from adjustment history.
        
        Returns:
            Dict: Statistics about threshold usage
        """
        if not self.adjustment_history:
            return {"total_decisions": 0}
        
        total = len(self.adjustment_history)
        clarified = sum(1 for x in self.adjustment_history if x["should_clarify"])
        
        # Group by intent
        by_intent = {}
        for record in self.adjustment_history:
            intent = record["intent"]
            if intent not in by_intent:
                by_intent[intent] = {"total": 0, "clarified": 0, "avg_confidence": 0}
            
            by_intent[intent]["total"] += 1
            if record["should_clarify"]:
                by_intent[intent]["clarified"] += 1
            by_intent[intent]["avg_confidence"] += record["confidence"]
        
        # Calculate averages
        for intent_stats in by_intent.values():
            if intent_stats["total"] > 0:
                intent_stats["avg_confidence"] /= intent_stats["total"]
                intent_stats["clarification_rate"] = (
                    intent_stats["clarified"] / intent_stats["total"]
                )
        
        return {
            "total_decisions": total,
            "total_clarified": clarified,
            "overall_clarification_rate": clarified / total if total > 0 else 0,
            "by_intent": by_intent
        }
    
    def reset_history(self) -> None:
        """Reset adjustment history (for new session/analysis)."""
        self.adjustment_history = []


# Test the confidence threshold manager
if __name__ == "__main__":
    print("CONFIDENCE THRESHOLD MANAGER TEST")
    print("=" * 70)
    
    manager = ConfidenceThresholdManager()
    
    # Test various scenarios
    test_cases = [
        ("fees", 0.55, True, 8),         # High agreement on fees, short query
        ("placements", 0.48, False, 45), # Disagreement, long query
        ("campus_life", 0.52, True, 15), # Agreement, medium query
        ("exams", 0.40, True, 12),       # Low confidence, agreement
        ("general_info", 0.50, False, 60),  # Disagreement, long query
    ]
    
    print("\nScenario Analysis:")
    print("-" * 70)
    
    for intent, conf, agree, length in test_cases:
        threshold = manager.get_threshold(intent, conf, agree, length)
        clarify = manager.should_clarify(intent, conf, agree, length)
        conf_level = manager.get_confidence_level(conf)
        
        print(f"\nIntent: {intent:15} | Confidence: {conf:.2f}")
        print(f"  Models Agree: {agree:5} | Query Length: {length:3}")
        print(f"  → Threshold: {threshold:.3f} | Clarify: {clarify} | Level: {conf_level.value}")
    
    print("\n" + "=" * 70)
    print("\nThreshold Settings by Intent:")
    print("-" * 70)
    
    for intent, threshold in sorted(manager.INTENT_THRESHOLDS.items()):
        print(f"  {intent:20} → {threshold:.3f}")
    
    print("\n" + "=" * 70)
    print("\nStatistics:")
    stats = manager.get_stats()
    print(f"  Total Decisions: {stats['total_decisions']}")
    print(f"  Total Clarified: {stats['total_clarified']}")
    print(f"  Clarification Rate: {stats['overall_clarification_rate']:.1%}")
