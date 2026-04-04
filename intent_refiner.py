"""
Intent refinement using conversation context and multi-turn awareness.
Improves intent detection accuracy by considering conversation history and user context.
"""

from typing import Dict, List, Tuple, Optional
from collections import Counter


class IntentRefiner:
    """Refines intent predictions using context and conversation history."""
    
    # Related intents for context-aware refinement
    INTENT_RELATIONSHIPS = {
        "fees_admission": ["fees", "admission", "cost", "payment"],
        "course_details": ["course_info", "curriculum", "syllabus", "modules"],
        "faculty": ["faculty", "professor", "instructor", "teach"],
        "placement": ["placement", "job", "recruitment", "company"],
        "exam": ["exam", "test", "assessment", "result"],
        "campus_life": ["campus", "hostel", "sports", "club", "facility"]
    }
    
    # Intent context preferences (if user asked X, next query likely Y)
    INTENT_SEQUENCES = {
        "admission": ["fees_admission", "course_details", "campus_life"],
        "fees_admission": ["payment", "admission", "course_details"],
        "course_details": ["faculty", "exam", "placement"],
        "faculty": ["course_details", "office_hours", "research"],
        "exam": ["result", "grade", "reevaluation"],
        "placement": ["company_details", "salary", "internship"],
    }
    
    def __init__(self, history_size: int = 5):
        """
        Initialize intent refiner.
        
        Args:
            history_size (int): Number of conversation turns to consider
        """
        self.history_size = history_size
        self.refinement_history = []
    
    def refine_intent(self,
                     predicted_intent: str,
                     confidence: float,
                     user_input: str,
                     conversation_history: List[Dict] = None,
                     emotion: str = "neutral") -> Dict:
        """
        Refine intent prediction using context.
        
        Args:
            predicted_intent (str): Initial intent prediction
            confidence (float): Prediction confidence
            user_input (str): Current user input
            conversation_history (List): Previous conversation turns
            emotion (str): Detected emotion
            
        Returns:
            Dict: Refined intent with updated confidence
        """
        conversation_history = conversation_history or []
        
        # Check for explicit intent indicators
        explicit_intent = self._detect_explicit_intent(user_input)
        if explicit_intent and explicit_intent != predicted_intent:
            # Override if confidence is high
            if self._is_explicit_indicator_strong(user_input, explicit_intent):
                return self._create_refinement(
                    intent=explicit_intent,
                    confidence=0.95,
                    reason="explicit_indicator",
                    original_intent=predicted_intent
                )
        
        # Use conversation context to refine
        context_refined_intent = self._refine_with_context(
            predicted_intent,
            conversation_history
        )
        
        # Check if context suggests a different related intent
        if context_refined_intent != predicted_intent:
            refined_confidence = confidence * 0.9  # Slightly lower confidence for context-refined
            reason = "context_suggestion"
        else:
            refined_confidence = confidence
            reason = "context_consistent"
        
        # Adjust confidence based on emotion for specific intents
        adjusted_confidence = self._adjust_confidence_for_emotion(
            refined_confidence,
            context_refined_intent,
            emotion
        )
        
        result = self._create_refinement(
            intent=context_refined_intent,
            confidence=adjusted_confidence,
            reason=reason,
            original_intent=predicted_intent,
            explicit_indicator=explicit_intent,
            context_window=len(conversation_history)
        )
        
        self.refinement_history.append(result)
        return result
    
    def _detect_explicit_intent(self, user_input: str) -> Optional[str]:
        """
        Detect if user explicitly states their intent.
        
        Examples:
        - "I want to know about fees" → fees
        - "Tell me about placements" → placement
        - "What are the exam dates?" → exam
        """
        text_lower = user_input.lower()
        
        # Explicit intent patterns
        explicit_patterns = {
            # Admission & Fees
            "admission": ["admission", "apply", "entrance", "entrance exam"],
            "fees": ["fees", "cost", "price", "payment", "tuition"],
            
            # Academics
            "course_details": ["course", "curriculum", "syllabus", "subject", "module"],
            "faculty": ["faculty", "professor", "instructor", "teacher"],
            
            # Placement & Career
            "placement": ["placement", "job", "recruitment", "internship", "company"],
            "internship": ["internship", "intern position"],
            
            # Exams & Results
            "exam": ["exam", "test", "assessment", "midterm", "final"],
            "result": ["result", "score", "grade", "mark", "announcement"],
            
            # Campus
            "campus_life": ["campus", "hostel", "dorm", "sports", "club", "facility"],
        }
        
        for intent, patterns in explicit_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent
        
        return None
    
    def _is_explicit_indicator_strong(self, user_input: str, intent: str) -> bool:
        """Check if explicit indicator is strong enough to override."""
        strong_phrases = [
            "i want",
            "i need",
            "tell me about",
            "what about",
            "how about",
            "can you tell",
            "can you explain"
        ]
        
        text_lower = user_input.lower()
        for phrase in strong_phrases:
            if phrase in text_lower and intent in text_lower:
                return True
        
        return False
    
    def _refine_with_context(self,
                            predicted_intent: str,
                            conversation_history: List[Dict]) -> str:
        """
        Refine intent using conversation history.
        
        Strategy:
        1. If confidence is low, check if previous intents provide clues
        2. Look for intent sequences (e.g., admission → fees)
        3. Check for repeated concepts in history
        """
        if not conversation_history or len(conversation_history) == 0:
            return predicted_intent
        
        # Get last few intents from history
        recent_intents = []
        for turn in conversation_history[-self.history_size:]:
            if "intent" in turn:
                recent_intents.append(turn["intent"])
        
        # Check for likely intent sequence
        if recent_intents:
            last_intent = recent_intents[-1]
            
            # If current intent is ambiguous but related to last intent
            if last_intent in self.INTENT_SEQUENCES:
                likely_next_intents = self.INTENT_SEQUENCES[last_intent]
                
                if predicted_intent in likely_next_intents:
                    # Prediction is consistent with conversation flow
                    return predicted_intent
                
                # Check if any recent intent appears in current likelihood
                for recent_intent in reversed(recent_intents):
                    if recent_intent in likely_next_intents:
                        # User might be following up on previous topic
                        return recent_intent
        
        return predicted_intent
    
    def _adjust_confidence_for_emotion(self,
                                       confidence: float,
                                       intent: str,
                                       emotion: str) -> float:
        """Adjust confidence based on emotional context."""
        # If user is stressed asking about results, boost confidence
        if emotion == "stressed" and intent in ["exam", "result", "placement"]:
            return min(confidence + 0.1, 1.0)
        
        # If user is confused, show lower confidence
        if emotion == "confused":
            return confidence * 0.85
        
        return confidence
    
    def get_intent_confidence_range(self,
                                   predicted_intent: str,
                                   conversation_history: List[Dict]) -> Tuple[float, float]:
        """
        Get confidence range for an intent based on conversation history.
        
        Returns:
            Tuple: (min_confidence, max_confidence)
        """
        base_min = 0.4
        base_max = 1.0
        
        # If intent appeared in recent history, boost confidence
        if conversation_history:
            recent_intents = [turn.get("intent") for turn in conversation_history[-3:]]
            if predicted_intent in recent_intents:
                base_min = 0.7
                base_max = 1.0
        
        return (base_min, base_max)
    
    def get_related_intents(self, primary_intent: str) -> List[str]:
        """Get intents related to the primary intent."""
        for cluster, intents in self.INTENT_RELATIONSHIPS.items():
            if primary_intent in intents:
                # Return other intents in same cluster
                return [i for i in intents if i != primary_intent]
        
        return []
    
    def suggest_followup_intents(self, current_intent: str) -> List[str]:
        """Suggest likely follow-up intents based on current question."""
        return self.INTENT_SEQUENCES.get(current_intent, [])
    
    def _create_refinement(self, **kwargs) -> Dict:
        """Create refinement result object."""
        return {
            "intent": kwargs.get("intent"),
            "confidence": kwargs.get("confidence", 0.5),
            "reason": kwargs.get("reason", "unknown"),
            "original_intent": kwargs.get("original_intent"),
            "explicit_indicator": kwargs.get("explicit_indicator"),
            "context_window": kwargs.get("context_window", 0),
            "refined": kwargs.get("intent") != kwargs.get("original_intent")
        }
    
    def get_refinement_stats(self) -> Dict:
        """Get refinement statistics."""
        if not self.refinement_history:
            return {"total_refinements": 0}
        
        refined_count = sum(1 for r in self.refinement_history if r.get("refined", False))
        reasons = Counter([r.get("reason") for r in self.refinement_history])
        
        return {
            "total_refinements": len(self.refinement_history),
            "refined_percentage": (refined_count / len(self.refinement_history)) * 100,
            "reason_breakdown": dict(reasons),
            "recent_refinements": self.refinement_history[-5:]
        }
