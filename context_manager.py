"""
Context management for multi-turn conversations.
Maintains conversation history, intents, entities, and state.
"""

from collections import deque
from datetime import datetime


class ConversationContext:
    """Manages conversation context and history for multi-turn conversations."""
    
    def __init__(self, max_history=5):
        """
        Initialize conversation context manager.
        
        Args:
            max_history (int): Maximum number of conversation turns to maintain
        """
        self.max_history = max_history
        self.conversation_history = deque(maxlen=max_history)
        self.last_intent = None
        self.last_confidence = 0.0
        self.last_emotion = "neutral"
        self.last_entities = {}
        self.session_start = datetime.now()
        self.total_turns = 0
    
    def add_turn(self, user_input, bot_response, intent, confidence, emotion, entities=None):
        """
        Add a conversation turn to the history.
        
        Args:
            user_input (str): User's message
            bot_response (str): Bot's response
            intent (str): Detected intent
            confidence (float): Intent confidence
            emotion (str): Detected emotion
            entities (dict): Extracted entities (optional)
        """
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "bot_response": bot_response,
            "intent": intent,
            "confidence": confidence,
            "emotion": emotion,
            "entities": entities or {}
        }
        
        self.conversation_history.append(turn)
        self.last_intent = intent
        self.last_confidence = confidence
        self.last_emotion = emotion
        self.last_entities = entities or {}
        self.total_turns += 1
    
    def get_history(self):
        """
        Get the conversation history.
        
        Returns:
            list: List of conversation turns
        """
        return list(self.conversation_history)
    
    def get_formatted_history(self, include_metadata=False):
        """
        Get conversation history formatted for LLM context.
        
        Args:
            include_metadata (bool): Include intent/emotion metadata
        
        Returns:
            str: Formatted conversation history
        """
        if not self.conversation_history:
            return "No previous conversation."
        
        lines = ["Recent conversation:"]
        for i, turn in enumerate(self.conversation_history, 1):
            lines.append(f"\nTurn {i}:")
            lines.append(f"User: {turn['user_input']}")
            lines.append(f"Assistant: {turn['bot_response']}")
            
            if include_metadata:
                lines.append(f"  [Intent: {turn['intent']}, Emotion: {turn['emotion']}]")
        
        return "\n".join(lines)
    
    def get_recent_intents(self, count=3):
        """
        Get the last N intents from conversation.
        
        Args:
            count (int): Number of recent intents to retrieve
        
        Returns:
            list: List of recent intent tags
        """
        intents = [turn['intent'] for turn in self.conversation_history]
        return intents[-count:] if intents else []
    
    def get_topic_continuity(self):
        """
        Analyze if the conversation is continuous on the same topic.
        
        Returns:
            bool: True if last 2+ turns involve the same intent
        """
        recent_intents = self.get_recent_intents(count=2)
        if len(recent_intents) >= 2:
            return recent_intents[0] == recent_intents[1]
        return False
    
    def get_context_summary(self):
        """
        Get a summary of the current conversation context.
        
        Returns:
            dict: Context summary
        """
        return {
            "total_turns": self.total_turns,
            "session_duration": (datetime.now() - self.session_start).total_seconds(),
            "last_intent": self.last_intent,
            "last_confidence": self.last_confidence,
            "last_emotion": self.last_emotion,
            "topic_continuity": self.get_topic_continuity(),
            "recent_intents": self.get_recent_intents(count=3),
            "last_entities": self.last_entities
        }
    
    def clear_history(self):
        """Clear all conversation history."""
        self.conversation_history.clear()
        self.last_intent = None
        self.last_confidence = 0.0
        self.last_emotion = "neutral"
        self.last_entities = {}
        self.session_start = datetime.now()
        self.total_turns = 0
    
    def should_ask_clarification(self, confidence_threshold=0.5):
        """
        Determine if bot should ask for clarification based on confidence.
        
        Args:
            confidence_threshold (float): Minimum confidence to not ask for clarification
        
        Returns:
            bool: True if clarification is needed
        """
        return self.last_confidence < confidence_threshold
    
    def get_clarification_prompt(self):
        """
        Get a clarification prompt based on the unclear intent.
        
        Returns:
            str: Clarification message
        """
        intents_help = {
            "fees": "Are you asking about tuition fees, payment plans, or scholarships?",
            "exams": "Would you like to know about exam dates, preparation, or results?",
            "timetable": "Are you looking for class schedules or lab timings?",
            "placements": "Are you interested in placement statistics or company information?",
            "library": "Are you asking about library timing, books, or facilities?",
            "general": "I didn't quite understand your question. Could you provide more details?"
        }
        
        prompt = intents_help.get(self.last_intent, intents_help["general"])
        return f"I'm not entirely sure. {prompt}"
    
    def get_prompt_context(self):
        """
        Get formatted context for LLM prompt injection.
        
        Returns:
            str: Context string for LLM
        """
        context_parts = []
        
        # Add recent conversation with full Q&A for better context
        if self.conversation_history:
            context_parts.append("RECENT CONVERSATION:")
            for turn in list(self.conversation_history)[-3:]:
                context_parts.append(f"  User: {turn['user_input']}")
                context_parts.append(f"  Assistant: {turn['bot_response'][:100]}...")
        else:
            context_parts.append("This is the start of the conversation.")
        
        # Add topic continuity info
        if self.get_topic_continuity():
            context_parts.append("\n[IMPORTANT: User is continuing on the SAME TOPIC as previous turn]")
        
        # Add emotion context
        if self.last_emotion != "neutral":
            context_parts.append(f"[TONE: User is {self.last_emotion.upper()}]")
        
        # Add last entities for pronoun/reference resolution
        if self.last_entities:
            context_parts.append(f"[CONTEXT: Last mentioned - {self.last_entities}]")
        
        return "\n".join(context_parts)
    
    def extract_entities(self, user_input: str, intent: str) -> dict:
        """
        Extract and track entities mentioned in user input.
        
        Args:
            user_input (str): User's message
            intent (str): Detected intent
        
        Returns:
            dict: Extracted entities
        """
        entities = {}
        
        # Track common entities by intent
        lower_input = user_input.lower()
        
        # Department/Program entities
        if intent == "fees" or intent == "admission" or intent == "placements":
            if any(word in lower_input for word in ["engineering", "engg", "cs", "computer"]):
                entities["department"] = "Engineering"
            elif any(word in lower_input for word in ["arts", "humanities", "liberal"]):
                entities["department"] = "Arts"
            elif any(word in lower_input for word in ["science", "physics", "chemistry"]):
                entities["department"] = "Science"
        
        # Fee-related entities
        if intent == "fees":
            if "scholarship" in lower_input:
                entities["fee_type"] = "Scholarship"
            elif "payment" in lower_input:
                entities["fee_type"] = "Payment options"
            elif "deadline" in lower_input:
                entities["fee_type"] = "Deadline"
        
        # Exam-related entities
        if intent == "exams":
            if "midterm" in lower_input:
                entities["exam_type"] = "Midterm"
            elif "final" in lower_input:
                entities["exam_type"] = "Final"
            elif "schedule" in lower_input or "date" in lower_input:
                entities["exam_info"] = "Dates"
        
        return entities
    
    def resolve_pronouns(self, user_input: str) -> str:
        """
        Resolve pronouns/vague references using conversation history.
        
        Args:
            user_input (str): Current user input
        
        Returns:
            str: Resolved message with context
        """
        lower_input = user_input.lower()
        resolved = user_input
        
        # Replace "it", "that", "this" with previous context
        if ("it" in lower_input or "that" in lower_input or "this" in lower_input) and self.last_entities:
            if "department" in self.last_entities:
                resolved = resolved.replace("it", self.last_entities["department"])
                resolved = resolved.replace("that", self.last_entities["department"])
        
        return resolved
    
    def export_session(self):
        """
        Export the current session data.
        
        Returns:
            dict: Session data
        """
        return {
            "session_start": self.session_start.isoformat(),
            "total_turns": self.total_turns,
            "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "conversation_history": self.get_history(),
            "context_summary": self.get_context_summary()
        }
