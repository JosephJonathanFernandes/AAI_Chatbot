"""
Prompt engineering module for structured, context-aware LLM prompts.
Builds system and user prompts with confidence-based behavior control.
"""

import json
from typing import List, Dict, Optional


class PromptEngineer:
    """Constructs structured prompts for LLM with context awareness."""
    
    def __init__(self):
        """Initialize prompt engineer."""
        self.max_context_tokens = 1500
        self.max_history_turns = 5
    
    def build_system_prompt(self, 
                          intent: str, 
                          confidence: float = 0.5, 
                          emotion: str = "neutral",
                          is_in_scope: bool = True,
                          scope_reason: str = "domain_keywords") -> str:
        """
        Build a structured system prompt for the LLM.
        
        Args:
            intent (str): Detected intent
            confidence (float): Confidence score (0-1), default 0.5
            emotion (str): Detected emotion, default 'neutral'
            is_in_scope (bool): Whether query is within college domain
            scope_reason (str): Reason for scope determination
        
        Returns:
            str: System prompt for LLM
        """
        system_prompt = """You are an AI assistant for a college. Your role is to help students and staff with college-related queries.

CRITICAL CONSTRAINTS:
1. You MUST ONLY answer college-related questions. Do NOT answer questions outside college domain.
2. If a query is outside your scope, respond EXACTLY with: "This is beyond my scope as a college assistant. I can help with fees, exams, placements, faculty, admissions, library, hostel, and other college-related information."
3. Do NOT hallucinate or invent information. Use ONLY provided knowledge or explicitly state uncertainty.
4. Be polite, helpful, and student-friendly in all responses.
5. ALWAYS keep responses SHORT: 1-2 sentences max for direct answers. NO elaborate explanations or background info unless specifically asked.

QUERY ANALYSIS:
- User Intent: {intent}
- Intent Confidence: {confidence:.1%}
- Detected Emotion: {emotion}
- Domain Scope: {"IN-SCOPE" if is_in_scope else "OUT-OF-SCOPE"} (Reason: {scope_reason})

BEHAVIOR INSTRUCTIONS:
"""
        
        # Add emotion-specific instructions
        if emotion == "stressed":
            system_prompt += """- User appears STRESSED. Respond calmly, reassuringly, and provide concrete guidance.
  Include keywords like "You can", "It's manageable", "Here's how to..."
"""
        elif emotion == "confused":
            system_prompt += """- User appears CONFUSED. Provide step-by-step explanations, use examples, and be extra clear.
  Break down information into digestible parts.
"""
        elif emotion == "happy":
            system_prompt += """- User appears HAPPY/SATISFIED. Respond positively and match their enthusiastic tone.
"""
        elif emotion == "angry":
            system_prompt += """- User appears ANGRY/FRUSTRATED. Be empathetic, apologetic where appropriate, and solution-focused.
  Validate their concern first, then provide help.
"""
        
        # Add confidence-based instructions
        if confidence < 0.3 and is_in_scope:
            system_prompt += f"""
- Intent confidence is LOW ({confidence:.1%}). Ask ONE short clarification: "Are you asking about [A] or [B]?"
"""
        
        # Add scope-based instructions
        if not is_in_scope:
            system_prompt += """
- This query is OUTSIDE college domain. Respond with the standard out-of-scope message above.
"""
        else:
            system_prompt += """
- This query IS within college domain. Provide helpful, accurate information.
"""
        
        system_prompt += """
TONE & STYLE:
- Professional but approachable
- Use "I can help with..." and "Let me clarify..."
- Avoid jargon; use simple English
- Be encouraging and supportive

Remember: Your goal is to help students make informed decisions about their college experience."""
        
        return system_prompt
    
    def build_user_prompt(self,
                         user_input: str,
                         intent: str,
                         confidence: float = 0.5,
                         conversation_context: Optional[str] = None,
                         conversation_history: List[Dict[str, str]] = None,
                         relevant_knowledge: Optional[str] = None,
                         time_context: Optional[str] = None) -> str:
        """
        Build a structured user prompt with context.
        
        Args:
            user_input (str): User's question
            intent (str): Detected intent
            confidence (float): Confidence in intent detection
            conversation_context (str): Context from previous conversations
            conversation_history (List): Previous conversation turns
            relevant_knowledge (str): Relevant college information
            time_context (str): Time-based context (e.g., "It's exam season")
        
        Returns:
            str: User prompt with context
        """
        prompt = f"""QUERY: {user_input}

CONTEXT INFORMATION:
"""
        
        # Add detected intent and confidence
        prompt += f"- Detected Intent: {intent} (Confidence: {confidence:.1%})\n"
        
        # Add conversation context if provided
        if conversation_context:
            prompt += f"- Conversation Context: {conversation_context}\n"
        
        # Add conversation history if provided
        if conversation_history and len(conversation_history) > 0:
            prompt += f"\n- Conversation History ({len(conversation_history)} previous turns):\n"
            for i, turn in enumerate(conversation_history[-self.max_history_turns:], 1):
                prompt += f"  Turn {i}:\n"
                # Defensive: handle both dict and non-dict turns
                if isinstance(turn, dict):
                    user_text = turn.get('user_input', '')[:200]
                    bot_text = turn.get('bot_response', '')[:200]
                else:
                    user_text = str(turn)[:200] if turn else ""
                    bot_text = ""
                prompt += f"    User: {user_text}...\n"
                if bot_text:
                    prompt += f"    You: {bot_text}...\n"
        
        # Add relevant knowledge base information
        if relevant_knowledge:
            prompt += f"\n- RELEVANT COLLEGE INFORMATION:\n{relevant_knowledge}\n"
        
        # Add time-based context
        if time_context:
            prompt += f"\n- TIME CONTEXT: {time_context}\n"
        
        # Add final instruction
        prompt += f"""
RESPOND TO THE QUERY ABOVE:
- Use the provided context information if relevant
- If information is not in the knowledge base, you can use general college knowledge
- Always be helpful and student-friendly
- Ask for clarification if the query is ambiguous"""
        
        return prompt
    
    def build_clarification_prompt(self, intent: str, detected_intent_confidence: float) -> str:
        """
        Build a prompt for asking clarification questions.
        
        Args:
            intent (str): Detected intent
            detected_intent_confidence (float): Confidence score
        
        Returns:
            str: Clarification prompt template
        """
        clarification_templates = {
            "fees": "Are you asking about tuition costs, financial aid, or payment plans?",
            "exams": "Are you asking about exam dates or preparation?",
            "timetable": "Are you asking about class schedules or academic calendar dates?",
            "placements": "Are you asking about placement stats or company hiring?",
            "library": "Are you asking about access hours or book resources?",
            "admission": "Are you asking about eligibility or application process?"
        }
        
        template = clarification_templates.get(intent, 
            "Could you provide more details about what you're asking? This will help me give you the best answer.")
        
        return template
    
    def build_fallback_prompt(self) -> str:
        """Build a fallback prompt for handling errors."""
        return """I'm having trouble processing your request at the moment. 
Could you please try asking again? If the issue persists, please contact the college administration office.

In the meantime, you can ask me about:
- Fees and financial aid
- Exams and academic schedules
- Placements and career services
- Faculty and department information
- Library and campus facilities"""
