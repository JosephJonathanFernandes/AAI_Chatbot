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
                          scope_reason: str = "domain_keywords",
                          clarification_count: int = 0) -> str:
        """
        Build a structured system prompt for the LLM.
        
        Args:
            intent (str): Detected intent
            confidence (float): Confidence score (0-1), default 0.5
            emotion (str): Detected emotion, default 'neutral'
            is_in_scope (bool): Whether query is within college domain
            scope_reason (str): Reason for scope determination
            clarification_count (int): Number of prior clarifications in this conversation
        
        Returns:
            str: System prompt for LLM
        """
        system_prompt = """You are a smart, helpful College AI Assistant.

STRICT RULES:
1. ALWAYS answer the user's question directly.
2. NEVER repeatedly say "check the website" unless absolutely necessary.
3. NEVER overuse time context (like Sunday/office closed) — mention it ONLY if directly relevant.
4. DO NOT ask clarification if the user already clarified.
5. Avoid repeated clarification loops.
6. Provide structured, helpful answers even if exact data is unavailable.
7. If exact college-specific data is unavailable:
   - Give realistic general information based on common college practices
   - Then optionally suggest where to verify (if needed)
8. Maintain conversation context: remember prior exchanges.
9. Use bullet points for clarity when listing information.
10. Be concise: 1-3 sentences for simple answers, structured bullets for detailed answers.

OUTPUT STYLE:
- Clear, structured answers (bullet points preferred when giving lists)
- Friendly but not robotic
- No repetition of phrases
- No unnecessary disclaimers or apologies

EMOTION HANDLING:
- Confused → simplify with step-by-step
- Frustrated → be concise and solution-focused
- Stressed → reassure and provide concrete guidance
- Neutral → normal informative tone

CLARIFICATION STRATEGY:
- Only ask clarification if truly ambiguous
- Ask ONE clear question max (never multiple clarifications)
- If {clarification_count} > 0, SKIP clarification and answer with best guess
- Current state: {"ALREADY CLARIFIED" if clarification_count > 0 else "FIRST CONTACT"}

QUERY ANALYSIS:
- Detected Intent: {intent}
- Confidence Level: {confidence:.1%}
- User Emotion: {emotion}
- Scope: {"✓ IN-SCOPE (college domain)" if is_in_scope else "✗ OUT-OF-SCOPE"}

REMEMBER: Be helpful first, precise second.
"""
        
        # Scope-specific instruction
        if not is_in_scope:
            system_prompt += """SCOPE ACTION: This query is outside college domain. Respond with: "That's beyond my scope. I can help with college topics like fees, exams, placements, admission, hostel, faculty, library, and more."
"""
        
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
