"""
LLM handler for Groq API with Ollama fallback.
Enhanced with structured prompt engineering, knowledge grounding (RAG-lite), 
and scope detection for controlled LLM responses.

Features:
- Always uses LLM for response generation (never bypass)
- Knowledge grounding reduces hallucination
- Scope detection catches out-of-domain queries
- Emotion-aware response toning
- Structured prompt engineering with context injection
"""

import os
import time
import requests
from typing import Optional, List, Dict
import json
from dotenv import load_dotenv
from utils import load_json_file, get_time_of_day
from time_context import TimeContext
from prompt_engineering import PromptEngineer
from scope_detector import ScopeDetector

# Load environment variables from .env file
load_dotenv()


class LLMHandler:
    """Handles LLM interactions with Groq API and Ollama fallback with knowledge grounding."""
    
    def __init__(self, groq_api_key: Optional[str] = None, college_data_path: str = "data/college_data.json"):
        """
        Initialize LLM handler with enhanced prompt engineering and scope detection.
        
        Args:
            groq_api_key (str): Groq API key (uses env var if not provided)
            college_data_path (str): Path to college knowledge base JSON
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_base_url = "https://api.groq.com/openai/v1"
        self.ollama_base_url = "http://localhost:11434/api/generate"
        
        # Load college knowledge base for RAG-lite
        self.college_data = load_json_file(college_data_path)
        
        # Initialize enhanced components
        self.time_context = TimeContext(college_data_path)
        self.prompt_engineer = PromptEngineer()
        self.scope_detector = ScopeDetector()
        
        # Model configuration
        self.groq_model = "llama-3.1-8b-instant"
        self.ollama_model = "mistral"
        
        # Track metrics
        self.fallback_count = 0
        self.groq_success_count = 0
        self.total_api_calls = 0
    
    def generate_response(self, 
                         user_input: str, 
                         intent: str, 
                         confidence: float,
                         emotion: str, 
                         conversation_history: List[Dict] = None,
                         stream: bool = False,
                         tone_guidelines: dict = None) -> dict:
        """
        Generate response using LLM with ALWAYS-ON strategy (never bypass LLM).
        
        CRITICAL: The LLM is ALWAYS used to generate the final response.
        Low confidence → instruct LLM to ask clarification
        Out-of-scope → instruct LLM to reply with standard message
        
        Args:
            user_input (str): User's question
            intent (str): Detected intent
            confidence (float): Intent confidence score (0-1)
            emotion (str): Detected emotion
            conversation_history (List): Previous conversation turns
            stream (bool): Enable streaming
            tone_guidelines (dict): Response tone guidelines (PHASE 2)
        
        Returns:
            dict: Response with metadata (response, source, time, is_in_scope, etc.)
        """
        self.total_api_calls += 1
        start_time = time.time()
        
        # STEP 1: Check scope (but don't bypass LLM - inform it about scope)
        scope_info = self.scope_detector.get_scope_info(user_input, intent, confidence)
        is_in_scope = scope_info["is_in_scope"]
        
        # STEP 2: Get relevant knowledge grounding (RAG-lite)
        relevant_knowledge = self._get_grounded_knowledge(intent, user_input)
        
        # STEP 3: Get time context
        time_context_str = self._get_time_context()
        
        # STEP 4: Format conversation history (last 3-5 turns)
        formatted_history = self._format_conversation_history(conversation_history)
        
        # STEP 5: Build structured prompts (ALWAYS use LLM)
        system_prompt = self.prompt_engineer.build_system_prompt(
            intent=intent,
            confidence=confidence,
            emotion=emotion,
            is_in_scope=is_in_scope,
            scope_reason=scope_info["reason"]
        )
        
        user_prompt = self.prompt_engineer.build_user_prompt(
            user_input=user_input,
            intent=intent,
            conversation_history=conversation_history or [],
            relevant_knowledge=relevant_knowledge,
            time_context=time_context_str
        )
        
        # STEP 6: PHASE 2 - Add tone guidelines if provided
        if tone_guidelines:
            tone_instruction = f"\n\nRESPONSE TONE: {tone_guidelines.get('tone', 'informative').upper()}\n"
            tone_instruction += f"- Format: {tone_guidelines.get('length', 'standard')}\n"
            tone_instruction += f"- Formality: {tone_guidelines.get('formality', 'professional')}\n"
            tone_instruction += f"- Detail Level: {tone_guidelines.get('detail_level', 'appropriate')}\n"
            if tone_guidelines.get('prefix'):
                tone_instruction += f"- Start with: {tone_guidelines['prefix']}\n"
            if tone_guidelines.get('suffix'):
                tone_instruction += f"- End with: {tone_guidelines['suffix']}\n"
            system_prompt += tone_instruction
        
        # STEP 7: Add confidence-based behavior instruction to system prompt
        if confidence < 0.3:
            # Low confidence → instruct LLM to ask clarification
            clarification_guide = self.prompt_engineer.build_clarification_prompt(intent, confidence)
            system_prompt += f"\n\nINSTRUCTION FOR LOW CONFIDENCE:\nAsk this clarification question before providing an answer:\n{clarification_guide}"
        
        # STEP 8: Add scope instruction to system prompt
        if not is_in_scope:
            system_prompt += f"\n\nIMPORTANT: This query is OUT-OF-SCOPE. Respond with:\n'{scope_info['out_of_scope_response']}'"
        
        # STEP 9: Try Groq API first
        result = self._call_groq_api(system_prompt, user_prompt)
        response_time = time.time() - start_time
        
        if result and not result.get("error"):
            self.groq_success_count += 1
            return {
                "response": result["response"],
                "error": None,
                "source": "groq",
                "time": response_time,
                "is_in_scope": is_in_scope,
                "intent": intent,
                "confidence": confidence,
                "emotion": emotion,
                "should_clarify": confidence < 0.3
            }
        
        # STEP 10: Fallback to Ollama
        print("⚠️ Groq API failed, falling back to Ollama...")
        result = self._call_ollama_api(system_prompt, user_prompt)
        response_time = time.time() - start_time
        self.fallback_count += 1
        
        if result and not result.get("error"):
            return {
                "response": result["response"],
                "error": None,
                "source": "ollama",
                "time": response_time,
                "is_in_scope": is_in_scope,
                "intent": intent,
                "confidence": confidence,
                "emotion": emotion,
                "should_clarify": confidence < 0.3
            }
        
        # STEP 10: If all fails, return fallback response
        fallback_response = self.prompt_engineer.build_fallback_prompt()
        return {
            "response": fallback_response,
            "error": "Both Groq and Ollama failed",
            "source": "fallback",
            "time": response_time,
            "is_in_scope": is_in_scope,
            "intent": intent,
            "confidence": confidence,
            "emotion": emotion,
            "should_clarify": False
        }
    
    def _get_grounded_knowledge(self, intent: str, query: str) -> str:
        """
        Get relevant knowledge grounding from college_data based on intent.
        This is RAG-lite - retrieve only relevant sections.
        
        Args:
            intent (str): Detected intent
            query (str): User's question (for context)
        
        Returns:
            str: Formatted relevant knowledge
        """
        knowledge_parts = []
        
        # Map intents to college_data sections
        intent_to_keys = {
            "fees": ["tuition_fees", "scholarships", "payment_plans"],
            "exams": ["exam_schedule", "grading_system", "exam_rules"],
            "timetable": ["academic_calendar", "class_schedule", "term_dates"],
            "placements": ["placement_stats", "companies", "placement_process"],
            "faculty": ["faculty_directory", "department_info", "office_hours"],
            "library": ["library_info", "resources", "access_hours"],
            "admission": ["admission_criteria", "application_process", "cutoff_marks"],
            "hostel": ["hostel_info", "accommodation", "hostel_rules"],
            "sports": ["sports_facilities", "sports_clubs", "events"],
            "college_info": ["college_overview", "mission", "vision"]
        }
        
        relevant_keys = intent_to_keys.get(intent, [])
        
        for key in relevant_keys:
            if key in self.college_data and self.college_data[key]:
                content = self.college_data[key]
                
                # Format based on content type
                if isinstance(content, dict):
                    formatted = "\n".join([f"- {k}: {v}" for k, v in content.items()])
                elif isinstance(content, list):
                    formatted = "\n".join([f"- {item}" for item in content])
                else:
                    formatted = str(content)
                
                knowledge_parts.append(f"**{key.replace('_', ' ').title()}:**\n{formatted}")
        
        if knowledge_parts:
            return "\n\n".join(knowledge_parts)
        return ""
    
    def _get_time_context(self) -> str:
        """Get time-based context for responses."""
        time_of_day = get_time_of_day()
        context = f"Current time of day: {time_of_day}"
        
        # Check if exam season or special date
        special_context = self.time_context.get_context_awareness_prompt()
        if special_context and special_context != "":
            context += f"\nSpecial context: {special_context}"
        
        return context
    
    def _format_conversation_history(self, history: List[Dict]) -> List[Dict]:
        """Format conversation history for LLM context."""
        if not history or len(history) == 0:
            return []
        
        # Return last 3 turns
        return history[-3:] if len(history) > 3 else history
    
    def _call_groq_api(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """
        Call Groq API for response generation.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
        
        Returns:
            dict: Response or None on error
        """
        if not self.groq_api_key:
            print("⚠️ GROQ_API_KEY not set in environment")
            return {"error": "No API key"}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.groq_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.6,
                "max_tokens": 300,
                "top_p": 0.85
            }
            
            response = requests.post(
                f"{self.groq_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data["choices"][0]["message"]["content"].strip()
                return {"response": response_text}
            else:
                print(f"Groq API error {response.status_code}")
                return {"error": f"Status {response.status_code}"}
        
        except requests.exceptions.Timeout:
            print("Groq API timeout")
            return {"error": "Timeout"}
        except requests.exceptions.ConnectionError:
            print("Groq API connection error")
            return {"error": "Connection error"}
        except Exception as e:
            print(f"Groq API error: {e}")
            return {"error": str(e)}
    
    def _call_ollama_api(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """
        Call Ollama API (local) for response generation.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
        
        Returns:
            dict: Response or None on error
        """
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "temperature": 0.7,
                "num_predict": 500
            }
            
            response = requests.post(
                self.ollama_base_url,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "").strip()
                
                if response_text:
                    return {"response": response_text}
                else:
                    return {"error": "Empty response"}
            else:
                print(f"Ollama API error {response.status_code}")
                return {"error": f"Status {response.status_code}"}
        
        except requests.exceptions.ConnectionError:
            print("Ollama not running. Start with: ollama serve")
            return {"error": "Ollama not available"}
        except Exception as e:
            print(f"Ollama error: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> dict:
        """Get handler statistics."""
        total_successes = self.groq_success_count + (self.total_api_calls - self.groq_success_count - self.fallback_count)
        
        return {
            "total_api_calls": self.total_api_calls,
            "groq_successes": self.groq_success_count,
            "fallback_count": self.fallback_count,
            "groq_success_rate": round(self.groq_success_count / max(self.total_api_calls, 1) * 100, 2),
            "fallback_rate": round(self.fallback_count / max(self.total_api_calls, 1) * 100, 2)
        }
