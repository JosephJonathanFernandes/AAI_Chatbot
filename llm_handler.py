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
from functools import lru_cache
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
        """
        try:
            # Safety check for user_input
            if not user_input or not isinstance(user_input, str):
                user_input = "Hello"
            
            self.total_api_calls += 1
            start_time = time.time()
            
            # STEP 1: Check scope
            scope_info = self.scope_detector.get_scope_info(user_input, intent, confidence)
            if not isinstance(scope_info, dict):
                print(f"ERROR: scope_info is {type(scope_info)}, not dict: {scope_info}")
                scope_info = {"is_in_scope": True, "reason": "error", "out_of_scope_response": "", "should_clarify": False, "confidence": 0.5}
            
            is_in_scope = scope_info.get("is_in_scope", True)
            
            # STEP 2: Get relevant knowledge grounding
            relevant_knowledge = self._get_grounded_knowledge(intent, user_input)
            
            # STEP 3: Get time context
            time_context_str = self._get_time_context()
            
            # STEP 4: Format conversation history
            formatted_history = self._format_conversation_history(conversation_history)
            
            # STEP 4B: Count prior clarifications to avoid clarification loops
            clarification_count = 0
            if conversation_history:
                for turn in conversation_history:
                    if isinstance(turn, dict):
                        bot_resp = turn.get('bot_response', '').lower()
                        # Count if bot asked a clarification question (?, or "are you asking")
                        if ('are you' in bot_resp or 'could you' in bot_resp or '?' in bot_resp) and len(bot_resp) < 200:
                            clarification_count += 1
            
            # STEP 5: Build structured prompts
            system_prompt = self.prompt_engineer.build_system_prompt(
                intent=intent,
                confidence=confidence,
                emotion=emotion,
                is_in_scope=is_in_scope,
                scope_reason=scope_info.get("reason", ""),
                clarification_count=clarification_count
            )
            
            user_prompt = self.prompt_engineer.build_user_prompt(
                user_input=user_input,
                intent=intent,
                conversation_history=conversation_history or [],
                relevant_knowledge=relevant_knowledge,
                time_context=time_context_str
            )
            
            # STEP 6: Add tone guidelines if provided
            if tone_guidelines and isinstance(tone_guidelines, dict):
                tone_instruction = f"\n\nRESPONSE TONE: {tone_guidelines.get('tone', 'informative').upper()}\n"
                tone_instruction += f"- Format: {tone_guidelines.get('length', 'standard')}\n"
                tone_instruction += f"- Formality: {tone_guidelines.get('formality', 'professional')}\n"
                tone_instruction += f"- Detail Level: {tone_guidelines.get('detail_level', 'appropriate')}\n"
                if tone_guidelines.get('prefix'):
                    tone_instruction += f"- Start with: {tone_guidelines['prefix']}\n"
                if tone_guidelines.get('suffix'):
                    tone_instruction += f"- End with: {tone_guidelines['suffix']}\n"
                system_prompt += tone_instruction
            
            # STEP 7: Add confidence-based behavior instruction (ONLY if no prior clarifications)
            if confidence < 0.4 and clarification_count == 0:
                clarification_guide = self.prompt_engineer.build_clarification_prompt(intent, confidence)
                system_prompt += f"\n\nINSTRUCTION FOR LOW CONFIDENCE:\nAsk this ONE clarification question:\n{clarification_guide}"
            elif clarification_count > 0:
                system_prompt += "\n\nCLARIFICATION ALREADY DONE: Provide your best answer without asking again."
            
            # STEP 8: Add scope instruction
            if not is_in_scope:
                system_prompt += f"\n\nIMPORTANT: This query is OUT-OF-SCOPE. Respond with:\n'{scope_info.get('out_of_scope_response', '')}'"
            
            # STEP 9: Try Groq API first
            result = self._call_groq_api(system_prompt, user_prompt)
            response_time = time.time() - start_time
            
            if result and isinstance(result, dict) and not result.get("error"):
                self.groq_success_count += 1
                return {
                    "response": result.get("response", ""),
                    "error": None,
                    "source": "groq",
                    "time": response_time,
                    "is_in_scope": is_in_scope,
                    "intent": intent,
                    "confidence": confidence,
                    "emotion": emotion,
                    "should_clarify": confidence < 0.4 and clarification_count == 0
                }
            
            # STEP 10: Fallback to Ollama
            print("⚠️ Groq API failed, falling back to Ollama...")
            result = self._call_ollama_api(system_prompt, user_prompt)
            response_time = time.time() - start_time
            self.fallback_count += 1
            
            if result and isinstance(result, dict) and not result.get("error"):
                return {
                    "response": result.get("response", ""),
                    "error": None,
                    "source": "ollama",
                    "time": response_time,
                    "is_in_scope": is_in_scope,
                    "intent": intent,
                    "confidence": confidence,
                    "emotion": emotion,
                    "should_clarify": confidence < 0.3
                }
            
            # STEP 11: If all fails, return contextual fallback based on detected intent
            fallback_response = self.prompt_engineer.build_fallback_prompt(intent)
            return {
                "response": fallback_response,
                "error": "Both APIs failed",
                "source": "fallback",
                "time": response_time,
                "is_in_scope": is_in_scope,
                "intent": intent,
                "confidence": confidence,
                "emotion": emotion,
                "should_clarify": False
            }
        
        except Exception as e:
            print(f"CRITICAL ERROR in generate_response: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "response": "I'm having technical difficulties. Please try again.",
                "error": str(e),
                "source": "error",
                "time": time.time() - start_time,
                "is_in_scope": True,
                "intent": intent,
                "confidence": 0.0,
                "emotion": emotion,
                "should_clarify": True
            }
    
    @lru_cache(maxsize=128)
    def _get_grounded_knowledge_cached(self, intent: str, query: str) -> str:
        """
        Cached knowledge retrieval to avoid recomputing for same intent+query combinations.
        LRU cache stores up to 128 unique intent/query pairs (~50KB memory).
        """
        return self._get_grounded_knowledge_impl(intent, query)
    
    def _get_grounded_knowledge(self, intent: str, query: str) -> str:
        """
        Get relevant knowledge grounding from college_data based on intent.
        This is RAG-lite - retrieve only relevant sections + structured generic answers.
        Uses caching to avoid redundant knowledge retrieval.
        
        Args:
            intent (str): Detected intent
            query (str): User's question (for context)
        
        Returns:
            str: Formatted relevant knowledge (cached)
        """
        # Use cached version for repeated intent/query combinations
        return self._get_grounded_knowledge_cached(intent, query)
    
    def _get_grounded_knowledge_impl(self, intent: str, query: str) -> str:
        """Implementation of knowledge grounding (called by cached wrapper)."""
        knowledge_parts = []
        
        # Generic structured answers to provide when exact data is unavailable
        generic_knowledge = {
            "fees": """
GENERIC COLLEGE FEE STRUCTURE (Use if exact data unavailable):
- Academic fees: Charged per semester/year
- Lab/Practical charges: For engineering/science programs
- Library & Development fees: Typically included
- Examination fees: For each exam/semester
- Registration/Admission fee: Usually one-time payment

Payment Options (typical):
- Online portal: UPI, Card, Net Banking
- Cheque/DD to college account
- Installment plans: Available (contact admissions office)

Admission fees are typically separate from tuition.
            """,
            "exams": """
GENERIC EXAM SCHEDULE (Use if exact dates unavailable):
- Mid-semester exams: Usually around 1/3rd through semester
- End-semester exams: Final month of semester
- Exam dates: Posted 2-3 weeks in advance
- Results: Usually 2-3 weeks after exam ends

Typical exam format:
- Written exams: 2-3 hours
- Practical exams: 3-4 hours
- Projects/Internals: Continuous evaluation
- Online exams: As per college schedule
            """,
            "placements": """
GENERIC PLACEMENT PROCESS:
- Eligibility: Usually min 60-70% CGPA
- Recruitment season: August-November (peak)
- Companies: 50-100+ companies typically visit
- Average package: ₹5-15 LPA (varies by stream)
- Placement rate: 85-95% (typical)
- Process: Resume → Group Discussion → Interview → Offer

Higher studies option: Available if not interested in campus placements
            """,
            "hostel": """
GENERIC HOSTEL INFO:
- Types: Single/Double/Triple sharing rooms
- Capacity: Usually 70-90% of student strength
- Facilities: WiFi, Water, Electricity, Common areas
- Mess: Usually included in hostel fee
- Rules: Curfew typically 10-11 PM, Visitor policy, ID required
- Gender: Separate hostels for boys/girls
- Fee: Usually ₹20,000-60,000/year (varies by city and room type)

Off-campus options: Available with college approval
            """,
            "admission": """
GENERIC ADMISSION PROCESS:
1. Check eligibility (10+2 or equivalent)
2. Fill application form (online)
3. Merit-based selection (entrance exam or 12th marks)
4. Document verification & counseling
5. Fee payment & enrollment
6. Orientation & classes begin

Entrance exams: JEE/BITSAT/GATE (varies by program)
Documents needed: 10th/12th marks, Aadhar, Address proof
Application fee: Usually ₹500-2000
            """
        }
        
        # Get specific college data first
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
        has_specific_data = False
        
        # OPTIMIZATION: Limit to top 2 knowledge items to reduce tokens by ~40%
        for key in relevant_keys[:2]:  # Only first 2 keys
            if key in self.college_data and self.college_data[key]:
                has_specific_data = True
                content = self.college_data[key]
                
                # Format based on content type (truncate to 150 chars per item)
                if isinstance(content, dict):
                    items = list(content.items())[:3]  # Limit to top 3 key-value pairs
                    formatted = "\n".join([f"- {k}: {str(v)[:80]}" for k, v in items])
                elif isinstance(content, list):
                    items = content[:3]  # Limit to top 3 items
                    formatted = "\n".join([f"- {str(item)[:80]}" for item in items])
                else:
                    formatted = str(content)[:200]
                
                knowledge_parts.append(f"**{key.replace('_', ' ').title()}:**\n{formatted}")
        
        if knowledge_parts:
            return "\n\n".join(knowledge_parts)
        
        # FALLBACK: Use generic knowledge if no specific college data found
        # This ensures LLM always has structured answer patterns
        if intent in generic_knowledge:
            return f"[Using generic college knowledge pattern]\n{generic_knowledge[intent]}"
        
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
        """Format conversation history for LLM context (optimized for token reduction)."""
        if not history or len(history) == 0:
            return []
        
        # Return only last 1-2 turns to reduce token usage (was 3)
        # Each turn ~ 100-150 tokens; limiting to 1-2 saves 150-300 tokens
        return history[-1:] if len(history) > 1 else history
    
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
                "max_tokens": 250,
                "top_p": 0.85
            }
            
            response = requests.post(
                f"{self.groq_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if not isinstance(data, dict):
                        return {"error": "Response is not JSON dict"}
                    response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if response_text:
                        return {"response": response_text}
                    else:
                        return {"error": "Empty response from Groq"}
                except (ValueError, KeyError, IndexError, TypeError) as parse_error:
                    print(f"Error parsing Groq response: {parse_error}")
                    return {"error": f"Parse error: {str(parse_error)}"}
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
                try:
                    data = response.json()
                    if not isinstance(data, dict):
                        return {"error": "Response is not JSON dict"}
                    response_text = data.get("response", "").strip()
                    if response_text:
                        return {"response": response_text}
                    else:
                        return {"error": "Empty response from Ollama"}
                except (ValueError, TypeError) as parse_error:
                    print(f"Error parsing Ollama response: {parse_error}")
                    return {"error": f"Parse error: {str(parse_error)}"}
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
