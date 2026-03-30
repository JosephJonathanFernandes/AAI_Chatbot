"""
LLM handler for Groq API with Ollama fallback.
Manages API calls, prompt engineering, and response generation.
"""

import os
import time
import requests
from typing import Optional
import json
from dotenv import load_dotenv
from utils import load_json_file, get_time_of_day

# Load environment variables from .env file
load_dotenv()


class LLMHandler:
    """Handles LLM interactions with Groq API and Ollama fallback."""
    
    def __init__(self, groq_api_key: Optional[str] = None, college_data_path: str = "data/college_data.json"):
        """
        Initialize LLM handler.
        
        Args:
            groq_api_key (str): Groq API key (uses env var if not provided)
            college_data_path (str): Path to college knowledge base JSON
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_base_url = "https://api.groq.com/openai/v1"
        self.ollama_base_url = "http://localhost:11434/api/generate"
        
        # Load college knowledge base
        self.college_data = load_json_file(college_data_path)
        
        # Model configuration
        self.groq_model = "llama-3.1-8b-instant"  # Active Groq model (tested 2026-03-30)
        self.ollama_model = "mistral"
        
        # Track fallback usage
        self.fallback_count = 0
        self.groq_success_count = 0
    
    def generate_response(self, user_input: str, intent: str, confidence: float, 
                         emotion: str, context: str = "", stream: bool = False) -> dict:
        """
        Generate response using LLM with context awareness.
        
        Args:
            user_input (str): User's question
            intent (str): Detected intent
            confidence (float): Intent confidence score
            emotion (str): Detected emotion
            context (str): Conversation context
            stream (bool): Enable streaming for faster response time
        
        Returns:
            dict: Contains 'response', 'error', 'source', 'time', 'should_clarify'
        """
        # Check if we need clarification due to low confidence
        # Only clarify for very vague/ambiguous questions (< 15% confidence)
        if confidence < 0.15:
            # Return clarification prompt instead of generating response
            clarification_map = {
                "fees": "Are you asking about tuition fees, payment plans, deposits, or scholarships?",
                "exams": "Do you want to know about exam dates, preparation tips, or passing criteria?",
                "timetable": "Are you looking for class schedules, lab timings, or when college starts?",
                "placements": "Are you interested in placement statistics, companies, or salary info?",
                "admission": "Which aspect interests you - application process, requirements, or cutoff marks?",
                "faculty": "Would you like faculty contact info, office hours, or research areas?",
                "library": "Are you asking about library hours, book availability, or e-resources?",
            }
            
            clarification = clarification_map.get(intent, "Could you provide more details about your question?")
            
            return {
                "response": f"I'm not entirely sure what you're looking for. {clarification}",
                "error": None,
                "source": "clarification",
                "time": 0,
                "should_clarify": True
            }
        
        # Build system prompt
        system_prompt = self._build_system_prompt(intent, confidence, emotion)
        
        # Build user prompt with context
        user_prompt = self._build_user_prompt(user_input, intent, context)
        
        # Try Groq first
        start_time = time.time()
        result = self._call_groq_api(system_prompt, user_prompt)
        response_time = time.time() - start_time
        
        if result and not result.get("error"):
            self.groq_success_count += 1
            return {
                "response": result["response"],
                "error": None,
                "source": "groq",
                "time": response_time,
                "should_clarify": False
            }
        
        # Fallback to Ollama
        print("⚠️ Groq API failed, falling back to Ollama...")
        start_time = time.time()
        result = self._call_ollama_api(system_prompt, user_prompt)
        response_time = time.time() - start_time
        self.fallback_count += 1
        
        if result and not result.get("error"):
            return {
                "response": result["response"],
                "error": None,
                "source": "ollama",
                "time": response_time,
                "should_clarify": False
            }
        
        # If both fail, return error
        return {
            "response": "I apologize, but I'm unable to process your request right now. Please try again later.",
            "error": "Both API and fallback failed",
            "source": "error",
            "time": response_time,
            "should_clarify": False
        }
    
    def _call_groq_api(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """
        Call Groq API for response generation.
        
        Args:
            system_prompt (str): System prompt with context
            user_prompt (str): User's question with context
        
        Returns:
            dict: API response or None on error
        """
        if not self.groq_api_key:
            print("⚠️ GROQ_API_KEY not set. Skipping Groq API.")
            return None
        
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
                timeout=8
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data["choices"][0]["message"]["content"].strip()
                }
            else:
                print(f"Groq API error: {response.status_code}")
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
            system_prompt (str): System prompt with context
            user_prompt (str): User's question with context
        
        Returns:
            dict: API response or None on error
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
                timeout=10  # 10 second timeout to prevent hanging
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "").strip()
                
                # Clean response
                if response_text:
                    return {"response": response_text}
                else:
                    return {"error": "Empty response"}
            else:
                print(f"Ollama API error: {response.status_code}")
                return {"error": f"Status {response.status_code}"}
        
        except requests.exceptions.ConnectionError:
            print("Ollama not running. Make sure to start Ollama with: ollama serve")
            return {"error": "Ollama not available"}
        except Exception as e:
            print(f"Ollama API error: {e}")
            return {"error": str(e)}
    
    def _build_system_prompt(self, intent: str, confidence: float, emotion: str) -> str:
        """
        Build context-aware system prompt (optimized for speed & accuracy).
        
        Args:
            intent (str): Detected intent
            confidence (float): Intent confidence
            emotion (str): Detected emotion
        
        Returns:
            str: System prompt for LLM
        """
        time_of_day = get_time_of_day()
        
        confidence_guidance = ""
        if confidence < 0.4:
            confidence_guidance = "\n[LOW CONFIDENCE: Ask for clarification before answering]"
        elif confidence < 0.6:
            confidence_guidance = "\n[MODERATE CONFIDENCE: Confirm understanding before detailed answer]"
        else:
            confidence_guidance = "\n[HIGH CONFIDENCE: Proceed with answering]"
        
        prompt = f"""You are a helpful college assistant from {self.college_data.get('college_name', 'Advanced Academic Institute')}.

CONTEXT AWARENESS RULES:
- MAINTAIN conversation flow - reference previous topics when relevant
- If user says "it", "that", "for X" - connect to earlier mentions
- Build on what was already discussed in the conversation
- If continuing same topic, provide NEW details, not repetition

ANTI-HALLUCINATION RULES:
- Only state facts from college knowledge base
- If uncertain about specifics: "Let me provide what I know..."
- Never invent dates, numbers, fees, or policies
- Admit uncertainty gracefully

COMMUNICATION STYLE:
1. Be DIRECT & CONCISE (2-3 sentences for factual Qs)
2. CLARIFY if request is ambiguous: "Just to confirm - are you asking about...?"
3. Always show you're connecting to previous conversation
4. Match tone to user emotion: {emotion}
5. For vague Qs: Ask guided questions before answering{confidence_guidance}

EXAMPLE PATTERNS:
- Previous Q: "What are fees?" → Answer: "$85k for engineering..."
- Follow-up Q: "for engineering?" → Don't repeat; add NEW info: "...with payment plans available..."
"""
        return prompt
    
    def _build_user_prompt(self, user_input: str, intent: str, context: str = "") -> str:
        """
        Build user prompt with relevant context (optimized for speed).
        
        Args:
            user_input (str): User's question
            intent (str): Detected intent
            context (str): Conversation context
        
        Returns:
            str: Formatted user prompt
        """
        # Minimal context for faster processing
        parts = []
        if context:
            parts.append(context)
        
        college_context = self._get_college_context(intent)
        if college_context:
            parts.append(college_context)
        
        parts.append(f"Q: {user_input}\nA: Be concise. Admit if unsure.")
        return "\n".join(parts)
    
    def _get_college_context(self, intent: str) -> str:
        """
        Extract relevant college data based on intent.
        
        Args:
            intent (str): Detected intent
        
        Returns:
            str: Formatted college information
        """
        context_parts = []
        
        if not self.college_data:
            return ""
        
        intent_data_map = {
            "fees": ("fees", "Fee Information"),
            "exams": ("exams", "Exam Information"),
            "timetable": ("timetable", "Class Schedule"),
            "placements": ("placements", "Placement Information"),
            "faculty": ("faculty", "Faculty Information"),
            "holidays": ("holidays", "Holiday Schedule"),
            "library": ("library", "Library Information"),
            "admission": ("admission", "Admission Information"),
            "departments": ("departments", "Departments")
        }
        
        if intent in intent_data_map:
            key, label = intent_data_map[intent]
            data = self.college_data.get(key)
            
            if data:
                context_parts.append(f"{label}:")
                context_parts.append(json.dumps(data, indent=2)[:500])  # Limit size
        
        return "\n".join(context_parts) if context_parts else ""
    
    def validate_response(self, response: str, intent: str) -> bool:
        """
        Validate if response is appropriate for the context.
        
        Args:
            response (str): Generated response
            intent (str): Intent tag
        
        Returns:
            bool: True if response is valid
        """
        # Check minimum length
        if len(response.strip()) < 5:
            return False
        
        # Avoid very short responses
        if len(response.strip()) < 20 and "sorry" in response.lower() and "can't" in response.lower():
            return False
        
        return True
    
    def get_stats(self) -> dict:
        """
        Get LLM handler statistics.
        
        Returns:
            dict: Usage statistics
        """
        total_calls = self.groq_success_count + self.fallback_count
        
        return {
            "total_calls": total_calls,
            "groq_success": self.groq_success_count,
            "fallback_count": self.fallback_count,
            "fallback_rate": round(self.fallback_count / max(total_calls, 1), 3),
            "groq_available": bool(self.groq_api_key)
        }
