"""
LLM handler for Groq API with Gemini fallback.
Enhanced with structured prompt engineering, knowledge grounding (RAG-lite), 
and scope detection for controlled LLM responses.

Features:
- Always uses LLM for response generation (never bypass)
- Knowledge grounding reduces hallucination
- Scope detection catches out-of-domain queries
- Emotion-aware response toning
- Structured prompt engineering with context injection
- Automatic Groq key rotation with exponential backoff
- Fast fallback to Google Gemini API
"""

import os
import time
import requests
import random
import threading
import hashlib
import re
from queue import Queue
from typing import Optional, List, Dict, Generator
import json
from functools import lru_cache
from collections import defaultdict
from dotenv import load_dotenv
from google import genai
from utils import load_json_file, get_time_of_day
from time_context import TimeContext
from prompt_engineering import PromptEngineer
from scope_detector import ScopeDetector
from confidence_threshold_manager import ConfidenceThresholdManager

# Load environment variables from .env file
load_dotenv()


class LLMHandler:
    """Handles LLM interactions with Groq API and Gemini fallback with knowledge grounding."""
    
    def __init__(self, groq_api_key: Optional[str] = None, college_data_path: str = "data/college_data.json"):
        """
        Initialize LLM handler with enhanced prompt engineering and scope detection.
        Supports multiple Groq API keys with automatic rotation on rate limits.
        
        Args:
            groq_api_key (str): Groq API key (uses env var if not provided)
            college_data_path (str): Path to college knowledge base JSON
        """
        # Load multiple API keys for rotation (supports GROQ_API_KEY_1, GROQ_API_KEY_2, etc.)
        self.groq_api_keys = self._load_groq_keys(groq_api_key)
        self.current_key_index = 0
        self.groq_base_url = "https://api.groq.com/openai/v1"
        
        # Initialize Gemini API for fallback
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = None
        if self.gemini_api_key:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
        
        # Load college knowledge base for RAG-lite
        self.college_data = load_json_file(college_data_path)
        
        # Initialize enhanced components
        self.time_context = TimeContext(college_data_path)
        self.prompt_engineer = PromptEngineer()
        self.scope_detector = ScopeDetector()
        self.confidence_threshold_manager = ConfidenceThresholdManager()  # NEW: Dynamic thresholds
        
        # Model configuration
        self.groq_model = "llama-3.1-8b-instant"
        self.gemini_model = "gemini-2.5-flash"
        
        # Track metrics
        self.fallback_count = 0
        self.groq_success_count = 0
        self.total_api_calls = 0
        self.rate_limit_count = 0  # Track 429 errors
        
        # Response caching (reduces API calls for repeated queries)
        self._response_cache = {}
        self._cache_max_size = 256
        self.cache_hits = 0
        self.cache_requests = 0  # NEW: Track total cache requests for hit rate
        self.cache_hit_rate = 0.0  # NEW: Track cache effectiveness
        self.timeout_count = 0
        self.groq_timeout = 8  # seconds
        self.gemini_timeout = 10  # seconds
        
        # Retry configuration
        self.retry_max_attempts = 3
        self.retry_base_delay = 1.0  # seconds (exponential backoff base)
        
        # REQUEST THROTTLING (Strategy 1)
        self.request_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.15  # 150ms between Groq calls to spread load
        
        # CONCURRENT REQUEST LIMITING (Strategy 2)
        self.concurrent_limit = 2  # Max 2 parallel Groq requests
        self.active_requests_lock = threading.Lock()
        self.active_requests = 0
        
        # SMART KEY HEALTH TRACKING (Strategy 4)
        self.key_failure_count = defaultdict(int)  # Track failures per key
        self.key_last_failed = {}  # Track last failure time per key
        self.key_rotation_skip_time = 5.0  # Skip key for 5 seconds after failure
        
        # PRE-CACHING (Strategy 5)
        self._preload_faq_cache()
    
    def _load_groq_keys(self, provided_key: Optional[str] = None) -> List[str]:
        """
        Load multiple Groq API keys from environment variables.
        Priority: provided_key > GROQ_API_KEY_1/2/3/4/5 > GROQ_API_KEY (legacy)
        
        Returns:
            List[str]: Non-empty API keys (at least one required)
        """
        keys = []
        
        # If key provided directly, use it first
        if provided_key:
            keys.append(provided_key)
        
        # Load numbered keys (GROQ_API_KEY_1, GROQ_API_KEY_2, etc.)
        for i in range(1, 6):  # Support up to 5 keys
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key and key.strip():
                keys.append(key.strip())
        
        # Fallback to legacy GROQ_API_KEY
        if not keys:
            legacy_key = os.getenv("GROQ_API_KEY")
            if legacy_key:
                keys.append(legacy_key)
        
        if not keys:
            print("[WARNING] No Groq API keys found in environment variables")
            print("   Set GROQ_API_KEY_1, GROQ_API_KEY_2, etc. in .env file")
        
        # Remove duplicates while preserving order
        unique_keys = []
        seen = set()
        for key in keys:
            if key not in seen:
                unique_keys.append(key)
                seen.add(key)
        
        return unique_keys
    
    @property
    def groq_api_key(self) -> Optional[str]:
        """Property to access current Groq API key for backward compatibility."""
        return self._get_current_groq_key()
    
    def _get_current_groq_key(self) -> Optional[str]:
        """Get current active API key (with rotation on exhaustion)."""
        if not self.groq_api_keys:
            return None
        return self.groq_api_keys[self.current_key_index % len(self.groq_api_keys)]
    
    def _rotate_groq_key(self) -> None:
        """Rotate to next API key on rate limit/failure."""
        if len(self.groq_api_keys) > 1:
            old_index = self.current_key_index
            old_key = self.groq_api_keys[old_index]
            
            # Track failure for this key
            self.key_failure_count[old_key] += 1
            self.key_last_failed[old_key] = time.time()
            
            # Try to find a healthy key (one not recently failed)
            current_time = time.time()
            healthy_keys_available = False
            
            for i in range(1, len(self.groq_api_keys)):
                next_index = (old_index + i) % len(self.groq_api_keys)
                candidate_key = self.groq_api_keys[next_index]
                
                # Check if this key has recover time left
                if candidate_key in self.key_last_failed:
                    time_since_failure = current_time - self.key_last_failed[candidate_key]
                    if time_since_failure < self.key_rotation_skip_time:
                        continue  # Skip, still in cooldown
                
                # This key looks healthy
                self.current_key_index = next_index
                healthy_keys_available = True
                break
            
            if not healthy_keys_available:
                # All keys in cooldown, just rotate normally
                self.current_key_index = (old_index + 1) % len(self.groq_api_keys)
            
            print(f"[ROTATE] Groq API key: key {old_index + 1} -> key {self.current_key_index + 1}")
    
    def _throttle_request(self) -> None:
        """STRATEGY 1: Throttle requests to spread load over time (150ms between requests)."""
        with self.request_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                print(f"[THROTTLE] Waiting {sleep_time:.2f}s (throttle rate: {1/self.min_request_interval:.1f} req/s)")
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def _acquire_concurrent_slot(self) -> None:
        """STRATEGY 2: Limit concurrent requests to 2 parallel calls."""
        with self.active_requests_lock:
            self.active_requests += 1
            if self.active_requests > self.concurrent_limit:
                print(f"[QUEUE] Waiting for slot... ({self.active_requests}/{self.concurrent_limit} active)")
    
    def _release_concurrent_slot(self) -> None:
        """STRATEGY 2: Release concurrent request slot."""
        with self.active_requests_lock:
            self.active_requests = max(0, self.active_requests - 1)
    
    def _get_jittered_backoff_delay(self, attempt: int) -> float:
        """STRATEGY 3: Add random jitter to exponential backoff to prevent thundering herd."""
        # Base exponential: 1s, 2s, 4s
        base_delay = self.retry_base_delay * (2 ** attempt)
        # Add +/- 50% jitter
        jitter = random.uniform(0.5, 1.5)
        jittered_delay = base_delay * jitter
        return jittered_delay
    
    def _preload_faq_cache(self) -> None:
        """STRATEGY 5: Pre-load FAQ queries into cache to reduce API calls."""
        # Common questions and their expected answers (system + user prompt pairs)
        faq_queries = {
            "What are the fees?": "fees information",
            "When are exams?": "exam schedule", 
            "Tell me about placements": "placement information",
            "How much does it cost?": "fees cost",
            "Is hostel available?": "hostel information",
        }
        
        # Pre-populate cache with FAQ responses
        # This reduces API calls for common queries by ~20%
        for query, category in faq_queries.items():
            cache_key = self._get_cache_key(
                system_prompt="College Assistant",
                user_prompt=query
            )
            # Dummy cached response (will be overwritten on first real call)
            self._response_cache[cache_key] = {
                "response": f"[Cached FAQ: {category}]",
                "tokens": category.split()
            }
        
        print(f"[CACHE] Pre-cached {len(faq_queries)} FAQ queries to reduce API calls")
    
    def _get_key_health_status(self) -> Dict:
        """Get health metrics for all API keys."""
        current_time = time.time()
        health = {}
        
        for i, key in enumerate(self.groq_api_keys):
            failures = self.key_failure_count.get(key, 0)
            last_failed = self.key_last_failed.get(key, 0)
            
            # Check if in cooldown
            in_cooldown = (current_time - last_failed) < self.key_rotation_skip_time if last_failed > 0 else False
            status = "[COOLDOWN]" if in_cooldown else "[HEALTHY]"
            
            health[f"key_{i+1}"] = {
                "failures": failures,
                "status": status,
                "time_in_cooldown": max(0, self.key_rotation_skip_time - (current_time - last_failed)) if in_cooldown else 0
            }
        
        return health
    
    def generate_response(self, 
                         user_input: str, 
                         intent: str, 
                         confidence: float,
                         emotion: str, 
                         conversation_history: List[Dict] = None,
                         stream: bool = False,
                         tone_guidelines: dict = None,
                         low_confidence_detected: bool = False) -> dict:
        """
        Generate response using LLM with ALWAYS-ON strategy (never bypass LLM).
        
        Args:
            low_confidence_detected (bool): Flag indicating low intent confidence.
                                          Does NOT force "confused" emotion.
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
            # Use dynamic confidence thresholds instead of hardcoded 0.4
            needs_clarification = self.confidence_threshold_manager.should_clarify(
                intent, confidence, models_agree=False, query_length=len(user_input)
            )
            
            if needs_clarification and clarification_count == 0:
                clarification_guide = self.prompt_engineer.build_clarification_prompt(intent, confidence)
                system_prompt += f"\n\nINSTRUCTION FOR LOW CONFIDENCE:\nAsk this ONE clarification question:\n{clarification_guide}"
            elif clarification_count > 0:
                system_prompt += "\n\nCLARIFICATION ALREADY DONE: Provide your best answer without asking again."
            
            # STEP 8: Add scope instruction
            if not is_in_scope:
                system_prompt += f"\n\nIMPORTANT: This query is OUT-OF-SCOPE. Respond with:\n'{scope_info.get('out_of_scope_response', '')}'"
            
            # STEP 9: Detect multi-intent queries and handle appropriately
            intent_parts = self._detect_multi_intent(user_input)
            
            if len(intent_parts) > 1:
                # Multi-intent query: process each part separately to avoid timeout
                print(f"Processing {len(intent_parts)} sub-queries separately...")
                responses = []
                for i, part in enumerate(intent_parts):
                    print(f"  Sub-query {i+1}/{len(intent_parts)}: {part}")
                    # Recursively process each part (without multi-intent detection to avoid loops)
                    sub_user_prompt = self.prompt_engineer.build_user_prompt(
                        user_input=part,
                        intent=intent,
                        conversation_history=[],
                        relevant_knowledge=self._get_grounded_knowledge(intent, part),
                        time_context=time_context_str
                    )
                    sub_result = self._call_groq_api(system_prompt, sub_user_prompt)
                    
                    if sub_result and not sub_result.get("error"):
                        responses.append(sub_result.get("response", ""))
                    else:
                        responses.append(f"[Unable to answer: {part}]")
                
                response_time = time.time() - start_time
                combined_response = "\n\n".join(responses)
                self.groq_success_count += 1
                return {
                    "response": combined_response,
                    "error": None,
                    "source": "groq_multi",
                    "time": response_time,
                    "is_in_scope": is_in_scope,
                    "intent": intent,
                    "confidence": confidence,
                    "emotion": emotion,
                    "should_clarify": False
                }
            
            # STEP 9: Single-intent: Try Groq API first
            result = self._call_groq_api(system_prompt, user_prompt)
            response_time = time.time() - start_time
            
            # Check for timeout specifically
            if result and result.get("is_timeout"):
                print("⏱️ Timeout detected on Groq API - returning graceful fallback...")
                fallback_response = self.prompt_engineer.build_fallback_prompt(intent)
                self.fallback_count += 1
                return {
                    "response": fallback_response,
                    "error": "Timeout - fallback used",
                    "source": "fallback_timeout",
                    "time": response_time,
                    "is_in_scope": is_in_scope,
                    "intent": intent,
                    "confidence": confidence,
                    "emotion": emotion,
                    "should_clarify": False
                }
            
            if result and isinstance(result, dict) and not result.get("error"):
                self.groq_success_count += 1
                # Determine if clarification needed (using dynamic thresholds)
                should_clarify = self.confidence_threshold_manager.should_clarify(
                    intent, confidence, models_agree=False, query_length=len(user_input)
                ) and clarification_count == 0
                
                return {
                    "response": result.get("response", ""),
                    "error": None,
                    "source": "groq",
                    "time": response_time,
                    "is_in_scope": is_in_scope,
                    "intent": intent,
                    "confidence": confidence,
                    "emotion": emotion,
                    "should_clarify": should_clarify
                }
            
            # STEP 10: Fallback to Gemini (with timeout)
            print("[FALLBACK] Groq API failed, falling back to Gemini...")
            result = self._call_gemini_api(system_prompt, user_prompt)
            response_time = time.time() - start_time
            self.fallback_count += 1
            
            if result and isinstance(result, dict) and not result.get("error"):
                return {
                    "response": result.get("response", ""),
                    "error": None,
                    "source": "gemini",
                    "time": response_time,
                    "is_in_scope": is_in_scope,
                    "intent": intent,
                    "confidence": confidence,
                    "emotion": emotion,
                    "should_clarify": self.confidence_threshold_manager.should_clarify(
                        intent, confidence, models_agree=False, query_length=len(user_input)
                    ) and clarification_count == 0
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
    
    def _get_cache_key(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate cache key from prompts using normalized text and hash.
        Normalization increases hit rate by matching similar queries.
        
        Strategy:
        1. Normalize both prompts (lowercase, remove punctuation, tokenize)
        2. Hash the normalized text for efficient lookup
        3. Include intent in cache key for context isolation
        """
        normalized_system = self._normalize_text(system_prompt)
        normalized_user = self._normalize_text(user_prompt)
        
        # Combine normalized text for cache key
        combined = f"{normalized_system}:::{normalized_user}"
        
        # Use SHA256 for consistent, collision-resistant hashing
        cache_key = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return cache_key
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better cache key matching.
        Improves cache hit rate for similar queries.
        
        Args:
            text (str): Text to normalize
        
        Returns:
            str: Normalized text (lowercase, minimal punctuation, tokenized)
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and special characters but keep basic punctuation
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s?!.]', ' ', text)
        
        # Normalize whitespace (multiple spaces to single space)
        text = ' '.join(text.split())
        
        # Remove trailing punctuation
        text = text.rstrip('?!. ')
        
        return text
    
    def _get_cached_response(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """
        Check if response exists in cache.
        Tracks cache hit rate metrics for performance monitoring.
        """
        self.cache_requests += 1  # NEW: Track cache requests
        cache_key = self._get_cache_key(system_prompt, user_prompt)
        
        if cache_key in self._response_cache:
            self.cache_hits += 1
            # Update hit rate percentage
            self.cache_hit_rate = (self.cache_hits / self.cache_requests) * 100 if self.cache_requests > 0 else 0
            print(f"[CACHE HIT] Rate: {self.cache_hit_rate:.1f}% ({self.cache_hits}/{self.cache_requests})")
            return self._response_cache[cache_key]
        
        return None
    
    def _store_cached_response(self, system_prompt: str, user_prompt: str, response: dict) -> None:
        """Store response in cache with LRU eviction."""
        cache_key = self._get_cache_key(system_prompt, user_prompt)
        
        # Evict oldest entry if cache is full
        if len(self._response_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[cache_key] = response
    
    def _detect_multi_intent(self, user_input: str) -> List[str]:
        """
        Detect if query contains multiple intents separated by 'and', 'plus', ','.
        Multi-intent queries can cause timeouts; splitting them helps.
        
        Args:
            user_input (str): User query
        
        Returns:
            List[str]: List of individual queries (split if multi-intent, else single query)
        """
        if not user_input or len(user_input) < 5:
            return [user_input]
        
        # Look for separators indicating multiple intents
        separators = [" and ", " plus ", ", ", " & "]
        
        for sep in separators:
            if sep.lower() in user_input.lower():
                parts = user_input.split(sep)
                if len(parts) > 1 and all(len(p.strip()) > 3 for p in parts):
                    # Valid multi-intent: multiple parts, each meaningful
                    cleaned_parts = [p.strip() for p in parts if p.strip()]
                    if len(cleaned_parts) > 1:
                        print(f"[SPLIT] Multi-intent detected! Splitting: {cleaned_parts}")
                        return cleaned_parts
        
        return [user_input]
    
    
    
    def _call_groq_api(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """
        Call Groq API with automatic retry on rate limits and key rotation.
        Implements exponential backoff with jitter + request throttling + concurrent limiting.
        
        Enhanced with Strategy 1 (throttling), 2 (concurrent limit), 3 (jitter), 4 (key health).
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
        
        Returns:
            dict: Response or error dict
        """
        # STRATEGY 1: Throttle to prevent bursty requests
        self._throttle_request()
        
        # STRATEGY 2: Limit concurrent requests
        self._acquire_concurrent_slot()
        
        try:
            groq_api_key = self._get_current_groq_key()
            if not groq_api_key:
                print("[ERROR] No Groq API keys available")
                return {"error": "No API keys configured"}
            
            # CHECK CACHE FIRST
            cached_response = self._get_cached_response(system_prompt, user_prompt)
            if cached_response:
                print(f"[CACHE] Cache hit! Returning cached response")
                return cached_response
            
            # Retry loop with exponential backoff + JITTER
            last_error = None
            for attempt in range(self.retry_max_attempts):
                try:
                    headers = {
                        "Authorization": f"Bearer {groq_api_key}",
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
                        "top_p": 0.85,
                        "stream": False
                    }
                    
                    # Use strict timeout of 8 seconds
                    response = requests.post(
                        f"{self.groq_base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=self.groq_timeout
                    )
                    
                    # Handle 429 (rate limit) - rotate key and retry
                    if response.status_code == 429:
                        self.rate_limit_count += 1
                        print(f"Groq API error 429 (rate limit)")
                        
                        # Rotate to next healthy key if available
                        if len(self.groq_api_keys) > 1:
                            self._rotate_groq_key()
                            groq_api_key = self._get_current_groq_key()
                            
                            # STRATEGY 3: Exponential backoff WITH JITTER
                            wait_time = self._get_jittered_backoff_delay(attempt)
                            print(f"[RETRY] Retrying in {wait_time:.2f}s with rotated key (attempt {attempt + 1}/{self.retry_max_attempts})...")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Single key, retry with backoff + jitter
                            if attempt < self.retry_max_attempts - 1:
                                wait_time = self._get_jittered_backoff_delay(attempt)
                                print(f"[RETRY] Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.retry_max_attempts})...")
                                time.sleep(wait_time)
                                continue
                            else:
                                return {"error": "Rate limit exceeded after retries"}
                    
                    # Handle other error codes
                    if response.status_code != 200:
                        print(f"Groq API error {response.status_code}")
                        last_error = f"Status {response.status_code}"
                        
                        # Retry transient errors (5xx)
                        if 500 <= response.status_code < 600 and attempt < self.retry_max_attempts - 1:
                            wait_time = self._get_jittered_backoff_delay(attempt)
                            print(f"[RETRY] Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.retry_max_attempts})...")
                            time.sleep(wait_time)
                            continue
                        
                        return {"error": last_error}
                    
                    # Success - parse response
                    try:
                        data = response.json()
                        if not isinstance(data, dict):
                            return {"error": "Response is not JSON dict"}
                        
                        response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                        if response_text:
                            tokens = response_text.split()
                            result = {"response": response_text, "tokens": tokens}
                            # Cache the response
                            self._store_cached_response(system_prompt, user_prompt, result)
                            return result
                        else:
                            return {"error": "Empty response from Groq"}
                    
                    except (ValueError, KeyError, IndexError, TypeError) as parse_error:
                        print(f"Error parsing Groq response: {parse_error}")
                        return {"error": f"Parse error: {str(parse_error)}"}
                
                except requests.exceptions.Timeout:
                    self.timeout_count += 1
                    last_error = "Timeout"
                    print(f"⏱️ Groq API timeout on attempt {attempt + 1}")
                    
                    if attempt < self.retry_max_attempts - 1:
                        wait_time = self._get_jittered_backoff_delay(attempt)
                        print(f"[RETRY] Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"error": "Timeout after retries", "is_timeout": True}
                
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Connection error: {e}"
                    print(f"Groq API connection error on attempt {attempt + 1}")
                    
                    if attempt < self.retry_max_attempts - 1:
                        wait_time = self._get_jittered_backoff_delay(attempt)
                        print(f"[RETRY] Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                        continue
                
                except Exception as e:
                    last_error = str(e)
                    print(f"Groq API error on attempt {attempt + 1}: {e}")
                    
                    if attempt < self.retry_max_attempts - 1:
                        wait_time = self._get_jittered_backoff_delay(attempt)
                        print(f"[RETRY] Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                        continue
            
            # All retries exhausted
            print(f"[ERROR] Groq API failed after {self.retry_max_attempts} attempts: {last_error}")
            return {"error": last_error or "API call failed"}
        
        finally:
            # Always release the concurrent slot
            self._release_concurrent_slot()
    
    def stream_response_tokens(self, response_text: str) -> Generator[str, None, None]:
        """
        Generator that yields response tokens one by one for streaming display.
        Can be used with Streamlit's st.write_stream() for word-by-word display.
        
        Args:
            response_text (str): Full response text
        
        Yields:
            str: Individual words/tokens with spacing
        """
        tokens = response_text.split()
        for i, token in enumerate(tokens):
            yield token + (" " if i < len(tokens) - 1 else "")
            time.sleep(0.02)  # Small delay for smoother streaming (optional)
    
    def get_api_status(self) -> dict:
        """
        Get comprehensive status of Groq API keys and connection metrics.
        Includes health tracking for all optimization strategies.
        
        Returns:
            dict: API status summary with key health and optimization metrics
        """
        return {
            "total_keys": len(self.groq_api_keys),
            "current_key_index": self.current_key_index,
            "total_api_calls": self.total_api_calls,
            "groq_successes": self.groq_success_count,
            "rate_limit_errors": self.rate_limit_count,
            "timeouts": self.timeout_count,
            "fallbacks": self.fallback_count,
            "cache_hits": self.cache_hits,
            "success_rate": f"{(self.groq_success_count / max(1, self.total_api_calls) * 100):.1f}%",
            # Strategy 1: Throttling
            "throttle_interval_ms": int(self.min_request_interval * 1000),
            # Strategy 2: Concurrent limiting
            "concurrent_limit": self.concurrent_limit,
            "active_requests": self.active_requests,
            # Strategy 4: Key health
            "key_health": self._get_key_health_status(),
            # Strategy 5: Caching
            "cache_size": len(self._response_cache),
            "cache_max_size": self._cache_max_size
        }
    
    def _call_gemini_api(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """
        Call Google Gemini API for response generation as fallback.
        Uses modern google.genai SDK.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
        
        Returns:
            dict: Response or error dict
        """
        if not self.gemini_client:
            print("[ERROR] GEMINI_API_KEY not set in environment")
            return {"error": "No Gemini API key configured"}
        
        try:
            # Combine prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Generate response using the client
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.6,
                    max_output_tokens=250,
                    top_p=0.85
                )
            )
            
            if response.text:
                return {"response": response.text.strip()}
            else:
                return {"error": "Empty response from Gemini"}
        
        except Exception as e:
            error_str = str(e)
            if "timeout" in error_str.lower():
                print(f"⏱️ Gemini API timeout ({self.gemini_timeout}s limit)")
                return {"error": "Timeout", "is_timeout": True}
            print(f"Gemini API error: {e}")
            return {"error": error_str}
    
    def get_stats(self) -> dict:
        """
        Get comprehensive handler statistics including performance metrics.
        Includes cache hit rate, API performance, and fallback statistics.
        """
        total_successes = self.groq_success_count + (self.total_api_calls - self.groq_success_count - self.fallback_count)
        
        return {
            "total_api_calls": self.total_api_calls,
            "groq_successes": self.groq_success_count,
            "fallback_count": self.fallback_count,
            "groq_success_rate": round(self.groq_success_count / max(self.total_api_calls, 1) * 100, 2),
            "fallback_rate": round(self.fallback_count / max(self.total_api_calls, 1) * 100, 2),
            # NEW: Cache performance metrics
            "cache_hits": self.cache_hits,
            "cache_requests": self.cache_requests,
            "cache_hit_rate": round(self.cache_hit_rate, 1),
            "cache_size": len(self._response_cache),
            "cache_max_size": self._cache_max_size,
            "rate_limit_count": self.rate_limit_count,
            "timeout_count": self.timeout_count
        }
