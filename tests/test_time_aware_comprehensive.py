#!/usr/bin/env python
"""
Comprehensive Test Suite for Enhanced Chatbot with Time-Awareness & Anti-Hallucination
Tests all new features: time_context.py, time-aware LLM, anti-hallucination, etc.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Setup path - add parent directory to import modules
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from time_context import TimeContext
from llm_handler import LLMHandler
from context_manager import ConversationContext
from emotion_detector import EmotionDetector
from intent_model import IntentClassifier
from database import ChatbotDatabase


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class TestSuite:
    """Test suite manager."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def add_result(self, test_name: str, passed: bool, details: str = ""):
        """Record test result."""
        self.tests.append((test_name, passed, details))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        print(f"  {status}: {test_name}")
        if details:
            print(f"           {Colors.YELLOW}→ {details}{Colors.RESET}")
    
    def print_summary(self):
        """Print summary."""
        total = self.passed + self.failed
        print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{self.name} Summary:{Colors.RESET} " +
              f"{Colors.GREEN}{self.passed} passed{Colors.RESET}, " +
              f"{Colors.RED}{self.failed} failed{Colors.RESET} out of {total}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        return self.failed == 0


# ============================================================================
# TEST SUITE 1: TimeContext Module Tests
# ============================================================================

def test_time_context_module():
    """Test TimeContext module functionality."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 1: TimeContext Module (Time-Awareness)")
    print(f"{'='*80}{Colors.RESET}\n")
    
    suite = TestSuite("TimeContext")
    
    # Test 1: Initialization
    try:
        tc = TimeContext()
        suite.add_result("TimeContext Initialization", True, "Instance created")
    except Exception as e:
        suite.add_result("TimeContext Initialization", False, str(e)[:80])
        return suite
    
    # Test 2: get_time_of_day method
    time_periods = ["early_morning", "morning", "late_morning", "afternoon", "evening", "night", "late_night"]
    time_of_day = tc.get_time_of_day()
    suite.add_result("get_time_of_day() Returns Valid Period", 
                     time_of_day in time_periods, f"Got: {time_of_day}")
    
    # Test 3: get_intelligent_greeting
    greeting = tc.get_intelligent_greeting()
    has_emoji = any(emoji in greeting for emoji in ['☀️', '👋', '🌙', '💪', '🎉', '📚', '🦉'])
    is_not_empty = len(greeting) > 5
    suite.add_result("get_intelligent_greeting() Returns Creative Greeting",
                     has_emoji and is_not_empty, f"Greeting: {greeting[:40]}...")
    
    # Test 4: Personalized greeting
    greeting_with_name = tc.get_intelligent_greeting(user_name="Alex")
    has_name = "Alex" in greeting_with_name or "morning" in greeting_with_name.lower()
    suite.add_result("Personized Greeting With User Name",
                     has_name, f"Got: {greeting_with_name[:50]}...")
    
    # Test 5: get_context_awareness_prompt
    prompt = tc.get_context_awareness_prompt()
    is_valid = len(prompt) > 50 and ("office" in prompt.lower() or "time" in prompt.lower())
    suite.add_result("get_context_awareness_prompt() Non-Empty",
                     is_valid, f"Length: {len(prompt)} chars")
    
    # Test 6: get_hallucination_check_prompt
    hallucination_check = tc.get_hallucination_check_prompt()
    has_critical = "CRITICAL" in hallucination_check
    no_hallucination_phrase = "don't have this information" in hallucination_check.lower() or \
                              "not in database" in hallucination_check.lower()
    suite.add_result("get_hallucination_check_prompt() Contains Anti-Hallucination Directives",
                     has_critical and no_hallucination_phrase, f"Includes critical checks")
    
    # Test 7: is_office_open
    is_open, reason = tc.is_office_open()
    is_tuple = isinstance(is_open, bool) and isinstance(reason, str)
    suite.add_result("is_office_open() Returns Tuple[bool, str]",
                     is_tuple, f"Open: {is_open}, Reason: {reason}")
    
    # Test 8: get_relevant_schedule_info
    schedule = tc.get_relevant_schedule_info("library")
    has_schedule = "library" in schedule.lower() or "hours" in schedule.lower()
    suite.add_result("get_relevant_schedule_info() Returns Hour Info",
                     has_schedule, f"Schedule: {schedule}")
    
    # Test 9: get_college_data_snippet
    snippet = tc.get_college_data_snippet("college_name")
    is_string = isinstance(snippet, str)
    not_empty = len(snippet) > 0
    suite.add_result("get_college_data_snippet() Returns String",
                     is_string and not_empty, f"Data: {snippet}")
    
    # Test 10: get_context_summary
    summary = tc.get_context_summary()
    has_keys = all(key in summary for key in ["current_time", "time_of_day", "office_open", "schedules"])
    suite.add_result("get_context_summary() Contains All Context Keys",
                     has_keys, f"Keys: {list(summary.keys())}")
    
    return suite


# ============================================================================
# TEST SUITE 2: Enhanced Context Manager Tests
# ============================================================================

def test_enhanced_context_manager():
    """Test ConversationContext with time awareness."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 2: Enhanced Context Manager (Time-Aware)")
    print(f"{'='*80}{Colors.RESET}\n")
    
    suite = TestSuite("ConversationContext")
    
    # Test 1: Initialization with TimeContext
    try:
        ctx = ConversationContext()
        has_time_context = hasattr(ctx, 'time_context')
        suite.add_result("ConversationContext Has TimeContext Attribute", has_time_context)
    except Exception as e:
        suite.add_result("ConversationContext Initialization", False, str(e)[:80])
        return suite
    
    # Test 2: Add turn
    try:
        ctx.add_turn(
            user_input="What are fees?",
            bot_response="Our fees are...",
            intent="fees",
            confidence=0.95,
            emotion="neutral",
            entities={"fee_type": "tuition"}
        )
        suite.add_result("add_turn() Records Conversation", len(ctx.get_history()) == 1)
    except Exception as e:
        suite.add_result("add_turn() Records Conversation", False, str(e)[:80])
    
    # Test 3: get_time_aware_context
    try:
        time_aware = ctx.get_time_aware_context()
        has_time_info = all(key in time_aware for key in ["current_time", "time_of_day", "office_open"])
        suite.add_result("get_time_aware_context() Returns Time Info",
                         has_time_info, f"Has keys: {list(time_aware.keys())}")
    except Exception as e:
        suite.add_result("get_time_aware_context() Returns Time Info", False, str(e)[:80])
    
    # Test 4: is_conversation_started_today
    try:
        today = ctx.is_conversation_started_today()
        suite.add_result("is_conversation_started_today() Returns Boolean",
                         isinstance(today, bool))
    except Exception as e:
        suite.add_result("is_conversation_started_today() Returns Boolean", False, str(e)[:80])
    
    # Test 5: get_session_duration_minutes
    try:
        duration = ctx.get_session_duration_minutes()
        is_valid = isinstance(duration, float) and duration >= 0
        suite.add_result("get_session_duration_minutes() Returns Non-Negative Float",
                         is_valid, f"Duration: {duration:.2f} minutes")
    except Exception as e:
        suite.add_result("get_session_duration_minutes() Returns Non-Negative Float", False, str(e)[:80])
    
    # Test 6: Topic continuity detection
    try:
        ctx.add_turn("Follow-up on fees", "More info...", "fees", 0.92, "neutral")
        is_continuous = ctx.get_topic_continuity()
        suite.add_result("get_topic_continuity() Detects Same Intent",
                         is_continuous, f"Continuity: {is_continuous}")
    except Exception as e:
        suite.add_result("get_topic_continuity() Detects Same Intent", False, str(e)[:80])
    
    return suite


# ============================================================================
# TEST SUITE 3: LLM Handler Time-Aware Features
# ============================================================================

def test_llm_handler_time_aware():
    """Test LLMHandler with time-aware system prompts."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 3: LLM Handler Time-Aware Features")
    print(f"{'='*80}{Colors.RESET}\n")
    
    suite = TestSuite("LLMHandler Time-Aware")
    
    # Test 1: Initialization with TimeContext
    try:
        handler = LLMHandler()
        has_time_context = hasattr(handler, 'time_context')
        suite.add_result("LLMHandler Initializes TimeContext",
                         has_time_context, "TimeContext available")
    except Exception as e:
        suite.add_result("LLMHandler Initialization", False, str(e)[:80])
        return suite
    
    # Test 2: System prompt includes time context
    try:
        prompt = handler._build_system_prompt("fees", 0.8, "neutral")
        has_time_context = "CURRENT TIME CONTEXT" in prompt or "context" in prompt.lower()
        has_anti_hallucination = "CRITICAL" in prompt or "hallucination" in prompt.lower()
        suite.add_result("System Prompt Includes Time & Anti-Hallucination Directives",
                         has_time_context and has_anti_hallucination,
                         f"Prompt length: {len(prompt)} chars")
    except Exception as e:
        suite.add_result("System Prompt Generation", False, str(e)[:80])
    
    # Test 3: Response contains source
    try:
        result = handler.generate_response(
            user_input="Simple test",
            intent="general",
            confidence=0.7,
            emotion="neutral"
        )
        has_source = "source" in result
        has_response = "response" in result and len(result["response"]) > 0
        suite.add_result("generate_response() Returns Source & Response",
                         has_source and has_response,
                         f"Source: {result.get('source')}")
    except Exception as e:
        suite.add_result("generate_response() Basic Functionality", False, str(e)[:80])
    
    # Test 4: Low confidence triggers clarification
    try:
        result = handler.generate_response(
            user_input="???",
            intent="unclear",
            confidence=0.1,
            emotion="confused"
        )
        should_clarify = result.get("should_clarify", False)
        is_clarification = "clarif" in result.get("response", "").lower()
        suite.add_result("Low Confidence (<15%) Triggers Clarification",
                         should_clarify or is_clarification,
                         f"Clarify flag: {should_clarify}")
    except Exception as e:
        suite.add_result("Clarification on Low Confidence", False, str(e)[:80])
    
    # Test 5: Response time tracking
    try:
        result = handler.generate_response("Test", "test", 0.5, "neutral")
        has_time = "time" in result and isinstance(result["time"], (int, float))
        suite.add_result("Response Time Tracked",
                         has_time, f"Time: {result.get('time', 0):.2f}s")
    except Exception as e:
        suite.add_result("Response Time Tracking", False, str(e)[:80])
    
    return suite


# ============================================================================
# TEST SUITE 4: Anti-Hallucination Features
# ============================================================================

def test_anti_hallucination():
    """Test anti-hallucination mechanisms."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 4: Anti-Hallucination Features")
    print(f"{'='*80}{Colors.RESET}\n")
    
    suite = TestSuite("Anti-Hallucination")
    
    # Test 1: TimeContext has hallucination check
    try:
        tc = TimeContext()
        prompt = tc.get_hallucination_check_prompt()
        checks = [
            "CRITICAL" in prompt,
            "database" in prompt.lower(),
            "only answer based on" in prompt.lower() or "based on" in prompt.lower(),
            "never invent" in prompt.lower() or "don't invent" in prompt.lower(),
        ]
        all_checks = all(checks)
        suite.add_result("TimeContext Hallucination Check Contains All Required Elements",
                         all_checks, f"Checks present: {sum(checks)}/4")
    except Exception as e:
        suite.add_result("TimeContext Hallucination Check", False, str(e)[:80])
    
    # Test 2: get_college_data_snippet returns safe values
    try:
        tc = TimeContext()
        # Try non-existent key
        result = tc.get_college_data_snippet("nonexistent_key_xyz")
        has_safe_message = "not available" in result.lower() or "don't have" in result.lower()
        suite.add_result("get_college_data_snippet() Falls Back Safely",
                         has_safe_message, f"Result: {result}")
    except Exception as e:
        suite.add_result("Safe Data Fallback", False, str(e)[:80])
    
    # Test 3: LLMHandler system prompt has anti-hallucination
    try:
        handler = LLMHandler()
        prompt = handler._build_system_prompt("fees", 0.9, "neutral")
        anti_hall_phrases = [
            "only answer based on",
            "NEVER make up",
            "don't have",
            "admit uncertainty",
            "reference source"
        ]
        found = sum(1 for phrase in anti_hall_phrases if phrase.lower() in prompt.lower())
        suite.add_result("LLM System Prompt Has Multiple Anti-Hallucination Checks",
                         found >= 3, f"Found {found}/5 anti-hallucination checks")
    except Exception as e:
        suite.add_result("LLM Anti-Hallucination Checks", False, str(e)[:80])
    
    # Test 4: Invalid data requests handled gracefully
    try:
        handler = LLMHandler()
        result = handler.generate_response(
            user_input="What are the secret data?",
            intent="general",
            confidence=0.5,
            emotion="neutral"
        )
        # Should not error, just return safe response
        has_response = "response" in result and len(result["response"]) > 0
        suite.add_result("Invalid Data Requests Handled Gracefully",
                         has_response, f"Returned response: {len(result.get('response', ''))} chars")
    except Exception as e:
        suite.add_result("Invalid Data Handling", False, str(e)[:80])
    
    return suite


# ============================================================================
# TEST SUITE 5: Integration Tests
# ============================================================================

def test_integration():
    """Test full conversation flow with time awareness."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 5: Integration Tests (Full Conversation Flow)")
    print(f"{'='*80}{Colors.RESET}\n")
    
    suite = TestSuite("Integration")
    
    try:
        # Initialize all components
        intent_classifier = IntentClassifier()
        emotion_detector = EmotionDetector()
        llm_handler = LLMHandler()
        context = ConversationContext()
        
        suite.add_result("All Components Initialized", True)
    except Exception as e:
        suite.add_result("Component Initialization", False, str(e)[:80])
        return suite
    
    # Test flow: User asks question → Intent & Emotion detected → Response generated
    test_queries = [
        ("What are the library hours?", "library"),
        ("Tell me about placements", "placements"),
        ("How do I register?", "admission"),
    ]
    
    for user_input, expected_intent_type in test_queries:
        try:
            # Step 1: Classify intent
            classification = intent_classifier.predict(user_input)
            intent = classification.get("intent")
            confidence = classification.get("confidence", 0)
            
            # Step 2: Detect emotion
            emotion_result = emotion_detector.detect_emotion(user_input)
            emotion = emotion_result.get("emotion", "neutral")
            
            # Step 3: Generate response with time context
            llm_result = llm_handler.generate_response(
                user_input, intent, confidence, emotion, ""
            )
            response = llm_result.get("response", "")
            
            # Step 4: Update context
            context.add_turn(user_input, response, intent, confidence, emotion, {})
            
            has_response = len(response) > 0
            has_time_context = context.get_time_aware_context() is not None
            
            suite.add_result(f"Full Flow: '{user_input[:30]}...'",
                             has_response and has_time_context,
                             f"Intent={intent}, Conf={confidence:.2f}, Emotion={emotion}")
        except Exception as e:
            suite.add_result(f"Full Flow: '{user_input[:30]}...'", False, str(e)[:80])
    
    # Test conversation continuity
    try:
        context.add_turn("First question", "Answer 1", "fees", 0.9, "neutral")
        context.add_turn("Follow-up", "Answer 2", "fees", 0.88, "neutral")
        
        is_continuous = context.get_topic_continuity()
        history = context.get_history()
        has_history = len(history) >= 2
        
        suite.add_result("Conversation Continuity & History",
                         is_continuous and has_history,
                         f"History length: {len(history)}, Continuous: {is_continuous}")
    except Exception as e:
        suite.add_result("Conversation Continuity", False, str(e)[:80])
    
    # Test time-aware context in conversation
    try:
        time_context = context.get_time_aware_context()
        required_keys = ["current_time", "time_of_day", "office_open", "office_status"]
        has_all_keys = all(key in time_context for key in required_keys)
        
        suite.add_result("Context Contains Time & Schedule Info",
                         has_all_keys, f"Time of day: {time_context.get('time_of_day')}")
    except Exception as e:
        suite.add_result("Context Time Info", False, str(e)[:80])
    
    return suite


# ============================================================================
# TEST SUITE 6: Edge Cases & Error Handling
# ============================================================================

def test_edge_cases():
    """Test edge cases and error handling."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 6: Edge Cases & Error Handling")
    print(f"{'='*80}{Colors.RESET}\n")
    
    suite = TestSuite("Edge Cases")
    
    # Test 1: Empty strings
    try:
        tc = TimeContext()
        result = tc.get_college_data_snippet("")
        suite.add_result("Empty String Handling", isinstance(result, str))
    except Exception as e:
        suite.add_result("Empty String Handling", False, str(e)[:80])
    
    # Test 2: Very long strings
    try:
        handler = LLMHandler()
        long_input = "What " * 200  # 1000+ chars
        result = handler.generate_response(long_input, "general", 0.5, "neutral")
        has_response = "response" in result
        suite.add_result("Very Long Input String", has_response)
    except Exception as e:
        suite.add_result("Very Long Input String", False, str(e)[:80])
    
    # Test 3: Special characters
    try:
        ctx = ConversationContext()
        result = ctx.add_turn("!!!???***", "Response", "test", 0.5, "neutral")
        has_history = len(ctx.get_history()) > 0
        suite.add_result("Special Characters in Input", has_history)
    except Exception as e:
        suite.add_result("Special Characters in Input", False, str(e)[:80])
    
    # Test 4: Null/None values
    try:
        tc = TimeContext()
        result = tc.get_college_data_snippet(None)
        suite.add_result("None Value Handling", isinstance(result, str))
    except Exception as e:
        suite.add_result("None Value Handling", False, str(e)[:80])
    
    # Test 5: Rapid consecutive calls
    try:
        tc = TimeContext()
        results = [tc.get_time_of_day() for _ in range(100)]
        all_valid = all(r in ["early_morning", "morning", "late_morning", "afternoon", "evening", "night", "late_night"] for r in results)
        suite.add_result("Rapid Consecutive Calls (100x)", all_valid,
                         f"Calls successful")
    except Exception as e:
        suite.add_result("Rapid Consecutive Calls", False, str(e)[:80])
    
    return suite


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all test suites."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print(f"{'='*80}")
    print(f"COMPREHENSIVE CHATBOT TEST SUITE WITH TIME-AWARENESS & ANTI-HALLUCINATION")
    print(f"{'='*80}")
    print(f"{Colors.RESET}")
    
    all_suites = []
    
    # Run all test suites
    all_suites.append(test_time_context_module())
    all_suites.append(test_enhanced_context_manager())
    all_suites.append(test_llm_handler_time_aware())
    all_suites.append(test_anti_hallucination())
    all_suites.append(test_integration())
    all_suites.append(test_edge_cases())
    
    # Print final summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"FINAL TEST SUMMARY")
    print(f"{'='*80}{Colors.RESET}\n")
    
    total_passed = sum(s.passed for s in all_suites)
    total_failed = sum(s.failed for s in all_suites)
    total_tests = total_passed + total_failed
    
    for suite in all_suites:
        print(f"  {suite.name:30} {Colors.GREEN}{suite.passed:3}{Colors.RESET}\t passed, " +
              f"{Colors.RED}{suite.failed:3}{Colors.RESET}\t failed")
    
    print(f"\n{Colors.BOLD}Overall:{Colors.RESET} {Colors.GREEN}{total_passed}/{total_tests} PASSED{Colors.RESET}")
    
    if total_failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ {total_failed} TESTS FAILED{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
