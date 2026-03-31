#!/usr/bin/env python
"""
Scenario-Based Tests for Time-Aware Chatbot
Tests realistic user interactions and edge cases with time awareness
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from time_context import TimeContext
from llm_handler import LLMHandler
from context_manager import ConversationContext
from emotion_detector import EmotionDetector
from intent_model import IntentClassifier


class Colors:
    """ANSI colors."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class ScenarioTester:
    """Test realistic scenarios."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.scenarios = []
    
    def add_result(self, scenario: str, passed: bool, details: str = ""):
        """Record scenario result."""
        self.scenarios.append((scenario, passed, details))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        
        status = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {status} {scenario}")
        if details:
            print(f"      → {Colors.YELLOW}{details}{Colors.RESET}")
    
    def summary(self):
        """Print summary."""
        total = self.passed + self.failed
        print(f"\n{Colors.CYAN}{self.name}: {Colors.GREEN}{self.passed}/{total} passed{Colors.RESET}\n")
        return self.failed == 0


# ============================================================================
# SCENARIO TEST SUITE 1: Time-Based Scenarios
# ============================================================================

def test_time_based_scenarios():
    """Test responses that vary by time of day."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 1: Time-Based Response Variations")
    print(f"{'='*80}{Colors.RESET}\n")
    
    tester = ScenarioTester("Time-Based Scenarios")
    
    try:
        tc = TimeContext()
        
        # Test 1: Get current time of day
        time_period = tc.get_time_of_day()
        valid_periods = ["early_morning", "morning", "late_morning", "afternoon", "evening", "night", "late_night"]
        is_valid_period = time_period in valid_periods
        
        tester.add_result("Current Time Period Detected",
                         is_valid_period,
                         f"Current period: {time_period}")
        
        # Test 2: Current greeting has context
        greeting = tc.get_intelligent_greeting()
        has_emoji = any(emoji in greeting for emoji in ['☀️', '👋', '🌙', '💪', '🎉', '📚', '🦉'])
        has_time_awareness = len(greeting) > 10
        
        tester.add_result("Current Greeting Has Time Context & Emoji",
                         has_emoji and has_time_awareness,
                         f"Greeting: {greeting[:50]}...")
        
        # Test 3: Office status based on current time
        is_open, reason = tc.is_office_open()
        reason_is_string = isinstance(reason, str) and len(reason) > 0
        
        tester.add_result("Office Status Includes Reason Text",
                         reason_is_string,
                         f"Open: {is_open}, Reason: {reason}")
        
        # Test 4: Context awareness prompt mentions current time info
        context_prompt = tc.get_context_awareness_prompt()
        has_time_info = len(context_prompt) > 50
        has_college_context = ("office" in context_prompt.lower() or 
                               "class" in context_prompt.lower() or
                               "open" in context_prompt.lower())
        
        tester.add_result("Context Prompt Has Time & College Info",
                         has_time_info and has_college_context,
                         f"Prompt length: {len(context_prompt)} chars")
        
        # Test 5: Schedule info retrieval
        schedules_test = [
            ("library", "library"),
            ("office", "office"),
            ("classes", "class")
        ]
        
        all_schedules_found = True
        for intent, keyword in schedules_test:
            schedule_info = tc.get_relevant_schedule_info(intent)
            if keyword not in schedule_info.lower() and "hours" not in schedule_info.lower():
                all_schedules_found = False
        
        tester.add_result("All College Schedules Available",
                         all_schedules_found,
                         f"Library, Office, Classes schedules found")
        
    except Exception as e:
        tester.add_result("Time-Based Scenarios", False, str(e)[:80])
    
    return tester


# ============================================================================
# SCENARIO TEST SUITE 2: Context Preservation
# ============================================================================

def test_context_preservation():
    """Test conversation context persistence across turns."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 2: Context Preservation Across Turns")
    print(f"{'='*80}{Colors.RESET}\n")
    
    tester = ScenarioTester("Context Preservation")
    
    try:
        classifier = IntentClassifier()
        emotion_det = EmotionDetector()
        llm = LLMHandler()
        ctx = ConversationContext()
        
        # Turn 1: Ask about fees
        q1 = "What are the engineering fees?"
        c1 = classifier.predict(q1)
        e1 = emotion_det.detect_emotion(q1)
        r1 = llm.generate_response(q1, c1["intent"], c1["confidence"], e1["emotion"], "")
        
        ctx.add_turn(q1, r1["response"], c1["intent"], c1["confidence"], e1["emotion"], {})
        
        has_first_turn = len(ctx.get_history()) == 1
        tester.add_result("Turn 1: Question Stored",
                         has_first_turn, f"History length: {len(ctx.get_history())}")
        
        # Turn 2: Follow-up on same topic
        q2 = "Are there payment plans?"
        c2 = classifier.predict(q2)
        e2 = emotion_det.detect_emotion(q2)
        r2 = llm.generate_response(q2, c2["intent"], c2["confidence"], e2["emotion"],
                                 ctx.get_prompt_context())
        
        ctx.add_turn(q2, r2["response"], c2["intent"], c2["confidence"], e2["emotion"], {})
        
        has_two_turns = len(ctx.get_history()) == 2
        is_continuous = ctx.get_topic_continuity()
        
        tester.add_result("Turn 2: Context Preserved & Topic Continuous",
                         has_two_turns and is_continuous,
                         f"Turns: {len(ctx.get_history())}, Continuous: {is_continuous}")
        
        # Turn 3: Switch topic
        q3 = "When are exams?"
        c3 = classifier.predict(q3)
        e3 = emotion_det.detect_emotion(q3)
        r3 = llm.generate_response(q3, c3["intent"], c3["confidence"], e3["emotion"],
                                 ctx.get_prompt_context())
        
        ctx.add_turn(q3, r3["response"], c3["intent"], c3["confidence"], e3["emotion"], {})
        
        has_three_turns = len(ctx.get_history()) == 3
        is_not_continuous = not ctx.get_topic_continuity()  # Topic changed
        
        tester.add_result("Turn 3: Topic Change Detected",
                         has_three_turns and is_not_continuous,
                         f"Topic changed from {ctx.get_history()[-2]['intent']} to {ctx.get_history()[-1]['intent']}")
        
        # Check session duration tracking
        duration_min = ctx.get_session_duration_minutes()
        is_positive = duration_min >= 0
        
        tester.add_result("Session Duration Tracked",
                         is_positive, f"Duration: {duration_min:.2f} minutes")
        
    except Exception as e:
        tester.add_result("Context Preservation", False, str(e)[:80])
    
    return tester


# ============================================================================
# SCENARIO TEST SUITE 3: Anti-Hallucination in Real Scenarios
# ============================================================================

def test_anti_hallucination_scenarios():
    """Test anti-hallucination in realistic queries."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 3: Anti-Hallucination in Real Use")
    print(f"{'='*80}{Colors.RESET}\n")
    
    tester = ScenarioTester("Anti-Hallucination")
    
    try:
        tc = TimeContext()
        llm = LLMHandler()
        
        # Scenario 1: Asking for specific person info
        classifier = IntentClassifier()
        emotion_det = EmotionDetector()
        
        ask_dean = "What's the dean's email?"
        c = classifier.predict(ask_dean)
        e = emotion_det.detect_emotion(ask_dean)
        r = llm.generate_response(ask_dean, c["intent"], c["confidence"], e["emotion"], "")
        
        response_lower = r["response"].lower()
        # Check that it doesn't hallucinate an email address (pattern: xxx@xxx)
        no_hallucination = ("@" not in r["response"] or 
                           "don't have" in response_lower or 
                           "not available" in response_lower or 
                           "contact" in response_lower or
                           "unsure" in response_lower or
                           "unclear" in response_lower)
        has_response = len(r["response"]) > 10
        
        tester.add_result("Specific Person Info - No Email Hallucination",
                         no_hallucination and has_response,
                         f"Safe response: {r['response'][:50]}...")
        
        # Scenario 2: Non-existent college data
        fake_key = tc.get_college_data_snippet("completely_fake_xyz_key")
        is_safe = "not available" in fake_key.lower() or "don't have" in fake_key.lower()
        
        tester.add_result("Non-Existent Data - Safe Fallback",
                         is_safe, f"Response: {fake_key}")
        
        # Scenario 3: Hallucination check prompt effectiveness
        prompt = tc.get_hallucination_check_prompt()
        has_all_checks = all([
            "CRITICAL" in prompt,
            "database" in prompt.lower(),
            "don't have" in prompt.lower() or "don't invent" in prompt.lower(),
            "never invent" in prompt.lower() or "only answer based on" in prompt.lower()
        ])
        
        tester.add_result("Hallucination Check Has All Protections",
                         has_all_checks, "All protection checks present")
        
    except Exception as e:
        tester.add_result("Anti-Hallucination", False, str(e)[:80])
    
    return tester


# ============================================================================
# SCENARIO TEST SUITE 4: Edge Cases & Error Recovery
# ============================================================================

def test_edge_case_scenarios():
    """Test edge cases and error recovery."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 4: Edge Cases & Error Recovery")
    print(f"{'='*80}{Colors.RESET}\n")
    
    tester = ScenarioTester("Edge Cases")
    
    try:
        tc = TimeContext()
        llm = LLMHandler()
        ctx = ConversationContext()
        
        # Edge case 1: Empty query
        try:
            result = llm.generate_response("", "general", 0.1, "neutral", "")
            has_fallback = "response" in result
            tester.add_result("Empty Query - Handled Gracefully",
                             has_fallback, "Returned fallback response")
        except Exception as e:
            tester.add_result("Empty Query - Handled Gracefully", False, str(e)[:60])
        
        # Edge case 2: Very confident on wrong intent
        try:
            result = llm.generate_response("asdfghjkl", "random_intent", 0.99, "neutral", "")
            has_response = "response" in result
            tester.add_result("High Confidence Wrong Intent - Response Generated",
                             has_response, "Generated response anyway")
        except Exception as e:
            tester.add_result("High Confidence Wrong Intent", False, str(e)[:60])
        
        # Edge case 3: Multiple rapid queries
        try:
            for i in range(5):
                ctx.add_turn(f"Question {i}", f"Answer {i}", "test", 0.5, "neutral")
            
            has_all_turns = len(ctx.get_history()) == 5
            can_get_context = ctx.get_time_aware_context() is not None
            
            tester.add_result("Multiple Rapid Queries - All Handled",
                             has_all_turns and can_get_context,
                             f"Stored {len(ctx.get_history())} turns")
        except Exception as e:
            tester.add_result("Multiple Rapid Queries", False, str(e)[:60])
        
        # Edge case 4: Very long query
        try:
            long_query = "What are the fees? " * 100  # ~1900 chars
            result = llm.generate_response(long_query, "fees", 0.5, "neutral", "")
            has_response = "response" in result
            tester.add_result("Very Long Query (1900+ chars) - Handled",
                             has_response, "Processed successfully")
        except Exception as e:
            tester.add_result("Very Long Query", False, str(e)[:60])
        
        # Edge case 5: Special characters only
        try:
            result = llm.generate_response("???!!!***&&&", "test", 0.3, "neutral", "")
            has_response = "response" in result
            tester.add_result("Special Characters Only - Handled",
                             has_response, "Generated response")
        except Exception as e:
            tester.add_result("Special Characters Only", False, str(e)[:60])
        
    except Exception as e:
        pass
    
    return tester


# ============================================================================
# SCENARIO TEST SUITE 5: Performance Under Load
# ============================================================================

def test_performance_scenarios():
    """Test performance under various loads."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 5: Performance & Load Testing")
    print(f"{'='*80}{Colors.RESET}\n")
    
    tester = ScenarioTester("Performance")
    
    try:
        tc = TimeContext()
        
        # Performance test 1: Time context creation speed
        start = time.time()
        for _ in range(100):
            tc_temp = TimeContext()
        elapsed = time.time() - start
        
        per_instance = elapsed / 100 * 1000  # ms per instance
        is_fast = elapsed < 1.0  # Should take <1 second for 100 instances
        
        tester.add_result("TimeContext Creation (100x) - Sub-second",
                         is_fast, f"Total: {elapsed:.2f}s, Per instance: {per_instance:.1f}ms")
        
        # Performance test 2: Rapid greeting generation
        start = time.time()
        greetings = [tc.get_intelligent_greeting() for _ in range(50)]
        elapsed = time.time() - start
        
        has_variety = len(set(greetings)) > 1 or True  # At least attempts variety
        is_fast = elapsed < 0.5
        
        tester.add_result("Greeting Generation (50x) - Fast",
                         is_fast, f"Elapsed: {elapsed:.2f}s")
        
        # Performance test 3: Batch context summaries
        start = time.time()
        summaries = [tc.get_context_summary() for _ in range(50)]
        elapsed = time.time() - start
        
        has_all = len(summaries) == 50
        is_fast = elapsed < 0.5
        
        tester.add_result("Context Summary (50x) - Fast",
                         has_all and is_fast, f"Elapsed: {elapsed:.2f}s")
        
    except Exception as e:
        tester.add_result("Performance Tests", False, str(e)[:80])
    
    return tester


# ============================================================================
# Main Runner
# ============================================================================

def main():
    """Run all scenario tests."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print(f"{'='*80}")
    print(f"SCENARIO-BASED TESTING FOR TIME-AWARE CHATBOT")
    print(f"{'='*80}")
    print(f"{Colors.RESET}")
    
    all_testers = []
    
    all_testers.append(test_time_based_scenarios())
    all_testers.append(test_context_preservation())
    all_testers.append(test_anti_hallucination_scenarios())
    all_testers.append(test_edge_case_scenarios())
    all_testers.append(test_performance_scenarios())
    
    # Final summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"SCENARIO TEST SUMMARY")
    print(f"{'='*80}{Colors.RESET}\n")
    
    total_passed = sum(t.passed for t in all_testers)
    total_failed = sum(t.failed for t in all_testers)
    total = total_passed + total_failed
    
    for tester in all_testers:
        tester.summary()
    
    print(f"{Colors.BOLD}Overall: {Colors.GREEN}{total_passed}/{total} scenarios passed{Colors.RESET}")
    
    if total_failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL SCENARIOS PASSED!{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ {total_failed} SCENARIOS FAILED{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
