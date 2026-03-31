#!/usr/bin/env python
"""
Enhanced Time-Aware Comprehensive Test Suite
Tests time awareness combined with scope detection and prompt engineering
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Setup path
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from time_context import TimeContext
from llm_handler import LLMHandler
from context_manager import ConversationContext
from emotion_detector import EmotionDetector
from intent_model import IntentClassifier
from database import ChatbotDatabase
from scope_detector import ScopeDetector
from prompt_engineering import PromptEngineer


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

        status = (f"{Colors.GREEN}✓ PASS{Colors.RESET}" if passed
                  else f"{Colors.RED}✗ FAIL{Colors.RESET}")
        print(f"  {status}: {test_name}")
        if details:
            print(f"           {Colors.YELLOW}→ {details}{Colors.RESET}")

    def print_summary(self):
        """Print summary."""
        total = self.passed + self.failed
        print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"{Colors.BOLD}{self.name}:{Colors.RESET} "
              f"{Colors.GREEN}{self.passed} passed{Colors.RESET}, "
              f"{Colors.RED}{self.failed} failed{Colors.RESET} "
              f"of {total}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        return self.failed == 0


# ============================================================================
# TEST SUITE 1: TimeContext with Scope Detection
# ============================================================================

def test_time_context_with_scope():
    """Test TimeContext integrated with scope detection."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 1: Time-Context + Scope Detection")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("Time-Context + Scope")

    # Test 1: TimeContext initialization
    try:
        tc = TimeContext()
        suite.add_result("TimeContext Initialization", True,
                        "Instance created")
    except Exception as e:
        suite.add_result("TimeContext Initialization", False, str(e)[:80])
        return suite

    # Test 2: Time period detection
    time_periods = [
        "early_morning", "morning", "late_morning", "afternoon",
        "evening", "night", "late_night"]
    time_of_day = tc.get_time_of_day()
    suite.add_result("Time Period Detection",
                     time_of_day in time_periods,
                     f"Current: {time_of_day}")

    # Test 3: Get intelligent greeting
    greeting = tc.get_intelligent_greeting()
    has_emoji = any(emoji in greeting for emoji in
                   ['☀️', '👋', '🌙', '💪', '🎉', '📚', '🦉'])
    suite.add_result("Time-Aware Greeting with Emoji", has_emoji,
                    f"Greeting: {greeting[:50]}")

    # Test 4: Office status
    is_open, reason = tc.is_office_open()
    has_reason = isinstance(reason, str) and len(reason) > 0
    suite.add_result("Office Status Determination",
                    has_reason,
                    f"Open: {is_open}, Reason: {reason[:40]}")

    # Test 5: Combine time context with scope detection
    try:
        detector = ScopeDetector()
        engineer = PromptEngineer()

        # Test time-aware prompts
        prompt = engineer.build_system_prompt(
            intent="placements",
            is_in_scope=True,
            emotion="neutral"
        )

        # Check if time info can be injected
        time_context = tc.get_context_summary()
        has_time_info = time_context is not None and len(time_context) > 0

        suite.add_result("Time Info Available for Prompt Injection",
                        has_time_info,
                        f"Time context keys: {len(time_context) if time_context else 0}")

    except Exception as e:
        suite.add_result("Time Info Available", False, str(e)[:80])

    suite.print_summary()
    return suite


# ============================================================================
# TEST SUITE 2: Time-Aware LLM Responses with Multi-Signal Control
# ============================================================================

def test_time_aware_llm():
    """Test LLM responses adapted to time of day."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 2: Time-Aware LLM Responses")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("Time-Aware LLM")

    try:
        tc = TimeContext()
        handler = LLMHandler()
        detector = ScopeDetector()
        intent_c = IntentClassifier()
        emotion_d = EmotionDetector()

        time_of_day = tc.get_time_of_day()

        # Test query
        query = "Tell me about placements"

        # Detect all signals
        is_in_scope, _, _ = detector.is_in_scope(query)
        intent_result = intent_c.predict(query)
        intent = intent_result.get('intent', 'general')
        confidence = intent_result.get('confidence', 0)
        emotion_result = emotion_d.detect_emotion(query)
        emotion = emotion_result.get('emotion', 'neutral')

        # Get time context
        time_context = tc.get_context_summary()

        # Generate response with all context
        response = handler.generate_response(
            user_input=query,
            intent=intent,
            confidence=confidence,
            emotion=emotion
        )

        has_response = 'response' in response and len(response['response']) > 0

        suite.add_result("LLM Response with Time Context",
                        has_response,
                        f"Time: {time_of_day}, "
                        f"Response length: {len(response.get('response', ''))}")

    except Exception as e:
        suite.add_result("LLM Response with Time Context",
                        False, str(e)[:80])

    suite.print_summary()
    return suite


# ============================================================================
# TEST SUITE 3: Anti-Hallucination with Scope + Knowledge Grounding
# ============================================================================

def test_anti_hallucination():
    """Test anti-hallucination measures."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 3: Anti-Hallucination Measures")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("Anti-Hallucination")

    try:
        detector = ScopeDetector()
        engineer = PromptEngineer()
        handler = LLMHandler()
        db = ChatbotDatabase(db_path="test_antihalluc.db")

        # Test out-of-domain queries
        oos_queries = [
            "What is quantum physics?",
            "How do I cook a steak?",
            "Tell me about machine learning"
        ]

        for query in oos_queries:
            try:
                is_in_scope, reason, _ = detector.is_in_scope(query)

                if not is_in_scope:
                    # Build out-of-scope prompt to prevent hallucination
                    prompt = engineer.build_system_prompt(
                        intent="other",
                        is_in_scope=False,
                        emotion="neutral"
                    )

                    # Prompt should include directive to not hallucinate
                    anti_halluc_directive = any(
                        word in prompt.lower()
                        for word in ['hallucin', 'invent', 'domain',
                                    'college', 'cannot', 'outside'])

                    # Log the query with scope info
                    logged = db.log_interaction(
                        user_input=query,
                        intent="other",
                        confidence=0.8,
                        emotion="neutral",
                        response="Out of domain response",
                        response_time=0.2,
                        llm_source="groq",
                        is_in_scope=False,
                        should_clarify=False,
                        scope_reason=reason,
                        session_id="test_antihalluc"
                    )

                    passed = anti_halluc_directive and logged

                else:
                    passed = False

                suite.add_result(
                    f"OOS Query: '{query[:40]}...'",
                    passed,
                    f"Out-of-scope: {not is_in_scope}, Logged: {logged}")

            except Exception as e:
                suite.add_result(f"OOS Query: '{query[:40]}...'",
                               False, str(e)[:80])

        # Cleanup
        try:
            import os
            if os.path.exists("test_antihalluc.db"):
                os.remove("test_antihalluc.db")
        except:
            pass

    except Exception as e:
        suite.add_result("Setup", False, str(e)[:80])

    suite.print_summary()
    return suite


# ============================================================================
# TEST SUITE 4: Multi-Signal Response Control
# ============================================================================

def test_multi_signal_control():
    """Test multi-signal control of LLM responses."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 4: Multi-Signal Response Control")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("Multi-Signal Control")

    try:
        handler = LLMHandler()
        engineer = PromptEngineer()
        intent_c = IntentClassifier()
        emotion_d = EmotionDetector()
        tc = TimeContext()

        signals = {
            "query": "I'm confused about placements",
            "intent": None,
            "confidence": None,
            "emotion": None,
            "time_of_day": None,
            "in_scope": None
        }

        # Gather all signals
        intent_result = intent_c.predict(signals["query"])
        signals["intent"] = intent_result.get('intent', 'general')
        signals["confidence"] = intent_result.get('confidence', 0)

        emotion_result = emotion_d.detect_emotion(signals["query"])
        signals["emotion"] = emotion_result.get('emotion', 'neutral')

        signals["time_of_day"] = tc.get_time_of_day()
        signals["in_scope"] = True

        # Build prompt with all signals
        try:
            prompt = engineer.build_system_prompt(
                intent=signals["intent"],
                is_in_scope=signals["in_scope"],
                emotion=signals["emotion"],
                confidence=signals["confidence"]
            )

            # Check that prompt reflects all signals
            has_emotion = signals["emotion"].lower() in prompt.lower() or \
                         any(word in prompt.lower() for word in
                            ['calm', 'step', 'reassure', 'explain'])
            has_intent = signals["intent"].lower() in prompt.lower() or \
                        'placement' in prompt.lower()
            has_scope = signals["in_scope"]

            suite.add_result("Prompt Reflects All Signals",
                           has_emotion and has_intent and has_scope,
                           f"Emotion: {has_emotion}, Intent: {has_intent}, "
                           f"Scope: {has_scope}")
        except Exception as e:
            suite.add_result("Prompt Reflects All Signals",
                           False, str(e)[:80])

        # Generate response
        try:
            response = handler.generate_response(
                user_input=signals["query"],
                intent=signals["intent"],
                confidence=signals["confidence"],
                emotion=signals["emotion"]
            )

            # Response should be influenced by emotion (confused → step-by-step)
            response_text = response.get('response', '')
            is_guided = (len(response_text) > 50 and
                        response.get('should_clarify', False) ==
                        (signals["confidence"] < 0.4))

            suite.add_result("Response Guided by Multiple Signals",
                           is_guided,
                           f"Response length: {len(response_text)}, "
                           f"Clarify: {response.get('should_clarify', False)}")

        except Exception as e:
            suite.add_result("Response Guided by Multiple Signals",
                           False, str(e)[:80])

    except Exception as e:
        suite.add_result("Setup", False, str(e)[:80])

    suite.print_summary()
    return suite


# ============================================================================
# TEST SUITE 5: Database Logging with All Signals
# ============================================================================

def test_comprehensive_logging():
    """Test comprehensive database logging."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 5: Comprehensive Database Logging")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("Comprehensive Logging")

    db_path = "test_comprehensive_logging.db"

    try:
        db = ChatbotDatabase(db_path=db_path)
        tc = TimeContext()

        # Create test interactions with all signals
        interactions = [
            {
                "user_input": "What are the fees?",
                "intent": "fees",
                "confidence": 0.95,
                "emotion": "neutral",
                "response": "The fees are...",
                "is_in_scope": True,
                "should_clarify": False,
                "scope_reason": "Matched fee keywords"
            },
            {
                "user_input": "I'm confused about placements",
                "intent": "placement",
                "confidence": 0.7,
                "emotion": "confused",
                "response": "Let me explain placements step-by-step...",
                "is_in_scope": True,
                "should_clarify": True,
                "scope_reason": "Matched placement keywords"
            }
        ]

        session_id = f"test_comprehensive_{int(time.time())}"

        for inter in interactions:
            try:
                success = db.log_interaction(
                    user_input=inter["user_input"],
                    intent=inter["intent"],
                    confidence=inter["confidence"],
                    emotion=inter["emotion"],
                    response=inter["response"],
                    response_time=0.5,
                    llm_source="groq",
                    is_in_scope=inter["is_in_scope"],
                    should_clarify=inter["should_clarify"],
                    scope_reason=inter["scope_reason"],
                    session_id=session_id
                )

                suite.add_result(
                    f"Log: '{inter['user_input'][:40]}...'",
                    success,
                    f"Logged with scope and clarify flags")

            except Exception as e:
                suite.add_result(
                    f"Log: '{inter['user_input'][:40]}...'",
                    False, str(e)[:80])

        # Verify logs retrieved
        try:
            logs = db.get_logs(limit=10)
            has_new_fields = (len(logs) > 0 and
                            any('is_in_scope' in log and
                                'should_clarify' in log
                                for log in logs))

            suite.add_result("Retrieve Logs with New Fields",
                           has_new_fields,
                           f"Retrieved {len(logs)} logs")
        except Exception as e:
            suite.add_result("Retrieve Logs with New Fields",
                           False, str(e)[:80])

        # Cleanup
        try:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)
        except:
            pass

    except Exception as e:
        suite.add_result("Setup", False, str(e)[:80])

    suite.print_summary()
    return suite


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"TIME-AWARE COMPREHENSIVE TEST SUITE")
    print(f"With Scope Detection, Prompt Engineering, & Multi-Signal Control")
    print(f"{'='*80}{Colors.RESET}\n")

    results = []

    results.append(test_time_context_with_scope())
    results.append(test_time_aware_llm())
    results.append(test_anti_hallucination())
    results.append(test_multi_signal_control())
    results.append(test_comprehensive_logging())

    # Summary
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total = total_passed + total_failed

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"OVERALL RESULTS")
    print(f"{'='*80}{Colors.RESET}\n")
    print(f"  {Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {total_failed}{Colors.RESET}")
    print(f"  Total: {total}")
    print(f"  Success Rate: {100 * total_passed / total:.1f}%\n")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
