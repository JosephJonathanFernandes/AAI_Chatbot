#!/usr/bin/env python
"""
Test Suite for New Features: Scope Detection, Prompt Engineering, and Enhanced LLM Handler.
Tests all enhanced features including ALWAYS-ON LLM, RAG-lite knowledge grounding, and
multi-signal response control.
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Setup path
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from scope_detector import ScopeDetector
from prompt_engineering import PromptEngineer
from llm_handler import LLMHandler
from database import ChatbotDatabase
from context_manager import ConversationContext
from emotion_detector import EmotionDetector
from intent_model import IntentClassifier


class Colors:
    """ANSI color codes for terminal output."""
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
        print(f"{Colors.BOLD}{self.name} Summary:{Colors.RESET} "
              f"{Colors.GREEN}{self.passed} passed{Colors.RESET}, "
              f"{Colors.RED}{self.failed} failed{Colors.RESET} "
              f"out of {total}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        return self.failed == 0


# ============================================================================
# TEST SUITE 1: Scope Detection Module
# ============================================================================

def test_scope_detector():
    """Test ScopeDetector module."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 1: Scope Detection (Domain Filtering)")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("ScopeDetector")

    # Test 1: Initialization
    try:
        detector = ScopeDetector()
        suite.add_result("ScopeDetector Initialization", True,
                        "Instance created successfully")
    except Exception as e:
        suite.add_result("ScopeDetector Initialization", False, str(e)[:80])
        return suite

    # Test 2: In-scope queries
    in_scope_queries = [
        "What are the fees?",
        "When are exams?",
        "Tell me about placements",
        "What's the timetable?",
        "Who is the faculty?",
        "Can I use the library?",
        "What about hostel?",
        "Tell me about admission"
    ]

    for query in in_scope_queries:
        try:
            is_in_scope, reason, confidence = detector.is_in_scope(query)
            suite.add_result(f"In-Scope Query: '{query[:30]}...'",
                           is_in_scope, f"Reason: {reason[:50]}")
        except Exception as e:
            suite.add_result(f"In-Scope Query: '{query[:30]}...'",
                           False, str(e)[:80])

    # Test 3: Out-of-scope queries
    out_of_scope_queries = [
        "What is quantum physics?",
        "Tell me about politics",
        "How do I cook pasta?",
        "What's the weather?",
        "Can you write my code?"
    ]

    for query in out_of_scope_queries:
        try:
            is_in_scope, reason, confidence = detector.is_in_scope(query)
            suite.add_result(f"Out-of-Scope Query: '{query[:30]}...'",
                           not is_in_scope, f"Reason: {reason[:50]}")
        except Exception as e:
            suite.add_result(f"Out-of-Scope Query: '{query[:30]}...'",
                           False, str(e)[:80])

    # Test 4: Scope info structure
    try:
        is_in_scope, reason, confidence = detector.is_in_scope("What are fees?")
        has_scope = isinstance(is_in_scope, bool)
        has_reason = isinstance(reason, str) and len(reason) > 0
        has_confidence = isinstance(confidence, float) and 0 <= confidence <= 1
        suite.add_result("Scope Detection Returns Valid Structure",
                       has_scope and has_reason and has_confidence,
                       f"Scope: {is_in_scope}, Confidence: {confidence:.2f}")
    except Exception as e:
        suite.add_result("Scope Detection Returns Valid Structure",
                       False, str(e)[:80])

    suite.print_summary()
    return suite


# ============================================================================
# TEST SUITE 2: Prompt Engineering Module
# ============================================================================

def test_prompt_engineer():
    """Test PromptEngineer module."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 2: Prompt Engineering (Multi-Signal Control)")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("PromptEngineer")

    # Test 1: Initialization
    try:
        engineer = PromptEngineer()
        suite.add_result("PromptEngineer Initialization", True,
                        "Instance created successfully")
    except Exception as e:
        suite.add_result("PromptEngineer Initialization", False, str(e)[:80])
        return suite

    # Test 2: Build system prompt
    try:
        system_prompt = engineer.build_system_prompt(
            intent="fees",
            is_in_scope=True,
            emotion="neutral"
        )
        has_content = len(system_prompt) > 50
        has_intent = "fees" in system_prompt.lower()
        suite.add_result("Build System Prompt",
                       has_content and has_intent,
                       f"Prompt length: {len(system_prompt)}")
    except Exception as e:
        suite.add_result("Build System Prompt", False, str(e)[:80])

    # Test 3: Build user prompt with context
    try:
        user_prompt = engineer.build_user_prompt(
            user_input="What's the fee?",
            intent="fees",
            confidence=0.85,
            conversation_context="Previous: Asked about admission"
        )
        has_content = len(user_prompt) > 30
        has_input = "fee" in user_prompt.lower()
        suite.add_result("Build User Prompt",
                       has_content and has_input,
                       f"Prompt length: {len(user_prompt)}")
    except Exception as e:
        suite.add_result("Build User Prompt", False, str(e)[:80])

    # Test 4: Emotion-aware prompts
    emotions = ["stressed", "confused", "angry", "happy", "neutral"]
    for emotion in emotions:
        try:
            system_prompt = engineer.build_system_prompt(
                intent="general",
                is_in_scope=True,
                emotion=emotion
            )
            has_emotion_instruction = any(
                word in system_prompt.lower()
                for word in ["calm", "step-by-step", "empathetic", "encouragement"]
            )
            suite.add_result(f"Emotion-Aware Prompt: {emotion}",
                           has_emotion_instruction,
                           "Contains emotion-specific instruction")
        except Exception as e:
            suite.add_result(f"Emotion-Aware Prompt: {emotion}",
                           False, str(e)[:80])

    # Test 5: Confidence-based prompts
    try:
        high_conf_prompt = engineer.build_system_prompt(
            intent="fees",
            is_in_scope=True,
            emotion="neutral"
        )
        low_conf_prompt = engineer.build_system_prompt(
            intent="general",
            is_in_scope=True,
            emotion="neutral",
            confidence=0.2
        )
        prompts_differ = high_conf_prompt != low_conf_prompt
        suite.add_result("Confidence Affects Prompt Construction",
                       prompts_differ,
                       "High/Low confidence prompts differ")
    except Exception as e:
        suite.add_result("Confidence Affects Prompt Construction",
                       False, str(e)[:80])

    # Test 6: Out-of-scope prompts
    try:
        oos_prompt = engineer.build_system_prompt(
            intent="general",
            is_in_scope=False,
            emotion="neutral"
        )
        has_template = "template" in oos_prompt.lower() or "domain" in oos_prompt.lower()
        suite.add_result("Out-of-Scope Prompt Template",
                       has_template,
                       "Prompt includes scope-aware instruction")
    except Exception as e:
        suite.add_result("Out-of-Scope Prompt Template", False, str(e)[:80])

    suite.print_summary()
    return suite


# ============================================================================
# TEST SUITE 3: Enhanced LLM Handler
# ============================================================================

def test_enhanced_llm_handler():
    """Test enhanced LLMHandler with new features."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 3: Enhanced LLM Handler (ALWAYS-ON Strategy)")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("Enhanced LLMHandler")

    # Test 1: Initialization
    try:
        handler = LLMHandler()
        suite.add_result("LLMHandler Initialization", True,
                        "Instance with new components created")
    except Exception as e:
        suite.add_result("LLMHandler Initialization", False, str(e)[:80])
        return suite

    # Test 2: ALWAYS-ON LLM strategy
    try:
        response = handler.generate_response(
            user_input="What are fees?",
            intent="fees",
            confidence=0.95,
            emotion="neutral"
        )
        has_response = 'response' in response and len(response['response']) > 0
        never_bypassed = response is not None
        suite.add_result("ALWAYS-ON LLM Strategy (Never Bypassed)",
                       has_response and never_bypassed,
                       f"Response length: {len(response.get('response', ''))}")
    except Exception as e:
        suite.add_result("ALWAYS-ON LLM Strategy (Never Bypassed)",
                       False, str(e)[:80])

    # Test 3: Response includes scope info
    try:
        response = handler.generate_response(
            user_input="What are fees?",
            intent="fees",
            confidence=0.95,
            emotion="neutral"
        )
        has_scope_flag = 'is_in_scope' in response
        has_clarify_flag = 'should_clarify' in response
        suite.add_result("Response Includes Scope & Clarify Flags",
                       has_scope_flag and has_clarify_flag,
                       "Response metadata complete")
    except Exception as e:
        suite.add_result("Response Includes Scope & Clarify Flags",
                       False, str(e)[:80])

    # Test 4: Low confidence triggers clarification
    try:
        response = handler.generate_response(
            user_input="Something about college?",
            intent="general",
            confidence=0.2,
            emotion="neutral"
        )
        clarify_flag = response.get('should_clarify', False)
        suite.add_result("Low Confidence Triggers Clarification",
                       clarify_flag,
                       "Clarification flag set for low confidence")
    except Exception as e:
        suite.add_result("Low Confidence Triggers Clarification",
                       False, str(e)[:80])

    # Test 5: Out-of-scope queries handled gracefully
    try:
        response = handler.generate_response(
            user_input="What is quantum physics?",
            intent="other",
            confidence=0.8,
            emotion="neutral"
        )
        has_response = 'response' in response and len(response['response']) > 0
        scope_flag = response.get('is_in_scope', True) == False
        suite.add_result("Out-of-Scope Query Handled",
                       has_response and scope_flag,
                       "Response generated with scope flag")
    except Exception as e:
        suite.add_result("Out-of-Scope Query Handled", False, str(e)[:80])

    suite.print_summary()
    return suite


# ============================================================================
# TEST SUITE 4: Enhanced Database Schema
# ============================================================================

def test_enhanced_database():
    """Test enhanced database schema with new fields."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 4: Enhanced Database Schema")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("Enhanced Database")

    # Use temporary database for testing
    db_path = "test_new_features.db"

    # Test 1: Initialization with new schema
    try:
        db = ChatbotDatabase(db_path=db_path)
        suite.add_result("Database Initialization with New Schema", True,
                        "Enhanced tables created")
    except Exception as e:
        suite.add_result("Database Initialization with New Schema",
                       False, str(e)[:80])
        return suite

    # Test 2: Log interaction with new fields
    try:
        success = db.log_interaction(
            user_input="What are fees?",
            intent="fees",
            confidence=0.95,
            emotion="neutral",
            response="The fees are ...",
            response_time=0.5,
            llm_source="groq",
            is_in_scope=True,
            should_clarify=False,
            scope_reason="Matched fee keywords",
            session_id="session_001"
        )
        suite.add_result("Log Interaction with New Fields", success,
                        "All fields logged successfully")
    except Exception as e:
        suite.add_result("Log Interaction with New Fields",
                       False, str(e)[:80])

    # Test 3: Log out-of-scope interaction
    try:
        success = db.log_interaction(
            user_input="What is quantum physics?",
            intent="other",
            confidence=0.8,
            emotion="neutral",
            response="I can only answer college questions",
            response_time=0.3,
            llm_source="groq",
            is_in_scope=False,
            should_clarify=False,
            scope_reason="Out of domain - Physics topic",
            session_id="session_001"
        )
        suite.add_result("Log Out-of-Scope Interaction", success,
                        "Out-of-scope interaction logged")
    except Exception as e:
        suite.add_result("Log Out-of-Scope Interaction",
                       False, str(e)[:80])

    # Test 4: Retrieve logs with scope info
    try:
        logs = db.get_logs(limit=10)
        has_scope_field = any('is_in_scope' in log for log in logs)
        has_scope_reason = any('scope_reason' in log for log in logs)
        has_session_field = any('session_id' in log for log in logs)
        suite.add_result("Retrieve Enhanced Logs",
                       has_scope_field and has_scope_reason and has_session_field,
                       f"Retrieved {len(logs)} logs with new fields")
    except Exception as e:
        suite.add_result("Retrieve Enhanced Logs", False, str(e)[:80])

    # Test 5: Analytics includes scope counts
    try:
        analytics = db.get_analytics_summary()
        has_interactions = 'total_interactions' in analytics
        has_confidence = 'average_confidence' in analytics
        has_response_time = 'average_response_time' in analytics
        suite.add_result("Analytics Summary Available",
                       has_interactions and has_confidence and has_response_time,
                       "Analytics computed successfully")
    except Exception as e:
        suite.add_result("Analytics Summary Available",
                       False, str(e)[:80])

    # Cleanup
    try:
        import os
        if os.path.exists(db_path):
            os.remove(db_path)
    except:
        pass

    suite.print_summary()
    return suite


# ============================================================================
# TEST SUITE 5: Integration Tests
# ============================================================================

def test_integrated_workflow():
    """Test complete workflow with all new components."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"TEST SUITE 5: Integrated Workflow (End-to-End)")
    print(f"{'='*80}{Colors.RESET}\n")

    suite = TestSuite("Integrated Workflow")

    try:
        # Initialize all components
        detector = ScopeDetector()
        engineer = PromptEngineer()
        handler = LLMHandler()
        db = ChatbotDatabase(db_path="test_integrated.db")
        context = ConversationContext()
        emotion_d = EmotionDetector()
        intent_c = IntentClassifier()
        suite.add_result("Initialize All Components", True,
                        "All modules ready")
    except Exception as e:
        suite.add_result("Initialize All Components", False, str(e)[:80])
        return suite

    # Test workflow: User query → Scope check → Intent/Emotion detection
    # → Prompt construction → LLM response → Database logging
    test_query = "What are the tuition fees for engineering?"

    try:
        # Step 1: Scope detection
        is_in_scope, reason, scope_conf = detector.is_in_scope(test_query)
        suite.add_result("Step 1: Scope Detection", is_in_scope,
                        f"Scope: {is_in_scope}, Reason: {reason[:40]}")
    except Exception as e:
        suite.add_result("Step 1: Scope Detection", False, str(e)[:80])

    try:
        # Step 2: Intent & Emotion detection
        intent_result = intent_c.predict(test_query)
        intent = intent_result.get('intent', 'unknown')
        intent_conf = intent_result.get('confidence', 0)
        emotion_result = emotion_d.detect_emotion(test_query)
        emotion = emotion_result.get('emotion', 'neutral')
        suite.add_result("Step 2: Intent & Emotion Detection",
                       intent is not None and emotion is not None,
                       f"Intent: {intent}, Emotion: {emotion}")
    except Exception as e:
        suite.add_result("Step 2: Intent & Emotion Detection",
                       False, str(e)[:80])

    try:
        # Step 3: Prompt construction
        sys_prompt = engineer.build_system_prompt(
            intent=intent,
            is_in_scope=is_in_scope,
            emotion=emotion
        )
        usr_prompt = engineer.build_user_prompt(
            user_input=test_query,
            intent=intent,
            confidence=intent_conf
        )
        suite.add_result("Step 3: Prompt Engineering",
                       len(sys_prompt) > 0 and len(usr_prompt) > 0,
                       f"System: {len(sys_prompt)}B, User: {len(usr_prompt)}B")
    except Exception as e:
        suite.add_result("Step 3: Prompt Engineering", False, str(e)[:80])

    try:
        # Step 4: LLM response
        response = handler.generate_response(
            user_input=test_query,
            intent=intent,
            confidence=intent_conf,
            emotion=emotion
        )
        suite.add_result("Step 4: LLM Response Generation",
                       'response' in response,
                       f"Response length: "
                       f"{len(response.get('response', ''))}")
    except Exception as e:
        suite.add_result("Step 4: LLM Response Generation",
                       False, str(e)[:80])

    try:
        # Step 5: Database logging
        success = db.log_interaction(
            user_input=test_query,
            intent=intent,
            confidence=intent_conf,
            emotion=emotion,
            response=response.get('response', ''),
            response_time=response.get('time', 0),
            llm_source=response.get('source', 'groq'),
            is_in_scope=is_in_scope,
            should_clarify=response.get('should_clarify', False),
            scope_reason=reason,
            session_id="test_session_001"
        )
        suite.add_result("Step 5: Database Logging",
                       success,
                       "Interaction logged with all metadata")
    except Exception as e:
        suite.add_result("Step 5: Database Logging", False, str(e)[:80])

    # Cleanup
    try:
        import os
        for f in ["test_new_features.db", "test_integrated.db"]:
            if os.path.exists(f):
                os.remove(f)
    except:
        pass

    suite.print_summary()
    return suite


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all test suites."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"AAI CHATBOT: NEW FEATURES TEST SUITE")
    print(f"Testing Scope Detection, Prompt Engineering, & Enhanced LLM Handler")
    print(f"{'='*80}{Colors.RESET}\n")

    results = []

    # Run all test suites
    results.append(test_scope_detector())
    results.append(test_prompt_engineer())
    results.append(test_enhanced_llm_handler())
    results.append(test_enhanced_database())
    results.append(test_integrated_workflow())

    # Print overall summary
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_tests = total_passed + total_failed

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"OVERALL TEST RESULTS")
    print(f"{'='*80}{Colors.RESET}\n")
    print(f"  Total Tests Run: {total_tests}")
    print(f"  {Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {total_failed}{Colors.RESET}")
    print(f"  Success Rate: {100 * total_passed / total_tests:.1f}%\n")

    if total_failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED!{Colors.RESET}\n")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.RESET}\n")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
