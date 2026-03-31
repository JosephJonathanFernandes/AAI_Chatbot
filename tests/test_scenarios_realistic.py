#!/usr/bin/env python
"""
Scenario-Based Tests for Enhanced Chatbot
Tests realistic user interactions with scope detection, prompts engineering, and new features
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
from scope_detector import ScopeDetector
from prompt_engineering import PromptEngineer
from database import ChatbotDatabase


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

        status = (f"{Colors.GREEN}✓{Colors.RESET}" if passed
                  else f"{Colors.RED}✗{Colors.RESET}")
        print(f"  {status} {scenario}")
        if details:
            print(f"      → {Colors.YELLOW}{details}{Colors.RESET}")

    def summary(self):
        """Print summary."""
        total = self.passed + self.failed
        print(f"\n{Colors.CYAN}{self.name}: "
              f"{Colors.GREEN}{self.passed}/{total} passed{Colors.RESET}\n")
        return self.failed == 0


# ============================================================================
# SCENARIO TEST SUITE 1: In-Scope Query Handling with Multi-Signal Detection
# ============================================================================

def test_in_scope_queries():
    """Test handling of in-domain queries."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 1: In-Scope Query Handling")
    print(f"{'='*80}{Colors.RESET}\n")

    tester = ScenarioTester("In-Scope Queries")

    try:
        detector = ScopeDetector()
        engineer = PromptEngineer()
        intent_c = IntentClassifier()
        emotion_d = EmotionDetector()
        handler = LLMHandler()

        queries = [
            "What are the engineering fees?",
            "When are the exams scheduled?",
            "Tell me about placements",
            "What's the hostel fee?"
        ]

        for query in queries:
            try:
                # Scope detection
                is_in_scope, reason, scope_conf = detector.is_in_scope(query)

                # Intent detection
                intent_result = intent_c.predict(query)
                intent = intent_result.get('intent', 'general')
                intent_conf = intent_result.get('confidence', 0)

                # Emotion detection
                emotion_result = emotion_d.detect_emotion(query)
                emotion = emotion_result.get('emotion', 'neutral')

                # Check scope is correct
                scope_correct = is_in_scope

                # Check intent is detected
                intent_detected = intent is not None and intent_conf > 0.3

                # Prompt engineering
                sys_prompt = engineer.build_system_prompt(
                    intent=intent,
                    confidence=intent_conf,
                    emotion=emotion,
                    is_in_scope=is_in_scope
                )
                prompt_valid = len(sys_prompt) > 50

                passed = (scope_correct and intent_detected and
                         prompt_valid)
                tester.add_result(
                    f"Query: '{query[:40]}...'",
                    passed,
                    f"Scope: {is_in_scope}, Intent: {intent}, "
                    f"Emotion: {emotion}")

            except Exception as e:
                tester.add_result(
                    f"Query: '{query[:40]}...'",
                    False, str(e)[:60])

    except Exception as e:
        tester.add_result("Setup", False, str(e)[:60])

    tester.summary()


# ============================================================================
# SCENARIO TEST SUITE 2: Out-of-Scope Query Handling
# ============================================================================

def test_out_of_scope_queries():
    """Test handling of out-of-domain queries."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 2: Out-of-Scope Query Handling")
    print(f"{'='*80}{Colors.RESET}\n")

    tester = ScenarioTester("Out-of-Scope Queries")

    try:
        detector = ScopeDetector()
        engineer = PromptEngineer()
        handler = LLMHandler()
        db = ChatbotDatabase(db_path="test_scenarios_oos.db")

        queries = [
            "What is quantum physics?",
            "How do I cook pasta?",
            "Tell me about politics",
            "Write my code for me"
        ]

        for query in queries:
            try:
                # Scope detection
                is_in_scope, reason, scope_conf = detector.is_in_scope(query)

                # Check out-of-scope
                out_of_scope_detected = not is_in_scope

                # Generate response
                response = handler.generate_response(
                    user_input=query,
                    intent="other",
                    confidence=0.6,
                    emotion="neutral"
                )

                response_generated = (
                    'response' in response and
                    len(response['response']) > 0)

                # Log to database
                success = db.log_interaction(
                    user_input=query,
                    intent="other",
                    confidence=0.6,
                    emotion="neutral",
                    response=response.get('response', ''),
                    response_time=response.get('time', 0),
                    llm_source=response.get('source', 'groq'),
                    is_in_scope=is_in_scope,
                    should_clarify=False,
                    scope_reason=reason,
                    session_id="test_oos"
                )

                passed = (out_of_scope_detected and response_generated
                         and success)
                tester.add_result(
                    f"Query: '{query[:40]}...'",
                    passed,
                    f"Detected OOS: {out_of_scope_detected}, "
                    f"Logged: {success}")

            except Exception as e:
                tester.add_result(
                    f"Query: '{query[:40]}...'",
                    False, str(e)[:60])

        # Cleanup
        try:
            import os
            if os.path.exists("test_scenarios_oos.db"):
                os.remove("test_scenarios_oos.db")
        except:
            pass

    except Exception as e:
        tester.add_result("Setup", False, str(e)[:60])

    tester.summary()


# ============================================================================
# SCENARIO TEST SUITE 3: Emotion-Based Response Adaptation
# ============================================================================

def test_emotion_based_responses():
    """Test emotion-aware response adaptation."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 3: Emotion-Based Response Adaptation")
    print(f"{'='*80}{Colors.RESET}\n")

    tester = ScenarioTester("Emotion-Based Responses")

    try:
        emotion_d = EmotionDetector()
        engineer = PromptEngineer()
        intent_c = IntentClassifier()

        emotion_test_cases = [
            ("I'm so stressed about the exams!", "stressed"),
            ("I'm confused about placements", "confused"),
            ("I'm angry about the fees!", "angry"),
        ]

        for query, expected_emotion in emotion_test_cases:
            try:
                # Detect emotion
                emotion_result = emotion_d.detect_emotion(query)
                emotion = emotion_result.get('emotion', '')

                # Get intent
                intent_result = intent_c.predict(query)
                intent = intent_result.get('intent', 'general')

                # Build emotion-aware prompt
                prompt = engineer.build_system_prompt(
                    intent=intent,
                    is_in_scope=True,
                    emotion=emotion
                )

                # Check if prompt contains emotion-aware instructions
                emotion_keywords = {
                    'stressed': ['calm', 'reassure'],
                    'confused': ['step-by-step', 'explain'],
                    'angry': ['empathetic', 'patience']
                }

                expected_keywords = emotion_keywords.get(emotion, [])
                has_emotion_instruction = any(
                    keyword in prompt.lower()
                    for keyword in expected_keywords)

                passed = has_emotion_instruction

                tester.add_result(
                    f"Emotion '{emotion}' - Query: '{query[:30]}...'",
                    passed,
                    f"Prompt adapted for emotion: {passed}")

            except Exception as e:
                tester.add_result(
                    f"Emotion - Query: '{query[:30]}...'",
                    False, str(e)[:60])

    except Exception as e:
        tester.add_result("Setup", False, str(e)[:60])

    tester.summary()


# ============================================================================
# SCENARIO TEST SUITE 4: Low Confidence Clarification
# ============================================================================

def test_low_confidence_clarification():
    """Test clarification when confidence is low."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 4: Low Confidence Clarification")
    print(f"{'='*80}{Colors.RESET}\n")

    tester = ScenarioTester("Low Confidence Handling")

    try:
        intent_c = IntentClassifier()
        handler = LLMHandler()
        engineer = PromptEngineer()

        ambiguous_queries = [
            "Tell me something",
            "What about stuff?",
            "I need information",
            "Help me please"
        ]

        for query in ambiguous_queries:
            try:
                # Get intent with potentially low confidence
                intent_result = intent_c.predict(query)
                intent = intent_result.get('intent', 'general')
                confidence = intent_result.get('confidence', 0)

                # Generate response
                response = handler.generate_response(
                    user_input=query,
                    intent=intent,
                    confidence=confidence,
                    emotion="neutral"
                )

                # Check if clarification is triggered for low confidence
                should_clarify = response.get('should_clarify', False)

                # Build clarification prompt if needed
                if confidence < 0.4:
                    clarif_prompt = engineer.build_clarification_prompt(
                        intent=intent,
                        detected_intent_confidence=confidence
                    )
                    has_clarif_prompt = len(clarif_prompt) > 20
                else:
                    has_clarif_prompt = True

                passed = has_clarif_prompt

                tester.add_result(
                    f"Query: '{query[:40]}...' (Conf: {confidence:.2f})",
                    passed,
                    f"Clarify: {should_clarify}, "
                    f"Conf: {confidence:.2f}")

            except Exception as e:
                tester.add_result(
                    f"Query: '{query[:40]}...'",
                    False, str(e)[:60])

    except Exception as e:
        tester.add_result("Setup", False, str(e)[:60])

    tester.summary()


# ============================================================================
# SCENARIO TEST SUITE 5: Multi-Turn Conversation with Context
# ============================================================================

def test_multi_turn_conversation():
    """Test multi-turn conversation with context management."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}")
    print(f"SCENARIO 5: Multi-Turn Conversation")
    print(f"{'='*80}{Colors.RESET}\n")

    tester = ScenarioTester("Multi-Turn Conversations")

    try:
        context = ConversationContext()
        detector = ScopeDetector()
        intention_c = IntentClassifier()
        handler = LLMHandler()
        db = ChatbotDatabase(db_path="test_scenarios_multiturn.db")

        # Simulate multi-turn conversation
        conversation = [
            "What are the fees?",
            "Is there a discount?",
            "When do I need to pay?"
        ]

        session_id = "test_multiturn_001"

        for i, query in enumerate(conversation, 1):
            try:
                # Get context
                context_str = context.get_formatted_history()

                # Scope check
                is_in_scope, reason, conf = detector.is_in_scope(query)

                # Intent check
                intent_result = intention_c.predict(query)
                intent = intent_result.get('intent', 'general')

                # Generate response
                response = handler.generate_response(
                    user_input=query,
                    intent=intent,
                    confidence=0.85,
                    emotion="neutral",
                    conversation_history=context.get_history()
                )

                # Update context AFTER getting response
                context.add_turn(user_input=query, bot_response=response.get('response', ''), intent=intent, confidence=0.85, emotion="neutral")

                # Log with session ID
                logged = db.log_interaction(
                    user_input=query,
                    intent=intent,
                    confidence=0.85,
                    emotion="neutral",
                    response=response.get('response', ''),
                    response_time=response.get('time', 0),
                    llm_source=response.get('source', 'groq'),
                    is_in_scope=is_in_scope,
                    should_clarify=False,
                    scope_reason=reason,
                    session_id=session_id
                )

                context_used = len(context_str) > 0
                response_valid = 'response' in response

                passed = logged and response_valid

                tester.add_result(
                    f"Turn {i}: '{query[:35]}...'",
                    passed,
                    f"Context used: {context_used}, Logged: {logged}")

            except Exception as e:
                tester.add_result(
                    f"Turn {i}: '{query[:35]}...'",
                    False, str(e)[:60])

        # Cleanup
        try:
            import os
            if os.path.exists("test_scenarios_multiturn.db"):
                os.remove("test_scenarios_multiturn.db")
        except:
            pass

    except Exception as e:
        tester.add_result("Setup", False, str(e)[:60])

    tester.summary()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all scenario tests."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"REALISTIC SCENARIO TESTS")
    print(f"Testing New Features: Scope Detection, Prompt Engineering, "
          f"Multi-Turn Context")
    print(f"{'='*80}{Colors.RESET}\n")

    test_in_scope_queries()
    test_out_of_scope_queries()
    test_emotion_based_responses()
    test_low_confidence_clarification()
    test_multi_turn_conversation()

    print(f"\n{Colors.CYAN}{'='*80}")
    print(f"Scenario testing completed!")
    print(f"{'='*80}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
