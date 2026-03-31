#!/usr/bin/env python
"""
Enhanced comprehensive edge case testing with new features.
Tests Groq API, Ollama fallback, scope detection, prompt engineering, and more.
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_handler import LLMHandler
from emotion_detector import EmotionDetector
from intent_model import IntentClassifier
from database import ChatbotDatabase
from scope_detector import ScopeDetector
from prompt_engineering import PromptEngineer


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result with colors."""
    status = (f"{Colors.GREEN}[PASS]{Colors.RESET}" if passed
              else f"{Colors.RED}[FAIL]{Colors.RESET}")
    print(f"  {status}: {name}")
    if details:
        print(f"         {Colors.YELLOW}{details}{Colors.RESET}")


class TestSuite:
    """Comprehensive test suite."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def add_result(self, name: str, passed: bool, details: str = ""):
        """Record test result."""
        self.tests.append((name, passed, details))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print_test(name, passed, details)

    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Test Summary: {Colors.GREEN}{self.passed} passed{Colors.RESET}, "
              f"{Colors.RED}{self.failed} failed{Colors.RESET}, "
              f"Total: {total}")
        print(f"{'='*70}\n")
        return self.failed == 0


def test_groq_api():
    """Test Groq API connectivity and model availability."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"GROQ API CONNECTIVITY TESTS")
    print(f"{'='*70}{Colors.RESET}\n")

    suite = TestSuite()

    # Test 1: Check if API key is available
    api_key = os.getenv('GROQ_API_KEY')
    suite.add_result("API Key Available",
                     api_key is not None and len(api_key) > 0,
                     f"Key length: {len(api_key) if api_key else 0}")

    if not api_key:
        print(f"{Colors.RED}ERROR: GROQ_API_KEY not set.{Colors.RESET}")
        return suite

    # Test 2: Test basic API connectivity
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": "Say 'OK'"}],
        "max_tokens": 10
    }

    try:
        start = time.time()
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            suite.add_result("Groq API Connectivity", True,
                           f"Response time: {elapsed:.2f}s")
        else:
            suite.add_result("Groq API Connectivity", False,
                           f"Status: {response.status_code}")
    except Exception as e:
        suite.add_result("Groq API Connectivity", False, str(e)[:100])

    # Test 3: Test LLMHandler initialization
    try:
        handler = LLMHandler()
        suite.add_result("LLMHandler Initialization", True,
                       f"Model: {handler.groq_model}")
    except Exception as e:
        suite.add_result("LLMHandler Initialization", False, str(e)[:100])

    # Test 4: Test response generation with scope detection
    try:
        handler = LLMHandler()
        detector = ScopeDetector()

        is_in_scope, scope_reason, scope_conf = detector.is_in_scope(
            "What are the fees?")

        response = handler.generate_response(
            user_input="What are the fees?",
            intent="fees",
            confidence=0.9,
            emotion="neutral"
        )

        has_text = 'response' in response and len(response['response']) > 0
        has_source = 'source' in response
        has_scope = 'is_in_scope' in response

        suite.add_result("LLMHandler Response with Scope Detection",
                       has_text and has_source and has_scope,
                       f"Source: {response.get('source', 'unknown')}")
    except Exception as e:
        suite.add_result("LLMHandler Response with Scope Detection",
                       False, str(e)[:100])

    return suite


def test_scope_detection():
    """Test scope detection with edge cases."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"SCOPE DETECTION TESTS")
    print(f"{'='*70}{Colors.RESET}\n")

    suite = TestSuite()

    try:
        detector = ScopeDetector()
        suite.add_result("ScopeDetector Initialization", True)
    except Exception as e:
        suite.add_result("ScopeDetector Initialization", False, str(e)[:100])
        return suite

    # Test domain keywords
    test_cases = [
        ("What are the fees?", True, "fees keyword"),
        ("Tell me about placements", True, "placement keyword"),
        ("What is quantum physics?", False, "out-of-domain topic"),
        ("How do I cook pasta?", False, "cooking unrelated"),
        ("Tell me about exams", True, "exams keyword"),
        ("Faculty information", True, "faculty keyword"),
        ("", True, "empty query"),
        ("123 456", False, "numbers only"),
    ]

    for query, expected_scope, description in test_cases:
        try:
            is_in_scope, reason, conf = detector.is_in_scope(query)
            passed = is_in_scope == expected_scope
            suite.add_result(f"Scope Detection: {description}",
                           passed,
                           f"Scope: {is_in_scope}, Reason: {reason[:40]}")
        except Exception as e:
            suite.add_result(f"Scope Detection: {description}",
                           False, str(e)[:100])

    return suite


def test_prompt_engineering():
    """Test prompt engineering with different signals."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"PROMPT ENGINEERING TESTS")
    print(f"{'='*70}{Colors.RESET}\n")

    suite = TestSuite()

    try:
        engineer = PromptEngineer()
        suite.add_result("PromptEngineer Initialization", True)
    except Exception as e:
        suite.add_result("PromptEngineer Initialization", False, str(e)[:100])
        return suite

    # Test emotion-aware prompts
    emotions = ["stressed", "confused", "angry", "happy", "neutral"]
    for emotion in emotions:
        try:
            prompt = engineer.build_system_prompt(
                intent="fees",
                is_in_scope=True,
                emotion=emotion
            )
            has_content = len(prompt) > 50
            suite.add_result(f"Emotion-Aware Prompt: {emotion}",
                           has_content,
                           f"Prompt length: {len(prompt)}")
        except Exception as e:
            suite.add_result(f"Emotion-Aware Prompt: {emotion}",
                           False, str(e)[:100])

    # Test different scopes
    for in_scope in [True, False]:
        try:
            prompt = engineer.build_system_prompt(
                intent="general",
                is_in_scope=in_scope,
                emotion="neutral"
            )
            scope_label = "In-Scope" if in_scope else "Out-of-Scope"
            suite.add_result(f"Scope-Aware Prompt: {scope_label}",
                           len(prompt) > 50,
                           f"Prompt length: {len(prompt)}")
        except Exception as e:
            suite.add_result(f"Scope-Aware Prompt: {scope_label}",
                           False, str(e)[:100])

    return suite


def test_emotion_detector():
    """Test emotion detector edge cases."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"EMOTION DETECTOR TESTS")
    print(f"{'='*70}{Colors.RESET}\n")

    suite = TestSuite()

    try:
        detector = EmotionDetector()
        suite.add_result("EmotionDetector Initialization", True)
    except Exception as e:
        suite.add_result("EmotionDetector Initialization", False, str(e)[:100])
        return suite

    # Test cases
    test_cases = [
        ("I'm so happy!", "positive emotion"),
        ("This is terrible!", "negative emotion"),
        ("I'm confused", "confusion"),
        ("The weather is nice", "neutral"),
        ("", "empty string"),
        ("!!!???***", "special characters"),
        ("hello " * 100, "very long text"),
    ]

    for text, description in test_cases:
        try:
            result = detector.detect_emotion(text)
            emotion = result.get('emotion')
            confidence = result.get('confidence', 0)
            is_valid = emotion in [
                'happiness', 'sadness', 'anger', 'fear', 'confusion',
                'neutral', 'happy', 'stressed', 'angry', 'sad', 'confused']
            suite.add_result(f"Emotion Detection: {description}",
                           is_valid,
                           f"Emotion: {emotion}, Conf: {confidence:.2f}")
        except Exception as e:
            suite.add_result(f"Emotion Detection: {description}",
                           False, str(e)[:100])

    return suite


def test_intent_classifier():
    """Test intent classifier edge cases."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"INTENT CLASSIFIER TESTS")
    print(f"{'='*70}{Colors.RESET}\n")

    suite = TestSuite()

    try:
        classifier = IntentClassifier()
        suite.add_result("IntentClassifier Initialization", True)
    except Exception as e:
        suite.add_result("IntentClassifier Initialization",
                       False, str(e)[:100])
        return suite

    # Test cases
    test_cases = [
        ("What are the fees?", "fees", "fee inquiry"),
        ("Tell me about placements", "placement", "placement query"),
        ("Can I get admitted?", "admission", "admission question"),
        ("", "general", "empty input"),
        ("???", "general", "special characters only"),
        ("a" * 1000, None, "very long input"),
        ("123 456 789", "general", "numbers only"),
    ]

    for text, expected_intent, description in test_cases:
        try:
            result = classifier.predict(text)
            intent = result.get('intent')
            confidence = result.get('confidence', 0)
            is_valid = (intent is not None and
                       0 <= confidence <= 1)
            suite.add_result(f"Intent Classification: {description}",
                           is_valid,
                           f"Intent: {intent}, Conf: {confidence:.2f}")
        except Exception as e:
            suite.add_result(f"Intent Classification: {description}",
                           False, str(e)[:100])

    return suite


def test_database_operations():
    """Test database operations with new fields."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"DATABASE OPERATIONS TESTS")
    print(f"{'='*70}{Colors.RESET}\n")

    suite = TestSuite()

    # Use temporary database for testing
    db_path = "test_temp_edge_cases.db"
    try:
        db = ChatbotDatabase(db_path=db_path)
        suite.add_result("Database Initialization", True)
    except Exception as e:
        suite.add_result("Database Initialization", False, str(e)[:100])
        return suite

    # Test logging with new fields
    try:
        success = db.log_interaction(
            user_input="What are fees?",
            intent="fees",
            confidence=0.95,
            emotion="neutral",
            response="The fees are...",
            response_time=1.5,
            llm_source="groq",
            is_in_scope=True,
            should_clarify=False,
            scope_reason="Matched fee keywords",
            session_id="test_session"
        )
        suite.add_result("Log Interaction with Enhanced Fields", success)
    except Exception as e:
        suite.add_result("Log Interaction with Enhanced Fields",
                       False, str(e)[:100])

    # Test retrieving logs
    try:
        logs = db.get_logs(limit=10)
        has_content = len(logs) > 0
        has_new_fields = (len(logs) > 0 and
                         any('is_in_scope' in log for log in logs))
        suite.add_result("Retrieve Logs with New Fields",
                       has_content and has_new_fields,
                       f"Retrieved {len(logs)} logs")
    except Exception as e:
        suite.add_result("Retrieve Logs with New Fields",
                       False, str(e)[:100])

    # Test analytics
    try:
        analytics = db.get_analytics_summary()
        has_data = len(analytics) > 0
        suite.add_result("Get Analytics Summary", has_data,
                       f"Analytics keys: {len(analytics)}")
    except Exception as e:
        suite.add_result("Get Analytics Summary", False, str(e)[:100])

    # Cleanup
    try:
        import os
        if os.path.exists(db_path):
            os.remove(db_path)
    except:
        pass

    return suite


def main():
    """Run all test suites."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"COMPREHENSIVE EDGE CASE TESTS")
    print(f"Including New Features: Scope Detection & Prompt Engineering")
    print(f"{'='*70}{Colors.RESET}\n")

    results = []

    # Run all test suites
    results.append(test_groq_api())
    results.append(test_scope_detection())
    results.append(test_prompt_engineering())
    results.append(test_emotion_detector())
    results.append(test_intent_classifier())
    results.append(test_database_operations())

    # Overall summary
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_tests = total_passed + total_failed

    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}{Colors.RESET}\n")
    print(f"  {Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {total_failed}{Colors.RESET}")
    print(f"  Total: {total_tests}\n")

    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
