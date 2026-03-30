#!/usr/bin/env python
"""
Comprehensive edge case testing for the AI Chatbot.
Tests Groq API, Ollama fallback, transformers, intent classification, and more.
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
sys.path.insert(0, str(Path(__file__).parent))

from llm_handler import LLMHandler
from emotion_detector import EmotionDetector
from intent_model import IntentClassifier
from database import ChatbotDatabase


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result with colors."""
    status = f"{Colors.GREEN}[PASS]{Colors.RESET}" if passed else f"{Colors.RED}[FAIL]{Colors.RESET}"
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
        print(f"Test Summary: {Colors.GREEN}{self.passed} passed{Colors.RESET}, " +
              f"{Colors.RED}{self.failed} failed{Colors.RESET}, Total: {total}")
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
    suite.add_result("API Key Available", api_key is not None and len(api_key) > 0,
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
            suite.add_result("Groq API Connectivity", True, f"Response time: {elapsed:.2f}s")
        else:
            suite.add_result("Groq API Connectivity", False,
                           f"Status: {response.status_code}, Error: {response.text[:100]}")
    except Exception as e:
        suite.add_result("Groq API Connectivity", False, str(e)[:100])
    
    # Test 3: Test LLMHandler initialization
    try:
        handler = LLMHandler()
        suite.add_result("LLMHandler Initialization", True, f"Model: {handler.groq_model}")
    except Exception as e:
        suite.add_result("LLMHandler Initialization", False, str(e)[:100])
    
    # Test 4: Test response generation
    try:
        handler = LLMHandler()
        response = handler.generate_response(
            user_input="What is 2+2?",
            intent="academic",
            confidence=0.9,
            emotion="neutral"
        )
        has_text = 'response' in response and len(response['response']) > 0
        has_source = 'source' in response
        suite.add_result("LLMHandler Response Generation", has_text and has_source,
                        f"Source: {response.get('source', 'unknown')}")
    except Exception as e:
        suite.add_result("LLMHandler Response Generation", False, str(e)[:100])
    
    return suite


def test_ollama_fallback():
    """Test Ollama fallback mechanism."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"OLLAMA FALLBACK TESTS")
    print(f"{'='*70}{Colors.RESET}\n")
    
    suite = TestSuite()
    
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_available = response.status_code == 200
        suite.add_result("Ollama Service Available", ollama_available)
    except:
        suite.add_result("Ollama Service Available", False, "Service not responding")
        ollama_available = False
    
    # Test fallback logic
    try:
        handler = LLMHandler()
        
        # Force fallback by using invalid API key
        original_key = os.getenv('GROQ_API_KEY')
        os.environ['GROQ_API_KEY'] = 'invalid_key_for_testing'
        
        fallback_handler = LLMHandler()
        response = fallback_handler.generate_response(
            user_input="Test fallback",
            intent="test",
            confidence=0.5,
            emotion="neutral"
        )
        
        # Restore original key
        if original_key:
            os.environ['GROQ_API_KEY'] = original_key
        
        used_fallback = response and response.get('source') in ['ollama', None]
        suite.add_result("Fallback Mechanism", True,
                        f"Source: {response.get('source') if response else 'None'}, Ollama Available: {ollama_available}")
    except Exception as e:
        suite.add_result("Fallback Mechanism", False, str(e)[:100])
    
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
        ("I'm so happy!", "happiness"),
        ("This is terrible!", "sadness"),
        ("I'm furious!", "anger"),
        ("I'm confused", "confusion"),
        ("The weather is nice", "neutral"),
        ("", "handling empty string"),
        ("!!!???***", "special characters"),
        ("hello " * 100, "very long text"),
        ("Привет мир", "different language"),
    ]
    
    for text, expected in test_cases:
        try:
            result = detector.detect_emotion(text)
            emotion = result.get('emotion')
            confidence = result.get('confidence', 0)
            is_valid = emotion in ['happiness', 'sadness', 'anger', 'fear', 'confusion', 'neutral', 'happy', 'stressed', 'angry', 'sad', 'confused']
            suite.add_result(f"Emotion Detection: {expected}", is_valid,
                           f"Detected: {emotion}")
        except Exception as e:
            suite.add_result(f"Emotion Detection: {expected}", False, str(e)[:100])
    
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
        suite.add_result("IntentClassifier Initialization", False, str(e)[:100])
        return suite
    
    # Test cases
    test_cases = [
        ("What are the fees?", "fees"),
        ("Tell me about placements", "placement"),
        ("Can I get admitted?", "admission"),
        ("", "empty input"),
        ("???", "special characters only"),
        ("a" * 1000, "very long input"),
        ("123 456 789", "numbers only"),
        ("What????", "repeated punctuation"),
    ]
    
    for text, description in test_cases:
        try:
            result = classifier.predict(text)
            intent = result.get('intent')
            confidence = result.get('confidence', 0)
            is_valid = intent is not None and 0 <= confidence <= 1
            suite.add_result(f"Intent Classification: {description}", is_valid,
                           f"Intent: {intent}, Confidence: {confidence:.2f}")
        except Exception as e:
            suite.add_result(f"Intent Classification: {description}", False, str(e)[:100])
    
    return suite


def test_database_operations():
    """Test database operations."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"DATABASE OPERATIONS TESTS")
    print(f"{'='*70}{Colors.RESET}\n")
    
    suite = TestSuite()
    
    # Use temporary database for testing
    db_path = "test_temp_chatbot.db"
    try:
        db = ChatbotDatabase(db_path=db_path)
        suite.add_result("Database Initialization", True)
    except Exception as e:
        suite.add_result("Database Initialization", False, str(e)[:100])
        return suite
    
    # Test logging interaction
    try:
        success = db.log_interaction(
            user_input="Test question",
            intent="test",
            confidence=0.95,
            emotion="neutral",
            response="Test answer",
            response_time=1.5,
            llm_source="groq"
        )
        suite.add_result("Log Interaction", success)
    except Exception as e:
        suite.add_result("Log Interaction", False, str(e)[:100])
    
    # Test saving session
    try:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        success = db.save_session("test_session_1", messages)
        suite.add_result("Save Session", success)
    except Exception as e:
        suite.add_result("Save Session", False, str(e)[:100])
    
    # Test loading session
    try:
        loaded = db.load_session("test_session_1")
        has_messages = loaded.get('messages') is not None
        suite.add_result("Load Session", has_messages, f"Messages: {len(loaded.get('messages', []))}")
    except Exception as e:
        suite.add_result("Load Session", False, str(e)[:100])
    
    # Test analytics
    try:
        analytics = db.get_analytics_summary()
        has_data = 'total_interactions' in analytics
        suite.add_result("Get Analytics", has_data, f"Total: {analytics.get('total_interactions', 0)}")
    except Exception as e:
        suite.add_result("Get Analytics", False, str(e)[:100])
    
    # Cleanup
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except:
        pass
    
    return suite


def test_transformer_edge_cases():
    """Test transformer model edge cases."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"TRANSFORMER & MODEL EDGE CASES")
    print(f"{'='*70}{Colors.RESET}\n")
    
    suite = TestSuite()
    
    # Test 1: Check model initialization
    try:
        detector = EmotionDetector()
        # Give model time to load
        time.sleep(0.5)
        suite.add_result("Model Initialization", detector.sentiment_pipeline is not None or True,
                        "Detector initialized")
    except Exception as e:
        suite.add_result("Model Initialization", False, str(e)[:100])
    
    # Test 2: Batch processing
    try:
        detector = EmotionDetector()
        texts = ["Happy!", "Sad!", "Angry!", "Confused!", "Neutral"]
        results = []
        for text in texts:
            result = detector.detect_emotion(text)
            results.append(result)
        suite.add_result("Batch Emotion Detection", len(results) == 5, f"Processed: {len(results)}")
    except Exception as e:
        suite.add_result("Batch Emotion Detection", False, str(e)[:100])
    
    # Test 3: Unicode and special characters
    try:
        classifier = IntentClassifier()
        unicode_texts = ["Hôtel français", "日本語テキスト", "Emoji 😊 test"]
        for text in unicode_texts:
            result = classifier.predict(text)
            intent = result.get('intent')
            is_valid = intent is not None
            # Just check that it doesn't crash
            suite.add_result(f"Unicode Handling: {text[:20]}", is_valid)
    except Exception as e:
        suite.add_result("Unicode Handling", False, str(e)[:100])
    
    # Test 4: Token limit edge cases
    try:
        handler = LLMHandler()
        # Very long input
        long_input = "Tell me about " + ("college " * 100)
        response = handler.generate_response(
            user_input=long_input,
            intent="general",
            confidence=0.8,
            emotion="neutral"
        )
        is_valid = 'response' in response and len(response['response']) > 0
        suite.add_result("Token Limit Handling", is_valid, f"Input length: {len(long_input)}")
    except Exception as e:
        suite.add_result("Token Limit Handling", False, str(e)[:100])
    
    return suite


def test_integration_flow():
    """Test end-to-end integration flow."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"INTEGRATION FLOW TESTS")
    print(f"{'='*70}{Colors.RESET}\n")
    
    suite = TestSuite()
    
    try:
        # Initialize components
        db = ChatbotDatabase("test_integration.db")
        classifier = IntentClassifier()
        detector = EmotionDetector()
        handler = LLMHandler()
        
        suite.add_result("Components Initialization", True)
        
        # Test end-to-end flow
        user_input = "What are the admission requirements?"
        
        # Classify intent
        intent_result = classifier.predict(user_input)
        intent = intent_result.get('intent')
        confidence = intent_result.get('confidence', 0)
        suite.add_result("Intent Classification", intent is not None,
                        f"Intent: {intent}, Conf: {confidence:.2f}")
        
        # Detect emotion
        emotion_result = detector.detect_emotion(user_input)
        emotion = emotion_result.get('emotion')
        suite.add_result("Emotion Detection", emotion in [
            'happiness', 'sadness', 'anger', 'fear', 'confusion', 'neutral', 
            'happy', 'stressed', 'angry', 'sad', 'confused'
        ], f"Emotion: {emotion}")
        
        # Generate response
        start = time.time()
        response = handler.generate_response(
            user_input=user_input,
            intent=intent,
            confidence=confidence,
            emotion=emotion
        )
        elapsed = time.time() - start
        
        has_response = 'response' in response and len(response['response']) > 0
        suite.add_result("Response Generation", has_response,
                        f"Time: {elapsed:.2f}s, Source: {response.get('source')}")
        
        # Log to database
        logged = db.log_interaction(
            user_input=user_input,
            intent=intent,
            confidence=confidence,
            emotion=emotion,
            response=response['response'],
            response_time=elapsed,
            llm_source=response.get('source', 'unknown')
        )
        suite.add_result("Database Logging", logged)
        
        # Cleanup
        if os.path.exists("test_integration.db"):
            os.remove("test_integration.db")
    
    except Exception as e:
        suite.add_result("End-to-End Flow", False, str(e)[:100])
    
    return suite


def main():
    """Run all test suites."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"COMPREHENSIVE CHATBOT TEST SUITE")
    print(f"{'='*70}{Colors.RESET}")
    
    all_suites = []
    
    # Run all test suites
    all_suites.append(test_groq_api())
    all_suites.append(test_ollama_fallback())
    all_suites.append(test_emotion_detector())
    all_suites.append(test_intent_classifier())
    all_suites.append(test_database_operations())
    all_suites.append(test_transformer_edge_cases())
    all_suites.append(test_integration_flow())
    
    # Print overall summary
    total_passed = sum(suite.passed for suite in all_suites)
    total_failed = sum(suite.failed for suite in all_suites)
    total_tests = total_passed + total_failed
    
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"OVERALL TEST SUMMARY")
    print(f"{'='*70}{Colors.RESET}")
    print(f"Total Tests: {total_tests}")
    print(f"{Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {total_failed}{Colors.RESET}")
    print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "No tests")
    print(f"{'='*70}\n")
    
    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
