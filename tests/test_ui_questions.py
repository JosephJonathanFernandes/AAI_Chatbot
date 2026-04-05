#!/usr/bin/env python3
"""
Enhanced CLI Test Script for UI Test Questions
Tests with scope detection, prompt engineering, and multi-signal control
"""

import sys
import json
import time
from colorama import Fore, Back, Style, init
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize colorama
init(autoreset=True)

# Import our modules
from intent_model import IntentClassifier
from emotion_detector import EmotionDetector
from llm_handler import LLMHandler
from context_manager import ConversationContext
from database import ChatbotDatabase
from scope_detector import ScopeDetector
from prompt_engineering import PromptEngineer


class UIQuestionsTesters:
    """Test all UI question categories with enhanced features."""

    def __init__(self):
        """Initialize all components."""
        print(f"{Fore.CYAN}Initializing chatbot components...{Style.RESET_ALL}")

        self.intent_classifier = IntentClassifier()
        self.emotion_detector = EmotionDetector()
        self.llm_handler = LLMHandler()
        self.context = ConversationContext()
        self.database = ChatbotDatabase()
        self.scope_detector = ScopeDetector()
        self.prompt_engineer = PromptEngineer()

        # Train intent classifier if needed
        if not self.intent_classifier.is_trained:
            print(f"{Fore.YELLOW}Training intent classifier...{Style.RESET_ALL}")
            self.intent_classifier.train()

        print(f"{Fore.GREEN}[OK] All components initialized!{Style.RESET_ALL}\n")

        # Define all test question categories
        self.question_sets = {
            "1": {
                "name": "FEES",
                "questions": [
                    "What are the tuition fees?",
                    "How much does the course cost?",
                    "Is there financial aid available?",
                    "Tell me about scholarship information",
                    "What are the payment options?",
                    "When is the fee payment deadline?",
                    "How do I pay fees?",
                    "Fee structure",
                    "Cost of admission"
                ]
            },
            "2": {
                "name": "EXAMS",
                "questions": [
                    "When is the exam scheduled?",
                    "What's the exam syllabus?",
                    "How to prepare for midterm exams?",
                    "What is the pass mark?",
                    "When are final exams?",
                    "What's my exam center?",
                    "Exam schedule",
                    "Test dates",
                    "How long does each exam take?"
                ]
            },
            "3": {
                "name": "TIMETABLE",
                "questions": [
                    "What's the class schedule?",
                    "When are the lectures?",
                    "What are the lab timings?",
                    "When does college start?",
                    "What are the break timings?",
                    "When is my class?",
                    "Class timings",
                    "Timetable"
                ]
            },
            "4": {
                "name": "PLACEMENTS",
                "questions": [
                    "Tell me about placements",
                    "What's the average salary?",
                    "Which companies visit campus?",
                    "What's the placement statistics?",
                    "Are there internship opportunities?",
                    "How to get placed?",
                    "Placement training",
                    "Recruitment process"
                ]
            },
            "5": {
                "name": "FACULTY",
                "questions": [
                    "Who are the faculty members?",
                    "What are faculty qualifications?",
                    "Faculty office hours?",
                    "How to contact faculty?",
                    "Faculty research areas",
                    "Department information"
                ]
            },
            "6": {
                "name": "LIBRARY",
                "questions": [
                    "What are library timings?",
                    "How to borrow books?",
                    "What's the library catalog?",
                    "Are there reading rooms?",
                    "Library membership",
                    "Can I access e-resources?"
                ]
            },
            "7": {
                "name": "ADMISSION",
                "questions": [
                    "How do I apply for admission?",
                    "What are the eligibility criteria?",
                    "What documents are required?",
                    "When is the admission deadline?",
                    "How much is the application fee?",
                    "Can I appeal an admission decision?"
                ]
            },
            "8": {
                "name": "HOSTEL",
                "questions": [
                    "What about hostel facilities?",
                    "How much is hostel fee?",
                    "Are meals included?",
                    "Hostel application process",
                    "Room sharing information",
                    "Hostel rules and regulations"
                ]
            }
        }

    def test_single_question(self, question: str, category: str) -> dict:
        """
        Test a single question with all new features.

        Args:
            question (str): The question to test
            category (str): Category name for tracking

        Returns:
            dict: Test results
        """
        result = {
            "question": question,
            "category": category,
            "scope_detected": False,
            "intent": None,
            "confidence": 0,
            "emotion": None,
            "response_generated": False,
            "response_text": "",
            "time_taken": 0,
            "logged_to_db": False,
            "should_clarify": False
        }

        try:
            start_time = time.time()

            # 1. Scope Detection
            is_in_scope, scope_reason, scope_conf = \
                self.scope_detector.is_in_scope(question)
            result["scope_detected"] = is_in_scope
            result["scope_reason"] = scope_reason

            # 2. Intent Classification
            intent, confidence = self.intent_classifier.predict(question)
            result["intent"] = intent or 'general'
            result["confidence"] = confidence or 0.0

            # 3. Emotion Detection
            emotion_result = self.emotion_detector.detect_emotion(question)
            result["emotion"] = emotion_result.get('emotion', 'neutral')

            # 4. Build Enhanced Prompt
            system_prompt = self.prompt_engineer.build_system_prompt(
                intent=result["intent"],
                is_in_scope=result["scope_detected"],
                emotion=result["emotion"],
                confidence=result["confidence"]
            )

            # 5. Generate Response
            response = self.llm_handler.generate_response(
                user_input=question,
                intent=result["intent"],
                confidence=result["confidence"],
                emotion=result["emotion"],
                conversation_history=self.context.get_history()
            )

            result["response_generated"] = True
            result["response_text"] = response.get('response', '')[:200]
            result["response_time"] = response.get('time', 0)
            result["should_clarify"] = response.get('should_clarify', False)

            # 6. Log to Database
            logged = self.database.log_interaction(
                user_input=question,
                intent=result["intent"],
                confidence=result["confidence"],
                emotion=result["emotion"],
                response=response.get('response', ''),
                response_time=response.get('time', 0),
                llm_source=response.get('source', 'groq'),
                is_in_scope=result["scope_detected"],
                should_clarify=response.get('should_clarify', False),
                scope_reason=scope_reason,
                session_id="ui_test_session"
            )

            result["logged_to_db"] = logged
            result["time_taken"] = time.time() - start_time

        except Exception as e:
            result["error"] = str(e)
            result["response_text"] = f"Error: {str(e)[:100]}"

        return result

    def test_category(self, category_key: str):
        """Test all questions in a category."""
        if category_key not in self.question_sets:
            print(f"{Fore.RED}Invalid category!{Style.RESET_ALL}")
            return

        category = self.question_sets[category_key]
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{Fore.CYAN}Testing Category: {category['name']}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

        results = []
        passed = 0
        failed = 0

        for i, question in enumerate(category['questions'], 1):
            result = self.test_single_question(question, category['name'])

            # Display result
            status = (
                f"{Fore.GREEN}✓{Style.RESET_ALL}" if
                (result["response_generated"] and result["logged_to_db"])
                else f"{Fore.RED}✗{Style.RESET_ALL}")

            print(f"{status} Q{i}: {question[:60]}")
            print(f"   Intent: {Fore.YELLOW}{result['intent']}"
                  f"{Style.RESET_ALL} "
                  f"Conf: {result['confidence']:.2f}, "
                  f"Scope: {Fore.GREEN if result['scope_detected'] else Fore.RED}"
                  f"{result['scope_detected']}{Style.RESET_ALL}")
            print(f"   Emotion: {result['emotion']}, "
                  f"Clarify: {result['should_clarify']}")
            print(f"   Response: {result['response_text'][:60]}...")
            print(f"   Time: {result['time_taken']:.2f}s, "
                  f"Logged: {Fore.GREEN if result['logged_to_db'] else Fore.RED}"
                  f"{result['logged_to_db']}{Style.RESET_ALL}\n")

            results.append(result)

            if result["response_generated"] and result["logged_to_db"]:
                passed += 1
            else:
                failed += 1

        # Category summary
        total = passed + failed
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{category['name']} Results: "
              f"{Fore.GREEN}{passed} passed{Style.RESET_ALL}, "
              f"{Fore.RED}{failed} failed{Style.RESET_ALL} "
              f"(Total: {total})")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

        return results

    def test_all_categories(self):
        """Test all categories."""
        print(f"\n{Style.BRIGHT}{Fore.CYAN}COMPREHENSIVE UI QUESTION TESTING{Style.RESET_ALL}")
        print(f"{Fore.CYAN}With Scope Detection & Multi-Signal Control{Style.RESET_ALL}\n")

        all_results = []
        total_passed = 0
        total_failed = 0

        for category_key in sorted(self.question_sets.keys()):
            results = self.test_category(category_key)
            if results:
                all_results.extend(results)
                for r in results:
                    if r["response_generated"] and r["logged_to_db"]:
                        total_passed += 1
                    else:
                        total_failed += 1

        # Overall summary
        total = total_passed + total_failed
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{'='*80}")
        print(f"OVERALL TEST RESULTS")
        print(f"{'='*80}{Style.RESET_ALL}\n")
        print(f"  {Fore.GREEN}Passed: {total_passed}{Style.RESET_ALL}")
        print(f"  {Fore.RED}Failed: {total_failed}{Style.RESET_ALL}")
        print(f"  Total: {total}")

        if total > 0:
            success_rate = 100 * total_passed / total
            print(f"  Success Rate: {success_rate:.1f}%\n")
        else:
            print()

        return all_results

    def interactive_mode(self):
        """Interactive testing mode."""
        print(f"\n{Style.BRIGHT}{Fore.CYAN}INTERACTIVE TEST MODE{Style.RESET_ALL}\n")
        print("Categories:")
        for key, category in self.question_sets.items():
            print(f"  {key}. {category['name']}")
        print(f"  0. Test All")
        print(f"  Q. Quit\n")

        while True:
            choice = input(f"{Fore.CYAN}Select category (0-8 or Q): "
                          f"{Style.RESET_ALL}").strip().upper()

            if choice == 'Q':
                break
            elif choice == '0':
                self.test_all_categories()
            elif choice in self.question_sets:
                self.test_category(choice)
            else:
                print(f"{Fore.RED}Invalid choice!{Style.RESET_ALL}\n")


def main():
    """Main entry point."""
    tester = UIQuestionsTesters()

    print(f"\n{Fore.CYAN}Choose mode:{Style.RESET_ALL}")
    print(f"1. Interactive Mode")
    print(f"2. Test All Categories")
    print(f"3. Test Single Category\n")

    mode = input(f"{Fore.CYAN}Select mode (1-3): {Style.RESET_ALL}").strip()

    if mode == '1':
        tester.interactive_mode()
    elif mode == '2':
        tester.test_all_categories()
    elif mode == '3':
        cat = input(f"{Fore.CYAN}Select category (1-8): "
                   f"{Style.RESET_ALL}").strip()
        tester.test_category(cat)
    else:
        print(f"{Fore.RED}Invalid mode!{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
        sys.exit(0)
