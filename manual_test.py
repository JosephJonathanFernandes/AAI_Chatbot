#!/usr/bin/env python3
"""
Interactive Manual Testing Script for College AI Chatbot

This script allows you to test the chatbot interactively without Streamlit.
Useful for quick validation and debugging.
"""

import sys
import time
from colorama import Fore, Back, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# Import our modules
from intent_model import IntentClassifier
from emotion_detector import EmotionDetector
from llm_handler import LLMHandler
from context_manager import ConversationContext
from database import ChatbotDatabase


class ManualTester:
    """Interactive manual testing for the chatbot."""
    
    def __init__(self):
        """Initialize all components."""
        print(f"{Fore.CYAN}Initializing chatbot components...{Style.RESET_ALL}")
        
        self.intent_classifier = IntentClassifier()
        self.emotion_detector = EmotionDetector()
        self.llm_handler = LLMHandler()
        self.context = ConversationContext()
        self.database = ChatbotDatabase()
        
        # Train intent classifier if needed
        if not self.intent_classifier.is_trained:
            print(f"{Fore.YELLOW}Training intent classifier...{Style.RESET_ALL}")
            self.intent_classifier.train()
        
        self.conversation_count = 0
        print(f"{Fore.GREEN}✓ All components initialized!{Style.RESET_ALL}\n")
    
    def display_menu(self):
        """Display main menu."""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}COLLEGE AI ASSISTANT - MANUAL TESTING MENU{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print("1. Test Intent Classification")
        print("2. Test Emotion Detection")
        print("3. Test Full Conversation Flow")
        print("4. Run Test Scenarios")
        print("5. Check Database")
        print("6. Performance Analysis")
        print("7. Exit")
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
    
    def test_intent_classification(self):
        """Test intent classification with predefined queries."""
        print(f"\n{Fore.CYAN}TESTING INTENT CLASSIFICATION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
        
        test_queries = {
            "admission": "How do I apply to your college?",
            "fees": "What are the engineering fees?",
            "placements": "Tell me about placements",
            "exams": "When are the final exams?",
            "library": "What are library hours?",
            "timetable": "What's my class schedule?",
            "faculty": "Who is the HOD of engineering?",
            "departments": "What departments do you have?",
            "holidays": "When are college holidays?",
            "greeter": "Hi there!",
            "gratitude": "Thank you so much!"
        }
        
        correct = 0
        total = len(test_queries)
        
        for expected_intent, query in test_queries.items():
            result = self.intent_classifier.predict(query)
            intent = result['intent']
            confidence = result['confidence']
            
            is_correct = intent == expected_intent
            correct += is_correct
            
            status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if is_correct else f"{Fore.RED}✗{Style.RESET_ALL}"
            print(f"{status} Query: {query[:40]:40} | Intent: {intent:15} | Conf: {confidence:.2%}")
        
        accuracy = (correct / total) * 100
        print(f"\n{'='*60}")
        print(f"Accuracy: {Fore.GREEN if accuracy >= 80 else Fore.YELLOW}{accuracy:.1f}%{Style.RESET_ALL} ({correct}/{total})")
    
    def test_emotion_detection(self):
        """Test emotion detection with predefined inputs."""
        print(f"\n{Fore.CYAN}TESTING EMOTION DETECTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
        
        test_cases = {
            "happy": "I'm so happy! This is amazing!",
            "stressed": "I'm worried about my exams",
            "angry": "This is terrible and awful!",
            "confused": "I don't understand this",
            "sad": "I'm feeling really down",
            "neutral": "What is the course schedule?"
        }
        
        for expected_emotion, text in test_cases.items():
            result = self.emotion_detector.detect_emotion(text)
            emotion = result['emotion']
            confidence = result['confidence']
            
            status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if emotion == expected_emotion else f"{Fore.BLUE}~{Style.RESET_ALL}"
            print(f"{status} Text: {text[:45]:45} | Emotion: {emotion:10} | Conf: {confidence:.2%}")
    
    def test_full_conversation(self):
        """Test full conversation flow interactively."""
        print(f"\n{Fore.CYAN}FULL CONVERSATION FLOW TEST{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
        print("Enter your message (or 'quit' to exit):\n")
        
        while True:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                print(f"{Fore.YELLOW}Please enter a message.{Style.RESET_ALL}")
                continue
            
            # Process the input
            start_time = time.time()
            
            # Step 1: Classify intent
            intent_result = self.intent_classifier.predict(user_input)
            
            # Step 2: Detect emotion
            emotion_result = self.emotion_detector.detect_emotion(user_input)
            
            # Step 3: Update context
            self.context.add_to_history(user_input, "user")
            
            elapsed = time.time() - start_time
            self.conversation_count += 1
            
            # Display results
            print(f"\n{Fore.CYAN}Analysis:{Style.RESET_ALL}")
            print(f"  Intent: {intent_result['intent']} ({intent_result['confidence']:.2%} confident)")
            print(f"  Emotion: {emotion_result['emotion']} ({emotion_result['confidence']:.2%} confident)")
            print(f"  Processing time: {elapsed:.2f}s")
            
            # Log to database
            self.database.save_log(
                user_input=user_input,
                bot_response="[Test Response]",
                intent=intent_result['intent'],
                confidence=intent_result['confidence'],
                emotion=emotion_result['emotion'],
                llm_source='test',
                response_time=elapsed
            )
            
            print()
    
    def run_test_scenarios(self):
        """Run predefined test scenarios."""
        print(f"\n{Fore.CYAN}RUNNING TEST SCENARIOS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
        
        scenarios = [
            ("Scenario 1", "Fresh student inquiry", "I'm new and want to apply. What do I need to know?"),
            ("Scenario 2", "Fee inquiry", "How much is the engineering course per year?"),
            ("Scenario 3", "Placement info", "Which companies recruit from your campus?"),
            ("Scenario 4", "Out-of-domain", "Can you teach me to code in Python?"),
            ("Scenario 5", "Exam schedule", "When are the summer exams?"),
        ]
        
        for scenario_num, description, query in scenarios:
            print(f"\n{Fore.YELLOW}{scenario_num}: {description}{Style.RESET_ALL}")
            print(f"Query: {query}")
            
            start_time = time.time()
            result = self.intent_classifier.predict(query)
            emotion = self.emotion_detector.detect_emotion(query)
            elapsed = time.time() - start_time
            
            print(f"  Intent: {result['intent']} ({result['confidence']:.2%})")
            print(f"  Emotion: {emotion['emotion']}")
            print(f"  Time: {elapsed:.2f}s")
            
            # Save to database
            self.database.save_log(
                user_input=query,
                bot_response=f"[Response to: {description}]",
                intent=result['intent'],
                confidence=result['confidence'],
                emotion=emotion['emotion'],
                llm_source='test',
                response_time=elapsed
            )
    
    def check_database(self):
        """Check database status and logs."""
        print(f"\n{Fore.CYAN}DATABASE STATUS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
        
        analytics = self.database.get_analytics_summary()
        
        if analytics:
            print(f"Total Interactions: {analytics.get('total_interactions', 0)}")
            print(f"Average Confidence: {analytics.get('average_confidence', 0):.2%}")
            print(f"Average Response Time: {analytics.get('average_response_time', 0):.2f}s")
            print(f"\nTop Intents:")
            for intent, count in list(analytics.get('top_intents', {}).items())[:5]:
                print(f"  - {intent}: {count}")
        else:
            print("No analytics data yet.")
        
        # List recent sessions
        sessions = self.database.list_sessions(limit=5)
        if sessions:
            print(f"\nRecent Sessions ({len(sessions)}):")
            for session in sessions:
                print(f"  - {session['session_id'][:8]}... ({session['total_turns']} turns)")
        else:
            print("No sessions recorded yet.")
    
    def performance_analysis(self):
        """Run performance analysis."""
        print(f"\n{Fore.CYAN}PERFORMANCE ANALYSIS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-'*60}{Style.RESET_ALL}")
        
        queries = [
            "What are the fees?",
            "Tell me about placements",
            "How do I apply?",
            "What's the exam date?",
            "Are there scholarships?"
        ]
        
        print(f"\nTesting {len(queries)} queries...")
        times = []
        
        for i, query in enumerate(queries, 1):
            start_time = time.time()
            
            # Test both intent and emotion
            self.intent_classifier.predict(query)
            self.emotion_detector.detect_emotion(query)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"  {i}. {elapsed:.2f}s - {query[:40]}")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"\n{'='*60}")
        print(f"Average Response Time: {Fore.GREEN if avg_time < 3 else Fore.YELLOW}{avg_time:.2f}s{Style.RESET_ALL}")
        print(f"Min Time: {min_time:.2f}s")
        print(f"Max Time: {max_time:.2f}s")
        print(f"Target: <3 seconds {'✓' if avg_time < 3 else '✗'}")
    
    def run(self):
        """Run the testing loop."""
        while True:
            self.display_menu()
            choice = input(f"{Fore.CYAN}Select option (1-7): {Style.RESET_ALL}").strip()
            
            if choice == "1":
                self.test_intent_classification()
            elif choice == "2":
                self.test_emotion_detection()
            elif choice == "3":
                self.test_full_conversation()
            elif choice == "4":
                self.run_test_scenarios()
            elif choice == "5":
                self.check_database()
            elif choice == "6":
                self.performance_analysis()
            elif choice == "7":
                print(f"\n{Fore.GREEN}Thank you for testing! Goodbye.{Style.RESET_ALL}\n")
                break
            else:
                print(f"{Fore.RED}Invalid option. Please try again.{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        tester = ManualTester()
        tester.run()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Testing interrupted by user.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)
