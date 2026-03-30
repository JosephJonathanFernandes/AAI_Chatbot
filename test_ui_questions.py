#!/usr/bin/env python3
"""
CLI Test Script for All UI Test Questions
Allows testing chatbot with predefined question sets by category
"""

import sys
import json
import time
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

# Import our modules
from intent_model import IntentClassifier
from emotion_detector import EmotionDetector
from llm_handler import LLMHandler
from context_manager import ConversationContext
from database import ChatbotDatabase


class UIQuestionsTesters:
    """Test all UI question categories."""
    
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
                    "Who is the professor for [subject]?",
                    "How can I contact faculty?",
                    "What are faculty office hours?",
                    "Who is the department head?",
                    "Are there guest lectures?",
                    "Faculty information",
                    "Teacher details"
                ]
            },
            "6": {
                "name": "HOLIDAYS",
                "questions": [
                    "When are the holidays?",
                    "What's the summer break schedule?",
                    "When is the winter break?",
                    "What are the semester break dates?",
                    "Holiday schedule",
                    "Break dates"
                ]
            },
            "7": {
                "name": "LIBRARY",
                "questions": [
                    "What are library timings?",
                    "How do I borrow books?",
                    "What e-resources are available?",
                    "Where is the computer lab?",
                    "Library information",
                    "Book availability",
                    "Library membership"
                ]
            },
            "8": {
                "name": "ADMISSION",
                "questions": [
                    "How do I apply to the college?",
                    "What are the admission requirements?",
                    "What's the application deadline?",
                    "What are the cutoff marks?",
                    "Do I need an entrance exam?",
                    "Document requirements",
                    "Merit list"
                ]
            },
            "9": {
                "name": "DEPARTMENTS",
                "questions": [
                    "What departments are available?",
                    "Which courses does the Engineering department offer?",
                    "What specializations are available?",
                    "Arts department",
                    "Science courses",
                    "Commerce programs"
                ]
            },
            "10": {
                "name": "GREETINGS",
                "questions": [
                    "Hi! How are you?",
                    "Hello, what's up?",
                    "Good morning!",
                    "Nice to meet you!",
                    "Hey there!",
                    "Greetings",
                    "Welcome"
                ]
            },
            "11": {
                "name": "GRATITUDE",
                "questions": [
                    "Thank you for your help!",
                    "Thanks a lot!",
                    "I appreciate your assistance",
                    "That was very helpful!",
                    "Thanks for helping"
                ]
            },
            "12": {
                "name": "EMOTIONAL/CONTEXT",
                "questions": [
                    "I'm really stressed about the exams, what should I do?",
                    "I'm so frustrated! The timetable changed again!",
                    "I'm confused about which department to choose",
                    "I'm very happy with the placement opportunities!",
                    "I'm worried about my grades",
                    "This is making me anxious, can you help?",
                    "I love this college, it's amazing!"
                ]
            },
            "13": {
                "name": "OUT-OF-SCOPE (Edge Cases)",
                "questions": [
                    "What's the weather today?",
                    "How do I cook pasta?",
                    "Tell me a joke",
                    "What's your favorite movie?",
                    "What's 2+2?",
                    "Can you help with my homework in physics?",
                    "I'm feeling depressed",
                    "How do I fix my laptop?"
                ]
            },
            "14": {
                "name": "EDGE CASES",
                "questions": [
                    "wot r the fees?",
                    "FEES",
                    "fee",
                    "tution",
                    "Tell me about fees and placements",
                    "When are exams and what's the syllabus?",
                    "When is it?",
                    "Tell me more"
                ]
            }
        }
    
    def display_menu(self):
        """Display test menu."""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}UI QUESTION TEST CATEGORIES{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        
        for key, value in self.question_sets.items():
            print(f"  {key:2s}. {value['name']:<30s} ({len(value['questions'])} questions)")
        
        print(f"\n  {'0':<2s}. Test ALL categories")
        print(f"  {'c':<2s}. Custom question")
        print(f"  {'q':<2s}. Quit\n")
    
    def test_single_question(self, question):
        """Test a single question and display results."""
        print(f"\n{Fore.YELLOW}Testing: {question}{Style.RESET_ALL}\n")
        
        try:
            start = time.time()
            
            # Intent Classification
            intent_result = self.intent_classifier.predict(question)
            intent = intent_result.get('intent')
            intent_conf = intent_result.get('confidence', 0)
            
            # Emotion Detection
            emotion_result = self.emotion_detector.detect_emotion(question)
            emotion = emotion_result.get('emotion')
            emotion_conf = emotion_result.get('confidence', 0)
            
            # Response Generation
            response = self.llm_handler.generate_response(
                user_input=question,
                intent=intent,
                confidence=intent_conf,
                emotion=emotion
            )
            
            elapsed = time.time() - start
            
            # Display results
            print(f"{Fore.GREEN}[RESULTS]{Style.RESET_ALL}")
            print(f"  Intent:        {Fore.CYAN}{intent}{Style.RESET_ALL} (confidence: {intent_conf:.2%})")
            print(f"  Emotion:       {Fore.CYAN}{emotion}{Style.RESET_ALL} (confidence: {emotion_conf:.2%})")
            print(f"  LLM Source:    {Fore.CYAN}{response.get('source', 'unknown')}{Style.RESET_ALL}")
            print(f"  Response Time: {Fore.CYAN}{elapsed:.2f}s{Style.RESET_ALL}")
            
            print(f"\n{Fore.YELLOW}Response:{Style.RESET_ALL}")
            print(f"  {response.get('response', 'No response generated')}\n")
            
            # Log to database
            self.database.log_interaction(
                user_input=question,
                intent=intent,
                confidence=intent_conf,
                emotion=emotion,
                response=response.get('response', ''),
                response_time=elapsed,
                llm_source=response.get('source', 'unknown')
            )
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}\n")
    
    def test_category(self, category_key):
        """Test all questions in a category."""
        if category_key not in self.question_sets:
            print(f"{Fore.RED}Invalid category!{Style.RESET_ALL}")
            return
        
        category = self.question_sets[category_key]
        print(f"\n{Fore.GREEN}Testing {category['name']} Category{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}\n")
        
        for i, question in enumerate(category['questions'], 1):
            print(f"{Fore.YELLOW}[{i}/{len(category['questions'])}]{Style.RESET_ALL} ", end="")
            self.test_single_question(question)
            time.sleep(0.5)  # Small delay between requests
    
    def test_all_categories(self):
        """Test all categories."""
        print(f"\n{Fore.MAGENTA}Testing ALL Categories{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'='*70}{Style.RESET_ALL}\n")
        
        for key in sorted(self.question_sets.keys()):
            self.test_category(key)
        
        print(f"\n{Fore.GREEN}[OK] All categories tested!{Style.RESET_ALL}\n")
    
    def run(self):
        """Run interactive test menu."""
        while True:
            self.display_menu()
            choice = input(f"{Fore.CYAN}Select option: {Style.RESET_ALL}").strip().lower()
            
            if choice == 'q':
                print(f"{Fore.YELLOW}Exiting...{Style.RESET_ALL}")
                break
            elif choice == '0':
                self.test_all_categories()
            elif choice == 'c':
                question = input(f"{Fore.CYAN}Enter your question: {Style.RESET_ALL}")
                if question:
                    self.test_single_question(question)
            elif choice in self.question_sets:
                self.test_category(choice)
            else:
                print(f"{Fore.RED}Invalid option!{Style.RESET_ALL}")


def main():
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print(f"COLLEGE AI ASSISTANT - UI QUESTION TEST SUITE")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    tester = UIQuestionsTesters()
    tester.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted!{Style.RESET_ALL}")
        sys.exit(0)
