"""
CLI runner for the college chatbot (optional).
Provides a command-line interface for testing and interaction.
Features Phase 2: Fast, accurate responses with emotion/intent improvements.
"""

import argparse
from intent_model import IntentClassifier
from emotion_detector import EmotionDetector
from llm_handler import LLMHandler
from context_manager import ConversationContext
from database import ChatbotDatabase
from session_greeter import SessionGreeter
from error_recovery import ErrorRecovery
from emotional_tone_detector import EmotionalToneDetector
from intent_refiner import IntentRefiner
from utils import get_time_of_day, is_college_domain_query
import time


class ChatbotCLI:
    """Command-line interface for the chatbot."""
    
    def __init__(self):
        """Initialize CLI chatbot."""
        print("🎓 Initializing College AI Assistant...\n")
        
        # Initialize models
        self.intent_classifier = IntentClassifier()
        self.emotion_detector = EmotionDetector()
        self.llm_handler = LLMHandler()
        self.database = ChatbotDatabase()
        self.context_manager = ConversationContext()
        
        # Phase 2: Enhanced modules
        self.session_greeter = SessionGreeter()
        self.error_recovery = ErrorRecovery()
        self.tone_detector = EmotionalToneDetector()
        self.intent_refiner = IntentRefiner()
        
        # Train model if needed
        if not self.intent_classifier.is_trained:
            print("Training intent classifier...")
            self.intent_classifier.train()
        
        print("✅ Chatbot initialized successfully!\n")
    
    def run(self):
        """Run the interactive CLI chatbot."""
        # Phase 2: Show contextual greeting
        print(self.session_greeter.greet(include_prompt=True))
        print("\n" + "=" * 60)
        print("College AI Assistant - CLI Mode")
        print("=" * 60)
        print("Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.lower() == "quit":
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == "help":
                    self._show_help()
                    continue
                
                if user_input.lower() == "stats":
                    self._show_stats()
                    continue
                
                if user_input.lower() == "clear":
                    self.context_manager.clear_history()
                    print("✓ Chat history cleared.\n")
                    continue
                
                # Process user input
                self._process_input(user_input)
            
            except KeyboardInterrupt:
                print("\n\nChatbot terminated.")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    
    def _process_input(self, user_input):
        """Process user input and generate response."""
        start_time = time.time()
        
        # Step 1: Classify intent
        classification = self.intent_classifier.predict(user_input)
        intent = classification.get("intent")
        confidence = classification.get("confidence", 0.0)
        
        # Step 2: Detect emotion
        emotion_result = self.emotion_detector.detect_emotion(user_input)
        emotion = emotion_result.get("emotion", "neutral")
        emotion_conf = emotion_result.get("confidence", 0.0)
        
        # PHASE 2: Step 3 - Refine intent using context (FAST & ACCURATE)
        history = self.context_manager.get_history()
        refined_result = self.intent_refiner.refine_intent(
            predicted_intent=intent,
            confidence=confidence,
            user_input=user_input,
            conversation_history=history,
            emotion=emotion
        )
        intent = refined_result["intent"]
        confidence = refined_result["confidence"]
        
        # PHASE 2: Step 4 - Detect emotional tone for adaptive responses
        tone_result = self.tone_detector.detect_tone(user_input, emotion, intent)
        emotional_tone = tone_result["tone_name"]
        tone_guidelines = self.tone_detector.get_response_guidelines(tone_result)
        
        # Step 5: Check if in domain
        is_in_domain = is_college_domain_query(intent, confidence)
        
        # PHASE 2: Step 6 - Error recovery check
        if confidence < 0.4:
            error_info = self.error_recovery.handle_confidence_error(confidence, intent)
            if not error_info.get("should_proceed", True):
                emotion = "confused"
        
        # Step 7: Generate context
        context = self.context_manager.get_prompt_context()
        
        # Step 8: Generate response with Phase 2 enhancements
        try:
            if not is_in_domain or confidence < 0.5:
                response = self.context_manager.get_clarification_prompt()
                llm_source = "clarification"
                response_time = 0.0
            else:
                llm_result = self.llm_handler.generate_response(
                    user_input,
                    intent,
                    confidence,
                    emotion,
                    context,
                    tone_guidelines=tone_guidelines  # PHASE 2
                )
                response = llm_result["response"]
                llm_source = llm_result.get("source", "unknown")
                response_time = llm_result.get("time", 0.0)
        
        except Exception as e:
            # PHASE 2: Error recovery - graceful fallback
            error_recovery_result = self.error_recovery.handle_api_error(
                error=e,
                operation="llm_response_generation",
                context={"intent": intent, "user_input": user_input[:50]}
            )
            response = error_recovery_result.get("message", "Let me reconsider that. Could you rephrase?")
            llm_source = "fallback"
            response_time = 0.0
        
        # Display response with tone indicator
        print(f"\nAssistant: {response}")
        if emotional_tone != "neutral":
            print(f"  📊 Tone: {emotional_tone.replace('_', ' ').title()}")
        
        # Update context
        self.context_manager.add_turn(
            user_input,
            response,
            intent,
            confidence,
            emotion
        )
        
        # Log to database
        self.database.log_interaction(
            user_input,
            intent,
            confidence,
            emotion,
            response,
            response_time,
            llm_source
        )
        
        # Show debug info
        total_time = time.time() - start_time
        print(f"\n[Intent: {intent} ({confidence:.0%})] "
              f"[Emotion: {emotion} ({emotion_conf:.0%})] "
              f"[Source: {llm_source}] "
              f"[Time: {total_time:.2f}s]\n")
    
    def _show_help(self):
        """Show available commands."""
        print("""
Available Commands:
  quit      - Exit the chatbot
  help      - Show this help message
  stats     - Show conversation statistics
  clear     - Clear chat history
        """)
    
    def _show_stats(self):
        """Show conversation statistics."""
        analytics = self.database.get_analytics_summary()
        context_summary = self.context_manager.get_context_summary()
        
        print("\n" + "=" * 60)
        print("📊 Conversation Statistics")
        print("=" * 60)
        
        print(f"Total Interactions: {analytics.get('total_interactions', 0)}")
        print(f"Average Confidence: {analytics.get('average_confidence', 0):.2%}")
        print(f"Average Response Time: {analytics.get('average_response_time', 0):.2f}s")
        
        print(f"\nCurrent Session:")
        print(f"  Turns: {context_summary['total_turns']}")
        print(f"  Duration: {context_summary['session_duration']:.0f}s")
        print(f"  Last Intent: {context_summary['last_intent']}")
        print(f"  Last Emotion: {context_summary['last_emotion']}")
        
        if analytics.get("top_intents"):
            print(f"\nTop Intents:")
            for intent, count in list(analytics["top_intents"].items())[:3]:
                print(f"  {intent}: {count}")
        
        if analytics.get("llm_source_distribution"):
            print(f"\nLLM Source Distribution:")
            for source, count in analytics["llm_source_distribution"].items():
                print(f"  {source}: {count}")
        
        print("=" * 60 + "\n")


def train_model():
    """Train an intent model."""
    print("Training intent classifier...")
    classifier = IntentClassifier()
    result = classifier.train()
    
    if result.get("success"):
        print(f"✓ Training successful!")
        sem_result = result.get('semantic', {})
        tfidf_result = result.get('tfidf', {})
        print(f"  Intents: {sem_result.get('intents', 'N/A')}")
        print(f"  Patterns: {sem_result.get('patterns', 'N/A')} (semantic) + {tfidf_result.get('patterns', 'N/A')} (TF-IDF)")
        print(f"  Status: Ensemble classifier ready (75% semantic + 25% TF-IDF) [IMPROVED]")
    else:
        print(f"✗ Training failed: {result.get('error')}")


def show_logs(limit=20):
    """Show recent logs."""
    database = ChatbotDatabase()
    logs = database.get_logs(limit=limit)
    
    print(f"\n{'='*80}")
    print(f"Last {limit} Interactions")
    print(f"{'='*80}")
    
    for i, log in enumerate(logs, 1):
        print(f"\n{i}. [{log['timestamp']}]")
        print(f"   Input: {log['user_input'][:60]}...")
        print(f"   Intent: {log['intent']} ({log['confidence']:.0%})")
        print(f"   Emotion: {log['emotion']}")
        print(f"   Response: {log['response'][:60]}...")
    
    print(f"\n{'='*80}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="College AI Assistant - CLI Mode"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the intent classifier"
    )
    parser.add_argument(
        "--logs",
        type=int,
        nargs="?",
        const=20,
        help="Show recent logs (default: 20)"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        default=True,
        help="Run interactive chat mode (default)"
    )
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.logs is not None:
        show_logs(args.logs)
    else:
        # Run chat mode
        chatbot = ChatbotCLI()
        chatbot.run()


if __name__ == "__main__":
    main()
