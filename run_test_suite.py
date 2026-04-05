"""
Extract all test questions from CHATBOT_QUESTION_TEST_CASES.txt and run them through the chatbot.
Store results in a txt file with questions and responses.
"""

import re
from pathlib import Path
from intent_model import IntentClassifier
from emotion_detector import EmotionDetector
from llm_handler import LLMHandler
from context_manager import ConversationContext


def extract_test_queries(test_file_path: str) -> list:
    """
    Extract all test queries from the test case file.
    
    Returns:
        list of tuples: (test_id, question)
    """
    queries = []
    
    with open(test_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to extract Test ID and Test Query
    pattern = r'TEST ID:\s*([A-Z0-9-]+).*?Test Query:\s*"([^"]+)"'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for test_id, query in matches:
        queries.append((test_id, query.strip()))
    
    return queries


def run_test_suite():
    """Run all test queries and collect responses."""
    
    print("Initializing chatbot components...")
    
    # Initialize components
    intent_classifier = IntentClassifier()
    emotion_detector = EmotionDetector()
    llm_handler = LLMHandler()
    conversation_context = ConversationContext()
    
    # Extract test queries
    print("Extracting test queries from CHATBOT_QUESTION_TEST_CASES.txt...")
    test_file = "CHATBOT_QUESTION_TEST_CASES.txt"
    test_queries = extract_test_queries(test_file)
    
    print(f"Found {len(test_queries)} test queries.")
    
    # Prepare output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("CHATBOT TEST SUITE - QUESTION & RESPONSE LOG")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Run each test query
    for idx, (test_id, query) in enumerate(test_queries, 1):
        print(f"[{idx}/{len(test_queries)}] Running TEST {test_id}...")
        
        try:
            # Detect intent
            intent_result = intent_classifier.predict(query)
            intent = intent_result.get("intent", "unknown")
            confidence = intent_result.get("confidence", 0.0)
            
            # Detect emotion
            emotion_result = emotion_detector.detect_emotion(query)
            emotion = emotion_result.get("emotion", "neutral") if isinstance(emotion_result, dict) else emotion_result
            
            # Generate response
            response_data = llm_handler.generate_response(
                user_input=query,
                intent=intent,
                confidence=confidence,
                emotion=emotion,
                conversation_history=[]
            )
            
            response = response_data.get("response", "No response generated")
            
        except Exception as e:
            response = f"[ERROR] {str(e)}"
        
        # Format output
        output_lines.append(f"TEST ID: {test_id}")
        output_lines.append(f"QUESTION: {query}")
        output_lines.append(f"RESPONSE: {response}")
        output_lines.append("-" * 80)
        output_lines.append("")
    
    # Write to file
    output_file = "test_results_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    print(f"\n✅ Test suite completed!")
    print(f"Results saved to: {output_file}")
    print(f"Total tests run: {len(test_queries)}")


if __name__ == "__main__":
    run_test_suite()
