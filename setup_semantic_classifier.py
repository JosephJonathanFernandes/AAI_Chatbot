#!/usr/bin/env python3
"""
Setup script for AAI Chatbot with new semantic intent classifier.
Trains the model and validates all enhancements.

Run this ONCE after updating code:
    python setup_semantic_classifier.py
"""

import json
import sys
from pathlib import Path
from text_preprocessor import TextPreprocessor
from intent_model import SemanticIntentClassifier
from utils import load_json_file


def setup_classifier():
    """Initialize and train semantic classifier."""
    print("\n" + "="*80)
    print("🚀 SEMANTIC INTENT CLASSIFIER SETUP")
    print("="*80)
    
    print("\n1️⃣  Initializing semantic classifier (using sentence-transformers)...")
    classifier = SemanticIntentClassifier()
    
    print("\n2️⃣  Training on enhanced intents.json...")
    result = classifier.train("data/intents.json")
    
    if not result.get("success"):
        print(f"❌ Training failed: {result.get('error')}")
        return False
    
    print(f"\n✅ Training successful!")
    print(f"   - Intents: {result['intents_count']}")
    print(f"   - Total patterns: {result['total_patterns']}")
    print(f"   - Intent tags: {', '.join(result['intents'][:5])}... (+{len(result['intents'])-5} more)")
    
    return classifier


def test_robustness(classifier):
    """Test classifier robustness against typos, Hinglish, slang."""
    print("\n" + "="*80)
    print("🧪 ROBUSTNESS TESTING")
    print("="*80)
    
    test_cases = [
        # Normal queries
        ("Tell me about placements", "placements", "Normal query"),
        ("What are exam dates?", "exams", "Normal  query"),
        ("Fee structure?", "fees", "Normal query"),
        
        # Typos
        ("cn i get a scolarship", "fees", "Typo: cn,scolarship"),
        ("wht are fees", "fees", "Typo: wht"),
        ("placements info pls", "placements", "Typo: pls"),
        
        # Hinglish
        ("Kya fees hain?", "fees", "Hinglish: Kya...hain"),
        ("exam kab hoga", "exams", "Hinglish: exam...kab"),
        ("placements ke liye kya chahiye", "placements", "Hinglish: mixed"),
        
        # Slang/Casual
        ("wht bout placements bc", "placements", "Slang: bc"),
        ("tell me abt campus life", "campus_life", "Casual: abt"),
        ("fee?", "fees", "Ultra-short"),
        
        # Broad queries
        ("Tell me about the college", "general_info", "Broad query"),
        
        # Out of scope
        ("What's the weather", "out_of_scope", "Out of scope"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for query, expected_intent, description in test_cases:
        detected_intent, confidence, debug_info = classifier.predict(query)
        is_correct = detected_intent == expected_intent
        correct += is_correct
        
        symbol = "✅" if is_correct else "❌"
        print(f"\n{symbol} {description}")
        print(f"   Query: '{query}'")
        print(f"   Expected: {expected_intent}, Got: {detected_intent} (confidence: {confidence:.3f})")
        if not is_correct:
            print(f"   Debug: {debug_info}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n" + "="*80)
    print(f"📊 ROBUSTNESS TEST RESULTS: {correct}/{total} correct ({accuracy:.1f}%)")
    print("="*80)
    
    return accuracy


def verify_enhancements():
    """Verify all data enhancements are in place."""
    print("\n" + "="*80)
    print("✔️  DATA ENHANCEMENTS VERIFICATION")
    print("="*80)
    
    # Check college_data.json
    print("\n1️⃣  Checking college_data.json...")
    college_data = load_json_file("data/college_data.json")
    
    checks = [
        ("Keywords in fees", "keywords" in college_data.get("fees", {})),
        ("FAQ entries", len(college_data.get("faq", [])) > 0),
        ("Payment modes", "payment_modes" in college_data.get("fees", {}).get("engineering", {})),
        ("Contact info", "contact" in college_data),
    ]
    
    for check_name, result in checks:
        symbol = "✓" if result else "✗"
        print(f"   {symbol} {check_name}: {result}")
    
    # Check intents.json
    print("\n2️⃣  Checking intents.json (enhanced)...")
    intents_data = load_json_file("data/intents.json")
    intents_list = intents_data.get("intents", [])
    tags = [i.get("tag") for i in intents_list]
    total_patterns = sum(len(i.get("patterns", [])) for i in intents_list)
    
    checks2 = [
        ("Total intents", len(intents_list) >= 14),
        ("Has out_of_scope", "out_of_scope" in tags),
        ("Has admissions", "admission" in tags),
        ("Has hostel", "hostel" in tags),
        ("Has campus_life", "campus_life" in tags),
        ("Has general_info", "general_info" in tags),
        ("Total patterns >= 600", total_patterns >= 600),
    ]
    
    for check_name, result in checks2:
        symbol = "✓" if result else "✗"
        print(f"   {symbol} {check_name}: {result}")
    
    print(f"\n   Intent tags ({len(intents_list)} total): {', '.join(tags)}")
    print(f"   Total patterns: {total_patterns}")
    
    # Check text preprocessor
    print("\n3️⃣  Checking text preprocessor...")
    test_inputs = [
        ("Kya fees hain?", "fees"),
        ("cn i get scholarship", "get scholarship"),
        ("wht about placements", "about placements"),
    ]
    
    for test_input, expected_contains in test_inputs:
        preprocessed = TextPreprocessor.preprocess(test_input)
        has_expected = expected_contains.lower() in preprocessed.lower()
        symbol = "✓" if has_expected else "✗"
        print(f"   {symbol} '{test_input}' → '{preprocessed}'")
    
    print("\n" + "="*80)
    print("✅ SETUP COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    try:
        # Step 1: Setup classifier
        classifier = setup_classifier()
        if not classifier:
            print("\n❌ Setup failed!")
            sys.exit(1)
        
        # Step 2: Test robustness
        accuracy = test_robustness(classifier)
        if accuracy < 70:
            print(f"⚠️  Warning: Accuracy {accuracy:.1f}% is below 70%")
        
        # Step 3: Verify enhancements
        verify_enhancements()
        
        print("\n🎉 All setup complete! Your chatbot is ready to roll.\n")
        print("Next steps:")
        print("  1. Run Streamlit: streamlit run app.py")
        print("  2. Test with queries: placements, fees, exams, etc.")
        print("  3. Try edge cases: typos, Hinglish, casual language")
        print()
        
    except Exception as e:
        print(f"\n❌ SETUP FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
