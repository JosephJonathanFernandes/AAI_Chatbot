"""
Semantic Intent Classifier using sentence-transformers.
Provides robust intent detection for typos, slang, Hinglish, and casual variations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sentence_transformers import SentenceTransformer, util
from text_preprocessor import TextPreprocessor
from utils import load_json_file


class SemanticIntentClassifier:
    """
    Classifies user intents using sentence embeddings and semantic similarity.
    MUCH more robust than TF-IDF for handling typos, slang, Hinglish.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic classifier.
        
        Args:
            model_name (str): Sentence-transformers model name (default: fast, multilingual-capable)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.intent_embeddings = {}  # intent -> list of embeddings
        self.intent_labels = []
        self.is_trained = False
        self.threshold = 0.35  # Similarity threshold (lower = more confident on harder queries)
    
    def train(self, intents_json_path: str = "data/intents.json") -> Dict:
        """
        Train the classifier by embedding all intent patterns.
        
        Args:
            intents_json_path (str): Path to intents.json
        
        Returns:
            dict: Training results
        """
        print(f"Loading intent dataset from {intents_json_path}...")
        intents_data = load_json_file(intents_json_path)
        
        if not intents_data.get("intents"):
            return {"success": False, "error": "No intents found"}
        
        # Preprocess and embed all patterns
        total_patterns = 0
        for intent in intents_data["intents"]:
            tag = intent.get("tag")
            patterns = intent.get("patterns", [])
            
            if not tag or not patterns:
                continue
            
            self.intent_labels.append(tag)
            
            # Preprocess all patterns for this intent
            preprocessed = [TextPreprocessor.preprocess(p) for p in patterns]
            
            # Embed them
            embeddings = self.model.encode(preprocessed, convert_to_tensor=True)
            self.intent_embeddings[tag] = embeddings
            
            total_patterns += len(patterns)
            print(f"  ✓ {tag}: {len(patterns)} patterns")
        
        self.is_trained = True
        result = {
            "success": True,
            "intents_count": len(self.intent_labels),
            "total_patterns": total_patterns,
            "model": self.model_name,
            "intents": self.intent_labels
        }
        
        print(f"\n✓ Training complete: {len(self.intent_labels)} intents, {total_patterns} patterns")
        return result
    
    def predict(self, user_input: str) -> Tuple[str, float, str]:
        """
        Predict intent for user input using semantic similarity.
        
        Args:
            user_input (str): Raw user input
        
        Returns:
            Tuple[str, float, str]: (intent, confidence, debug_info)
                - intent: Detected intent tag
                - confidence: Similarity score (0-1)
                - debug_info: Processing information
        """
        if not self.is_trained:
            return "unknown", 0.0, "Model not trained"
        
        # Preprocess input
        processed_input = TextPreprocessor.preprocess(user_input)
        
        if not processed_input:
            return "unknown", 0.0, f"Empty after preprocessing: '{user_input}'"
        
        # Embed the input
        input_embedding = self.model.encode(processed_input, convert_to_tensor=True)
        
        # Find best matching intent
        best_intent = None
        best_score = -1
        scores_by_intent = {}
        
        for intent_tag, intent_embeddings in self.intent_embeddings.items():
            # Calculate cosine similarity with all patterns for this intent
            similarities = util.pytorch_cos_sim(input_embedding, intent_embeddings)[0]
            
            # Take max similarity (best matching pattern for this intent)
            max_similarity = float(similarities.max().item())
            scores_by_intent[intent_tag] = max_similarity
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent_tag
        
        # Confidence is the max similarity score
        confidence = max(0.0, min(1.0, best_score))
        
        # Debug info with top 3 matches
        sorted_scores = sorted(scores_by_intent.items(), key=lambda x: x[1], reverse=True)[:3]
        debug_lines = [
            f"Input: '{user_input}' → '{processed_input}'",
            f"Top 3 matches: {', '.join([f'{i}({s:.2f})' for i, s in sorted_scores])}",
            f"Best: {best_intent} with confidence {confidence:.2f}"
        ]
        debug_info = " | ".join(debug_lines)
        
        return best_intent or "unknown", confidence, debug_info
    
    def predict_batch(self, user_inputs: List[str]) -> List[Tuple[str, float]]:
        """
        Predict intents for batch of inputs.
        
        Args:
            user_inputs (List[str]): List of user inputs
        
        Returns:
            List[Tuple[str, float]]: List of (intent, confidence) tuples
        """
        return [self.predict(inp)[0:2] for inp in user_inputs]


# Backward compatibility with old IntentClassifier interface
class IntentClassifier:
    """Wrapper for backward compatibility with existing code."""
    
    def __init__(self, model_path: str = "intent_model.pkl", vectorizer_path: str = "tfidf_vectorizer.pkl"):
        """Initialize classifier - now uses semantic approach."""
        self.semantic_classifier = SemanticIntentClassifier()
        self.is_trained = False
    
    def train(self, intents_json_path: str = "data/intents.json") -> Dict:
        """Train the semantic classifier."""
        result = self.semantic_classifier.train(intents_json_path)
        self.is_trained = result.get("success", False)
        return result
    
    def predict(self, user_input: str) -> Tuple[str, float]:
        """Predict intent (backward compatible)."""
        intent, confidence, _ = self.semantic_classifier.predict(user_input)
        return intent, confidence


if __name__ == "__main__":
    # Test the classifier
    print("SEMANTIC INTENT CLASSIFIER TEST")
    print("=" * 80)
    
    classifier = SemanticIntentClassifier()
    
    print("\n1. Training classifier...")
    result = classifier.train("data/intents.json")
    print(f"   Result: {result}")
    
    print("\n2. Testing with various inputs (typos, Hinglish, slang)...")
    test_cases = [
        ("Tell me about placements", "placements"),  # Normal
        ("wht are placements", "placements"),  # Typo
        ("cn i get info on placements", "placements"),  # Abbreviated
        ("Kya placements hain", "placements"),  # Hinglish
        ("fees kitne hain", "fees"),  # Hinglish
        ("How much do engineering fees cost?", "fees"),  # Normal
        ("Fee structure?", "fees"),  # Short
        ("exam schedule pls", "exams"),  # Casual
        ("When r exams?", "exams"),  # Very casual
        ("Tell me about the college", "general_info"),  # Broad query
        ("Politics", "out_of_scope"),  # Out of scope
    ]
    
    print()
    for user_input, expected in test_cases:
        intent, confidence, debug_info = classifier.predict(user_input)
        match = "✓" if intent == expected else "✗"
        print(f"{match} '{user_input}'")
        print(f"   Expected: {expected}, Got: {intent} (confidence: {confidence:.2f})")
        if intent != expected:
            print(f"   Debug: {debug_info}")
        print()
