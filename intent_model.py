"""
Ensemble Intent Classifier (Semantic + TF-IDF with weighted voting).
Provides robust intent detection for typos, slang, Hinglish, and casual variations.

Architecture:
- Primary: Sentence-transformers semantic similarity (70% weight, best for typos/slang)
- Fallback: TF-IDF + Logistic Regression (30% weight, catches edge cases)
- Ensemble: Weighted voting for maximum accuracy
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from text_preprocessor import TextPreprocessor
from utils import load_json_file


class SemanticIntentClassifier:
    """
    Classifies user intents using sentence embeddings (transformers).
    Robust to typos, slang, Hinglish, and casual variations.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.intent_embeddings = {}
        self.intent_labels = []
        self.is_trained = False
    
    def train(self, intents_json_path: str = "data/intents.json") -> Dict:
        """Train by embedding all intent patterns."""
        print(f"  Loading semantic patterns from {intents_json_path}...")
        intents_data = load_json_file(intents_json_path)
        
        if not intents_data.get("intents"):
            return {"success": False, "error": "No intents found"}
        
        total_patterns = 0
        for intent in intents_data["intents"]:
            tag = intent.get("tag")
            patterns = intent.get("patterns", [])
            
            if not tag or not patterns:
                continue
            
            self.intent_labels.append(tag)
            preprocessed = [TextPreprocessor.preprocess(p) for p in patterns]
            embeddings = self.model.encode(preprocessed, convert_to_tensor=True)
            self.intent_embeddings[tag] = embeddings
            total_patterns += len(patterns)
        
        self.is_trained = True
        return {"success": True, "patterns": total_patterns, "intents": len(self.intent_labels)}
    
    def predict(self, user_input: str) -> Tuple[str, float]:
        """Predict intent using semantic similarity."""
        if not self.is_trained:
            return "unknown", 0.0
        
        processed_input = TextPreprocessor.preprocess(user_input)
        if not processed_input:
            return "unknown", 0.0
        
        input_embedding = self.model.encode(processed_input, convert_to_tensor=True)
        best_intent = None
        best_score = -1
        
        for intent_tag, intent_embeddings in self.intent_embeddings.items():
            similarities = util.pytorch_cos_sim(input_embedding, intent_embeddings)[0]
            max_similarity = float(similarities.max().item())
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_intent = intent_tag
        
        confidence = max(0.0, min(1.0, best_score))
        return best_intent or "unknown", confidence


class TFIDFIntentClassifier:
    """
    Classifies user intents using TF-IDF + Logistic Regression.
    Used as fallback/ensemble partner to catch edge cases.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000, lowercase=True, stop_words='english',
            ngram_range=(1, 2), min_df=1, max_df=0.9
        )
        self.model = LogisticRegression(max_iter=200, random_state=42)
        self.intent_labels = []
        self.is_trained = False
    
    def train(self, intents_json_path: str = "data/intents.json") -> Dict:
        """Train TF-IDF classifier."""
        print(f"  Loading TF-IDF patterns from {intents_json_path}...")
        intents_data = load_json_file(intents_json_path)
        
        if not intents_data.get("intents"):
            return {"success": False, "error": "No intents found"}
        
        all_patterns = []
        all_labels = []
        total_patterns = 0
        
        for intent in intents_data["intents"]:
            tag = intent.get("tag")
            patterns = intent.get("patterns", [])
            
            if not tag or not patterns:
                continue
            
            self.intent_labels.append(tag)
            
            for pattern in patterns:
                preprocessed = TextPreprocessor.preprocess(pattern)
                all_patterns.append(preprocessed)
                all_labels.append(tag)
                total_patterns += 1
        
        X_tfidf = self.vectorizer.fit_transform(all_patterns)
        self.model.fit(X_tfidf, all_labels)
        self.is_trained = True
        
        return {"success": True, "patterns": total_patterns, "intents": len(self.intent_labels)}
    
    def predict(self, user_input: str) -> Tuple[str, float]:
        """Predict intent using TF-IDF."""
        if not self.is_trained:
            return "unknown", 0.0
        
        processed_input = TextPreprocessor.preprocess(user_input)
        if not processed_input:
            return "unknown", 0.0
        
        X_tfidf = self.vectorizer.transform([processed_input])
        pred_proba = self.model.predict_proba(X_tfidf)[0]
        pred_idx = np.argmax(pred_proba)
        
        intent = self.model.classes_[pred_idx]
        confidence = float(pred_proba[pred_idx])
        
        return intent, confidence


class EnsembleIntentClassifier:
    """
    Weighted ensemble combining Semantic (70%) + TF-IDF (30%).
    
    Predicts with high accuracy by leveraging both model strengths:
    - Semantic: Understands intent meaning, robust to typos/slang
    - TF-IDF: Pattern matching, catches edge cases
    """
    
    def __init__(self, semantic_weight: float = 0.70, tfidf_weight: float = 0.30):
        self.semantic_classifier = SemanticIntentClassifier()
        self.tfidf_classifier = TFIDFIntentClassifier()
        self.semantic_weight = semantic_weight
        self.tfidf_weight = tfidf_weight
        self.intent_labels = []
        self.is_trained = False
    
    def train(self, intents_json_path: str = "data/intents.json") -> Dict:
        """Train both classifiers."""
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE INTENT CLASSIFIER")
        print(f"  Semantic: {self.semantic_weight*100:.0f}% | TF-IDF: {self.tfidf_weight*100:.0f}%")
        print("="*80)
        
        sem_result = self.semantic_classifier.train(intents_json_path)
        tfidf_result = self.tfidf_classifier.train(intents_json_path)
        
        if sem_result.get("success") and tfidf_result.get("success"):
            self.intent_labels = self.semantic_classifier.intent_labels
            self.is_trained = True
            
            print("\n[OK] ENSEMBLE TRAINED:")
            print(f"  Intents: {sem_result['intents']} | Patterns: {sem_result['patterns']} (semantic) + {tfidf_result['patterns']} (TF-IDF)")
            print("="*80 + "\n")
            
            return {"success": True, "semantic": sem_result, "tfidf": tfidf_result}
        
        return {"success": False}
    
    def predict(self, user_input: str) -> Tuple[str, float, Dict]:
        """
        Predict intent using ensemble weighted voting.
        
        Returns: (intent, confidence, detailed_scores)
        """
        if not self.is_trained:
            return "unknown", 0.0, {}
        
        # Get predictions from both models
        sem_intent, sem_conf = self.semantic_classifier.predict(user_input)
        tfidf_intent, tfidf_conf = self.tfidf_classifier.predict(user_input)
        
        # Apply weights
        sem_weighted = sem_conf * self.semantic_weight
        tfidf_weighted = tfidf_conf * self.tfidf_weight
        
        # Ensemble decision logic
        if sem_intent == tfidf_intent and min(sem_conf, tfidf_conf) > 0.5:
            # Both agree strongly → high confidence
            final_intent = sem_intent
            final_confidence = max(sem_conf, tfidf_conf)
        elif sem_weighted > tfidf_weighted:
            # Semantic dominates (expected ~70% of time)
            final_intent = sem_intent
            final_confidence = sem_weighted
        else:
            # TF-IDF edges it out
            final_intent = tfidf_intent
            final_confidence = tfidf_weighted
        
        detailed_scores = {
            "semantic": {"intent": sem_intent, "confidence": sem_conf, "weighted": sem_weighted},
            "tfidf": {"intent": tfidf_intent, "confidence": tfidf_conf, "weighted": tfidf_weighted},
            "ensemble": {"intent": final_intent, "confidence": final_confidence}
        }
        
        return final_intent or "unknown", final_confidence, detailed_scores
    
    def predict_batch(self, user_inputs: List[str]) -> List[Tuple[str, float]]:
        """Predict batch of inputs."""
        return [(intent, conf) for intent, conf, _ in [self.predict(inp) for inp in user_inputs]]


# Backward compatibility: IntentClassifier now uses ensemble by default
class IntentClassifier:
    """Backward-compatible wrapper using ensemble by default."""
    
    def __init__(self, model_path: str = "intent_model.pkl", vectorizer_path: str = "tfidf_vectorizer.pkl", use_ensemble: bool = True):
        """Initialize classifier."""
        if use_ensemble:
            self.classifier = EnsembleIntentClassifier()
        else:
            # Fallback to semantic only
            self.classifier = SemanticIntentClassifier()
        self.is_trained = False
    
    def train(self, intents_json_path: str = "data/intents.json") -> Dict:
        """Train the classifier."""
        result = self.classifier.train(intents_json_path)
        self.is_trained = result.get("success", False)
        return result
    
    def predict(self, user_input: str) -> Tuple[str, float]:
        """Predict intent (backward compatible)."""
        if isinstance(self.classifier, EnsembleIntentClassifier):
            intent, confidence, _ = self.classifier.predict(user_input)
        else:
            intent, confidence = self.classifier.predict(user_input)
        return intent, confidence
    
    def get_model_info(self) -> Dict:
        """Get model information for status display."""
        return {
            "is_trained": self.is_trained,
            "intents_count": len(self.classifier.intent_labels) if hasattr(self.classifier, 'intent_labels') else 0
        }


# Test suite
if __name__ == "__main__":
    print("ENSEMBLE INTENT CLASSIFIER - COMPREHENSIVE TEST")
    print("="*80)
    
    classifier = IntentClassifier(use_ensemble=True)
    
    print("\n1. Training ensemble classifier...")
    result = classifier.train("data/intents.json")
    print(f"   Result: {result}")
    
    print("\n2. Testing with challenging inputs...")
    test_cases = [
        # Normal queries
        ("Tell me about placements", "placements"),
        ("How much do engineering fees cost?", "fees"),
        ("When are exams scheduled?", "exams"),
        
        # Typos
        ("wht are placements", "placements"),
        ("cn i get info on placements", "placements"),
        ("fee kitne hain", "fees"),
        
        # Hinglish
        ("Kya fees hain?", "fees"),
        ("Placement ke liye kya karna?", "placements"),
        
        # Slang
        ("wht bout placements bc", "placements"),
        ("fees?", "fees"),
        ("When r exams?", "exams"),
        
        # Broad queries
        ("Tell me about the college", "general_info"),
        
        # Out of scope
        ("Politics", "out_of_scope"),
        ("Tell me a joke", "out_of_scope"),
    ]
    
    passed = 0
    for user_input, expected in test_cases:
        intent, confidence = classifier.predict(user_input)
        match = "[OK]" if intent == expected else "[FAIL]"
        
        if intent == expected:
            passed += 1
        
        print(f"{match} '{user_input}'")
        print(f"   Expected: {expected:15} | Got: {intent:15} (conf: {confidence:.2f})")
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {passed}/{len(test_cases)} passed ({100*passed/len(test_cases):.0f}%)")
    print("="*80)
