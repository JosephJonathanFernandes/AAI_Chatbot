"""
Intent classification using scikit-learn (TF-IDF + Logistic Regression).
Trains and predicts user intents from input text.
"""

import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from utils import load_json_file


class IntentClassifier:
    """Classifies user intents using TF-IDF + Logistic Regression."""
    
    def __init__(self, model_path="intent_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
        """
        Initialize intent classifier.
        
        Args:
            model_path (str): Path to save/load trained model
            vectorizer_path (str): Path to save/load TF-IDF vectorizer
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.intent_labels = []
        self.is_trained = False
        
        # Try to load existing model
        if self._model_exists():
            self.load_model()
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize a new untrained model."""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        
        self.model = LogisticRegression(
            max_iter=200,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0
        )
    
    def train(self, intents_json_path="data/intents.json"):
        """
        Train the intent classifier from JSON dataset.
        
        Args:
            intents_json_path (str): Path to intents.json file
        
        Returns:
            dict: Training results and statistics
        """
        print("Loading intent dataset...")
        intents_data = load_json_file(intents_json_path)
        
        if not intents_data.get("intents"):
            print("Error: No intents found in dataset")
            return {"success": False, "error": "No training data"}
        
        # Prepare training data
        texts = []
        labels = []
        intent_labels_set = set()
        
        for intent in intents_data["intents"]:
            tag = intent.get("tag")
            patterns = intent.get("patterns", [])
            
            intent_labels_set.add(tag)
            
            for pattern in patterns:
                texts.append(pattern)
                labels.append(tag)
        
        self.intent_labels = sorted(list(intent_labels_set))
        
        print(f"Training on {len(texts)} samples from {len(self.intent_labels)} intents...")
        print(f"Intents: {self.intent_labels}")
        
        try:
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            # Calculate training accuracy
            train_accuracy = self.model.score(X, y)
            
            result = {
                "success": True,
                "samples_count": len(texts),
                "intents_count": len(self.intent_labels),
                "training_accuracy": round(train_accuracy, 3),
                "intents": self.intent_labels
            }
            
            print(f"✓ Model trained successfully with {train_accuracy:.1%} accuracy")
            return result
        
        except Exception as e:
            print(f"Error during training: {e}")
            return {"success": False, "error": str(e)}
    
    def predict(self, text, confidence_threshold=0.0):
        """
        Predict intent for given text.
        
        Args:
            text (str): User input text
            confidence_threshold (float): Minimum confidence to return prediction
        
        Returns:
            dict: Contains 'intent', 'confidence', 'all_probs' keys
        """
        if not self.is_trained or not self.model or not self.vectorizer:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "all_probs": {},
                "error": "Model not trained"
            }
        
        try:
            # Vectorize input
            X = self.vectorizer.transform([text])
            
            # Get predictions
            probabilities = self.model.predict_proba(X)[0]
            predicted_label = self.model.predict(X)[0]
            predicted_confidence = float(max(probabilities))
            
            # Create probability map for all intents
            all_probs = {label: float(prob) for label, prob in zip(self.model.classes_, probabilities)}
            
            # Apply confidence threshold
            if predicted_confidence < confidence_threshold:
                return {
                    "intent": "uncertain",
                    "confidence": 0.0,
                    "all_probs": all_probs,
                    "error": f"Confidence {predicted_confidence:.2f} below threshold {confidence_threshold}"
                }
            
            return {
                "intent": predicted_label,
                "confidence": round(predicted_confidence, 3),
                "all_probs": {k: round(v, 3) for k, v in all_probs.items()}
            }
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "all_probs": {},
                "error": str(e)
            }
    
    def batch_predict(self, texts):
        """
        Predict intents for multiple texts.
        
        Args:
            texts (list): List of input texts
        
        Returns:
            list: List of prediction results
        """
        return [self.predict(text) for text in texts]
    
    def save_model(self):
        """Save trained model and vectorizer to disk."""
        try:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Also save intent labels
            labels_path = self.model_path.replace('.pkl', '_labels.json')
            with open(labels_path, 'w') as f:
                json.dump(self.intent_labels, f)
            
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load pre-trained model and vectorizer from disk."""
        try:
            if not self._model_exists():
                print("Model files not found")
                return False
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load intent labels
            labels_path = self.model_path.replace('.pkl', '_labels.json')
            if Path(labels_path).exists():
                with open(labels_path, 'r') as f:
                    self.intent_labels = json.load(f)
            
            self.is_trained = True
            print(f"Model loaded from {self.model_path}")
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _model_exists(self):
        """Check if model files exist."""
        return Path(self.model_path).exists() and Path(self.vectorizer_path).exists()
    
    def get_model_info(self):
        """
        Get information about the trained model.
        
        Returns:
            dict: Model information
        """
        return {
            "is_trained": self.is_trained,
            "intents_count": len(self.intent_labels),
            "intents": self.intent_labels,
            "model_type": "LogisticRegression",
            "vectorizer_type": "TF-IDF"
        }
    
    def retrain_from_new_data(self, intents_json_path="data/intents.json"):
        """
        Retrain the model with new or updated intent data.
        
        Args:
            intents_json_path (str): Path to updated intents.json
        
        Returns:
            dict: Retraining results
        """
        self._initialize_model()  # Reset model
        return self.train(intents_json_path)
