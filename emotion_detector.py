"""
Transformer-based emotion detection using HuggingFace models.
Detects emotions: happy, stressed, confused, neutral, angry, sad.
"""

from transformers import pipeline
import torch


class EmotionDetector:
    """Detects user emotions using transformer-based sentiment analysis."""
    
    # Mapping from sentiment labels to college-specific emotions
    SENTIMENT_TO_EMOTION_MAP = {
        "POSITIVE": "happy",
        "NEGATIVE": "stressed",
        "NEUTRAL": "neutral"
    }
    
    # Detailed emotion mapping based on phrases
    DETAILED_EMOTION_KEYWORDS = {
        "angry": ["angry", "furious", "hate", "terrible", "awful", "horrible"],
        "stressed": ["stressed", "anxious", "worried", "frustrated", "confused", "help", "problem"],
        "happy": ["happy", "great", "excellent", "awesome", "love", "thanks", "good"],
        "sad": ["sad", "depressed", "upset", "down", "sorry", "disappointed"],
        "confused": ["confused", "unclear", "what", "how", "don't understand", "help you understand"]
    }
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize emotion detector with a transformer model.
        
        Args:
            model_name (str): HuggingFace model identifier
        """
        try:
            self.model_name = model_name
            self.device = 0 if torch.cuda.is_available() else -1
            
            # Load the sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=self.device
            )
            print(f"Emotion detector initialized with {model_name}")
        except Exception as e:
            print(f"Error initializing emotion detector: {e}")
            self.sentiment_pipeline = None
    
    def detect_emotion(self, text):
        """
        Detect emotions in the given text.
        
        Args:
            text (str): User input text
        
        Returns:
            dict: Contains 'emotion' and 'confidence' keys
        """
        if not text or not self.sentiment_pipeline:
            return {"emotion": "neutral", "confidence": 0.0}
        
        try:
            # First, check for keyword-based detailed emotions
            detailed_emotion = self._detect_detailed_emotion(text.lower())
            if detailed_emotion:
                return detailed_emotion
            
            # Fallback to sentiment analysis
            results = self.sentiment_pipeline(text[:512])  # Truncate for model limit
            
            if results and len(results) > 0:
                sentiment_label = results[0]['label']
                confidence = results[0]['score']
                
                # Map sentiment to emotion
                emotion = self.SENTIMENT_TO_EMOTION_MAP.get(sentiment_label, "neutral")
                
                return {
                    "emotion": emotion,
                    "confidence": round(confidence, 3)
                }
        
        except Exception as e:
            print(f"Error detecting emotion: {e}")
        
        return {"emotion": "neutral", "confidence": 0.0}
    
    def _detect_detailed_emotion(self, text):
        """
        Detect detailed emotions based on keywords.
        
        Args:
            text (str): Lowercase text to analyze
        
        Returns:
            dict or None: Emotion dict if matched, None otherwise
        """
        # Check for specific keywords
        for emotion, keywords in self.DETAILED_EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    # Calculate confidence based on keyword strength
                    word_count = len(text.split())
                    keyword_count = text.count(keyword)
                    confidence = min(0.95, 0.5 + (keyword_count / max(word_count, 1)) * 0.5)
                    
                    return {
                        "emotion": emotion,
                        "confidence": round(confidence, 3)
                    }
        
        return None
    
    def batch_detect_emotions(self, texts):
        """
        Detect emotions for multiple texts.
        
        Args:
            texts (list): List of text strings
        
        Returns:
            list: List of emotion detection results
        """
        results = []
        for text in texts:
            results.append(self.detect_emotion(text))
        return results
    
    def get_emotion_category(self, emotion):
        """
        Get category of emotion for prompt engineering.
        
        Args:
            emotion (str): Emotion name
        
        Returns:
            str: Category name for prompt context
        """
        positive_emotions = ["happy", "grateful"]
        negative_emotions = ["stressed", "confused", "angry", "sad"]
        
        if emotion in positive_emotions:
            return "positive"
        elif emotion in negative_emotions:
            return "negative"
        else:
            return "neutral"
    
    def get_emotion_aware_response_tone(self, emotion):
        """
        Get suggested response tone based on detected emotion.
        
        Args:
            emotion (str): Detected emotion
        
        Returns:
            str: Suggested tone for response
        """
        tone_map = {
            "happy": "cheerful and encouraging",
            "stressed": "calm and reassuring",
            "confused": "clear and detailed",
            "angry": "empathetic and respectful",
            "sad": "supportive and caring",
            "neutral": "professional and informative"
        }
        return tone_map.get(emotion, "professional and informative")
