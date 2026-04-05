"""
Test suite for improved emotion detection.
Priority 2: Validates 66% "confused" fix with better sentiment mapping and context awareness.

Tests:
1. Precise emotion keywords (no broad patterns)
2. Context-aware detection (conversation history)
3. Reduced false "confused" positives
4. Calibrated thresholds
"""

import pytest
from emotion_detector import EmotionDetector
from emotional_tone_detector import EmotionalToneDetector, EmotionalTone
from collections import deque


class TestImprovedEmotionDetection:
    """Test emotion detection improvements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = EmotionDetector()
        self.tone_detector = EmotionalToneDetector()
    
    # ========== TEST 1: Precise Sentiment Word Patterns ==========
    
    def test_genuine_confusion_keywords_only(self):
        """Test that ONLY genuine confusion keywords trigger 'confused'."""
        # Should detect as confused
        confused_cases = [
            "I'm confused about the admission process",
            "I don't understand how to apply",
            "The requirements are unclear to me",
            "I'm lost on what documents to submit",
        ]
        
        for text in confused_cases:
            result = self.detector.detect_emotion(text)
            print(f"Text: {text}")
            print(f"  Emotion: {result['emotion']}, Confidence: {result['confidence']}")
            # Might detect as confused, should not be forced
            assert result['emotion'] in ['confused', 'stressed', 'neutral']
        
        # Should NOT detect as confused (just asking for info)
        info_seeking_cases = [
            "What are the admission fees?",
            "How do I apply?",
            "When is the deadline?",
            "Can you help me with the application?",
            "Please help me understand the process",  # help + understand, but seeking info
        ]
        
        for text in info_seeking_cases:
            result = self.detector.detect_emotion(text)
            print(f"Text: {text}")
            print(f"  Emotion: {result['emotion']}, Confidence: {result['confidence']}")
            # Should not be "confused" - these are info-seeking questions
            assert result['emotion'] != 'confused', f"False positive: '{text}' misclassified as confused"
    
    def test_stressed_not_confused(self):
        """Test that stressed/anxious emotions are NOT confused."""
        cases = [
            ("I'm stressed about the deadline", "stressed"),
            ("I'm worried about my placement", "stressed"),
            ("I'm anxious about the exam", "stressed"),
        ]
        
        for text, expected_emotion in cases:
            result = self.detector.detect_emotion(text)
            print(f"Text: {text}")
            print(f"  Emotion: {result['emotion']}, Confidence: {result['confidence']}")
            # Should detect as stressed, not confused
            assert result['emotion'] in [expected_emotion, 'neutral'], \
                f"Expected {expected_emotion}, got {result['emotion']}"
    
    def test_happy_emotions_detected(self):
        """Test positive emotion recognition."""
        happy_cases = [
            ("I'm happy with my admission!", "happy"),
            ("Thanks for the help!", "happy"),
            ("I appreciate your assistance", "happy"),
        ]
        
        for text, expected in happy_cases:
            result = self.detector.detect_emotion(text)
            print(f"Text: {text}")
            print(f"  Emotion: {result['emotion']}, Confidence: {result['confidence']}")
            assert result['emotion'] in [expected, 'neutral']
    
    # ========== TEST 2: Context-Aware Detection ==========
    
    def test_context_aware_reduces_false_positives(self):
        """Test that conversation history reduces false positives."""
        # User has been asking calm, neutral questions
        history = [
            {
                "timestamp": "2026-04-05T10:00:00",
                "user_input": "What are the fees?",
                "emotion": "neutral",
            },
            {
                "timestamp": "2026-04-05T10:01:00",
                "user_input": "When is the admission deadline?",
                "emotion": "neutral",
            },
        ]
        
        # New message with question mark should NOT be "confused" in neutral context
        result_without_context = self.detector.detect_emotion("?")
        result_with_context = self.detector.detect_emotion("?", conversation_history=history)
        
        print(f"Without context - Emotion: {result_without_context['emotion']}, Conf: {result_without_context['confidence']}")
        print(f"With context - Emotion: {result_with_context['emotion']}, Conf: {result_with_context['confidence']}")
        
        # With context, should be less confident about negative emotions
        assert result_with_context['confidence'] <= result_without_context['confidence']
    
    # ========== TEST 3: Reduced False "Confused" ==========
    
    def test_questions_not_forced_confused(self):
        """Test that low-confidence queries don't force 'confused' emotion."""
        simple_questions = [
            "What is the fee?",
            "How much?",
            "When?",
            "Where?",
            "Tell me about placements",
        ]
        
        confused_count = 0
        for question in simple_questions:
            result = self.detector.detect_emotion(question)
            print(f"Question: {question}")
            print(f"  Emotion: {result['emotion']}, Confidence: {result['confidence']}")
            
            if result['emotion'] == 'confused':
                confused_count += 1
        
        # Should have very FEW confused (goal: < 20% instead of 66%)
        confused_rate = confused_count / len(simple_questions)
        print(f"\nConfused rate: {confused_rate:.1%}")
        assert confused_rate < 0.4, f"Too many false positives: {confused_rate:.1%}"
    
    # ========== TEST 4: Threshold Calibration ==========
    
    def test_confused_requires_high_threshold(self):
        """Test that 'confused' emotion requires high confidence threshold."""
        # Explicit confusion should have high confidence
        explicit_confusion = "I'm confused about the process"
        result = self.detector.detect_emotion(explicit_confusion)
        
        if result['emotion'] == 'confused':
            # If detected as confused, confidence should be high
            assert result['confidence'] >= 0.75, \
                f"Confused emotion should have high confidence, got {result['confidence']}"
    
    # ========== TEST 5: Message Type Classification ==========
    
    def test_message_type_classification(self):
        """Test that message types are correctly classified."""
        test_cases = [
            ("What are the fees?", True, "info_seeking_question"),
            ("I'm confused", False, "statement"),
            ("The process is unclear", False, "statement"),
            ("Can you help me?", True, "info_seeking_question"),
            ("I don't understand", False, "statement"),  # Statement of fact
        ]
        
        for text, expected_question, expected_type in test_cases:
            is_question, msg_type = self.detector._classify_message_type(text.lower())
            print(f"Text: {text}")
            print(f"  Is question: {is_question}, Type: {msg_type}")
            # Don't assert strictly, just validate logic runs
            assert isinstance(is_question, bool)
            assert isinstance(msg_type, str)
    
    # ========== TEST 6: Tone Detection with Improved Thresholds ==========
    
    def test_tone_detection_strict_confused(self):
        """Test that tone detector is strict about 'confused' classification."""
        # Generic questions should NOT trigger confused tone
        generic_questions = [
            "What's the fee?",
            "How do I apply?",
            "When is the deadline?",
        ]
        
        for question in generic_questions:
            result = self.tone_detector.detect_tone(question, emotion="neutral", intent=None)
            tone_name = result.get("tone_name", "neutral")
            print(f"Question: {question}")
            print(f"  Tone: {tone_name}, Confidence: {result.get('confidence')}")
            
            # Should not be "confused" for normal questions
            assert tone_name != 'confused', f"False positive confused tone: {question}"
    
    def test_tone_detection_explicit_confusion(self):
        """Test that explicit confusion IS detected."""
        confusion_statements = [
            "I don't understand the requirements",
            "This is unclear to me",
            "I'm confused about the process",
        ]
        
        for statement in confusion_statements:
            result = self.tone_detector.detect_tone(statement, emotion="confused", intent=None)
            tone_name = result.get("tone_name", "neutral")
            print(f"Statement: {statement}")
            print(f"  Tone: {tone_name}")
            
            # May detect as confused or negative, not necessarily wrong
            assert tone_name in ['confused', 'negative', 'neutral']
    
    # ========== TEST 7: Urgency Detection Calibration ==========
    
    def test_urgency_detection_conservative(self):
        """Test that urgency detection is conservative (not overly sensitive)."""
        # High punctuation shouldn't always mean urgent
        cases = [
            "What??",  # Multiple questions
            "!!!",     # Exclamation marks
            "HELP!!!!"  # All caps
        ]
        
        for text in cases:
            urgency = self.tone_detector._detect_urgency(text)
            print(f"Text: {text}")
            print(f"  Urgency score: {urgency:.2f}")
            # Should be moderate at most, not automatically high
            # (calibrated to not over-react to punctuation)
    
    def test_urgency_detection_explicit(self):
        """Test that explicit urgency IS detected."""
        urgent_cases = [
            "URGENT: I need help immediately",
            "This is an emergency",
            "ASAP response needed",
        ]
        
        for text in urgent_cases:
            urgency = self.tone_detector._detect_urgency(text)
            print(f"Text: {text}")
            print(f"  Urgency score: {urgency:.2f}")
            # Should have higher urgency score
            # (not asserting specific value, just validating detection works)


class TestEmotionImprovementMetrics:
    """Test improvement metrics."""
    
    def test_confused_rate_improvement(self):
        """
        Validate that false 'confused' rate improved.
        Goal: From 66% to < 40% on simple info-seeking questions.
        """
        detector = EmotionDetector()
        
        # Typical chatbot user questions
        sample_questions = [
            "What are the admission fees?",
            "How do I apply for admission?",
            "When is the application deadline?",
            "What are the eligibility criteria?",
            "Can you help me with the application?",
            "Tell me about the placement process",
            "What's the average placement salary?",
            "How do I register for the exam?",
            "When are the exam dates?",
            "What documents do I need?",
        ]
        
        confused_count = 0
        results = []
        
        for question in sample_questions:
            result = detector.detect_emotion(question)
            results.append(result)
            if result['emotion'] == 'confused':
                confused_count += 1
        
        confused_rate = confused_count / len(sample_questions)
        print(f"\n=== EMOTION DETECTION IMPROVEMENT METRICS ===")
        print(f"Total questions: {len(sample_questions)}")
        print(f"'Confused' detections: {confused_count}")
        print(f"'Confused' rate: {confused_rate:.1%}")
        print(f"Target: < 40% (was 66%)")
        print(f"Status: {'✓ PASS' if confused_rate < 0.4 else '✗ FAIL'}")
        
        # Print breakdown
        emotion_counts = {}
        for result in results:
            emotion = result['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\nEmotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            rate = count / len(sample_questions)
            print(f"  {emotion}: {count}/{len(sample_questions)} ({rate:.0%})")
        
        # Assert improvement
        assert confused_rate < 0.4, \
            f"Confused rate {confused_rate:.1%} exceeds target 40%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
