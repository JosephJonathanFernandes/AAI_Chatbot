"""
Enhanced emotional tone detection for adaptive response tailoring.
Combines emotion + intent + context for sophisticated tone-aware responses.
"""

import re
from typing import Dict, List, Tuple
from enum import Enum


class EmotionalTone(Enum):
    """Emotional tones for response adaptation."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    URGENT = "urgent"
    CONFUSED = "confused"


class EmotionalToneDetector:
    """Detects emotional tone for adaptive response generation."""
    
    # Intensity markers
    INTENSITY_MARKERS = {
        "very_strong": ["!!!!", "???", "URGENT", "EMERGENCY", "ASAP", "IMMEDIATELY"],
        "strong": ["!!!", "??", "URGENT", "HELP", "STRESS"],
        "moderate": ["!!", "?", "please", "quick"],
        "mild": [".", "?", "maybe", "might"]
    }
    
    # Tone patterns
    TONE_PATTERNS = {
        "urgent": {
            "keywords": ["urgent", "asap", "immediately", "emergency", "critical", "now"],
            "patterns": [r"!!+", r"\?\?+", r"URGENT|EMERGENCY"],
            "context": ["placement", "result", "deadline", "admission"],
            "min_confidence": 0.75
        },
        "confused": {
            # IMPROVED: Much stricter patterns for confused
            # Removed generic "what", "how", "why" - these are just questions
            # Confused requires explicit statements of not understanding
            "keywords": ["confused", "don't understand", "unclear", "don't know", "bewildered", "lost"],
            "patterns": [r"i don't understand", r"(unclear|confused|lost|bewildered)", r"can't figure out"],
            "context": [],  # Don't filter - if keyword matches, it's likely genuine
            "min_confidence": 0.80,  # High threshold
            "exclude_patterns": [r"can you.*help", r"what.*is", r"how.*to", r"where.*can"]
        },
        "frustrated": {
            "keywords": ["frustrated", "annoyed", "stuck", "problem", "issue", "not working"],
            "patterns": [r"(!{2,})", r"frustrated|annoyed|stuck"],
            "context": ["process", "system", "issue"],
            "min_confidence": 0.70
        },
        "happy": {
            "keywords": ["thank", "great", "excellent", "happy", "thanks", "good"],
            "patterns": [r"(:D)|(\^_\^)"],
            "context": ["admission", "placement", "result"],
            "min_confidence": 0.70
        },
        "stressed": {
            "keywords": ["stressed", "anxious", "worried", "pressure", "deadline"],
            "patterns": [r":/\s|worried|stressed"],
            "context": ["exam", "placement", "result"],
            "min_confidence": 0.70
        }
    }
    
    # Tone-specific response modifiers
    TONE_MODIFIERS = {
        "urgent": {
            "prefix": "🚨 Priority: ",
            "suffix": "\n⏰ I'll be brief and direct.",
            "tone": "concise_direct",
            "emphasis": "critical"
        },
        "confused": {
            "prefix": "📖 Let me clarify: ",
            "suffix": "\n💡 Does this help? Ask if you need more detail.",
            "tone": "clear_detailed",
            "emphasis": "educational"
        },
        "frustrated": {
            "prefix": "✅ Let's fix this: ",
            "suffix": "\n🤝 I'm here to help. Let me know if this resolves it.",
            "tone": "supportive_solution_focused",
            "emphasis": "supportive"
        },
        "happy": {
            "prefix": "🎉 Great question! ",
            "suffix": "\n😊 Glad to help!",
            "tone": "warm_encouraging",
            "emphasis": "positive"
        },
        "stressed": {
            "prefix": "🧘 Don't worry: ",
            "suffix": "\n✨ You've got this!",
            "tone": "reassuring_calm",
            "emphasis": "supportive"
        },
        "neutral": {
            "prefix": "📚 ",
            "suffix": "",
            "tone": "informative",
            "emphasis": "neutral"
        }
    }
    
    def __init__(self):
        """Initialize emotional tone detector."""
        self.detected_tones = []
    
    def detect_tone(self, 
                   text: str, 
                   emotion: str = "neutral",
                   intent: str = None) -> Dict:
        """
        Detect the emotional tone of the text.
        
        Args:
            text (str): User input text
            emotion (str): Pre-detected emotion
            intent (str): Detected intent
            
        Returns:
            Dict: Tone detection result with confidence
        """
        try:
            text_lower = text.lower()
            
            # Detect urgency
            urgency_score = self._detect_urgency(text)
            
            # Detect primary tone
            primary_tone = self._detect_primary_tone(text_lower, emotion)
            
            # Compute confidence
            confidence = self._calculate_tone_confidence(text, primary_tone, urgency_score)
            
            # Adjust tone based on urgency
            if urgency_score > 0.8 and primary_tone != EmotionalTone.URGENT:
                adjusted_tone = EmotionalTone.URGENT
            else:
                adjusted_tone = primary_tone
            
            result = {
                "tone": adjusted_tone,
                "tone_name": adjusted_tone.value,
                "confidence": confidence,
                "urgency_score": urgency_score,
                "emotion": emotion,
                "intent": intent,
                "modifiers": self.TONE_MODIFIERS.get(adjusted_tone.value, self.TONE_MODIFIERS["neutral"]),
                "explanation": self._get_tone_explanation(adjusted_tone, urgency_score)
            }
            
            self.detected_tones.append(result)
            return result
        
        except Exception as e:
            print(f"Error in detect_tone: {e}")
            # Return safe default
            return {
                "tone": EmotionalTone.NEUTRAL,
                "tone_name": "neutral",
                "confidence": 0.0,
                "urgency_score": 0.0,
                "emotion": emotion or "neutral",
                "intent": intent,
                "modifiers": self.TONE_MODIFIERS.get("neutral", {"prefix": "", "suffix": "", "emphasis": "normal"}),
                "explanation": "Default response"
            }
    
    def get_response_guidelines(self, tone_result: Dict) -> Dict:
        """
        Get response generation guidelines based on tone.
        
        Args:
            tone_result (Dict): Result from detect_tone()
            
        Returns:
            Dict: Guidelines for LLM response generation
        """
        try:
            if not tone_result or not isinstance(tone_result, dict):
                tone_result = {
                    "tone_name": "neutral",
                    "modifiers": self.TONE_MODIFIERS.get("neutral", {"emphasis": "normal", "prefix": "", "suffix": ""})
                }
            
            tone_name = tone_result.get("tone_name", "neutral")
            modifiers = tone_result.get("modifiers", self.TONE_MODIFIERS.get("neutral", {"emphasis": "normal", "prefix": "", "suffix": ""}))
            
            guidelines = {
                "tone": tone_name,
                "length": self._get_recommended_length(tone_name),
                "formality": self._get_formality_level(tone_name),
                "detail_level": self._get_detail_level(tone_name),
                "structure": self._get_response_structure(tone_name),
                "emphasis": modifiers.get("emphasis", "normal"),
                "prefix": modifiers.get("prefix", ""),
                "suffix": modifiers.get("suffix", ""),
            }
            
            return guidelines
        except Exception as e:
            print(f"Error in get_response_guidelines: {e}")
            # Return safe default
            return {
                "tone": "neutral",
                "length": "standard",
                "formality": "professional",
                "detail_level": "appropriate",
                "structure": "linear",
                "emphasis": "normal",
                "prefix": "",
                "suffix": "",
            }
    
    def _detect_urgency(self, text: str) -> float:
        """Detect urgency indicators with improved calibration."""
        score = 0.0
        text_lower = text.lower()
        
        # Check for urgent keywords (stricter)
        urgent_keywords = ["urgent", "asap", "emergency", "immediately", "critical", "now"]
        matches = sum(1 for kw in urgent_keywords if f" {kw} " in f" {text_lower} ")
        score += matches * 0.25  # Increased from 0.2 to be more selective
        
        # Check for punctuation intensity (more conservative)
        exc_count = text.count("!")
        question_count = text.count("?")
        # Reduce contribution from mere punctuation
        score += min(exc_count * 0.10, 0.25)  # Reduced from 0.15
        score += min(question_count * 0.05, 0.10)  # Reduced from 0.1
        
        # Check for caps intensity (more conservative)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        # Long all-caps messages are less likely; short ones might be typos
        if caps_ratio > 0.5 and len(text) > 15:
            score += min(caps_ratio * 0.3, 0.25)  # Reduced from 0.4
        
        return min(score, 1.0)
    
    def _detect_primary_tone(self, text_lower: str, emotion: str) -> EmotionalTone:
        """Detect primary emotional tone with improved threshold checking."""
        # Check tone patterns
        for tone_name, patterns in self.TONE_PATTERNS.items():
            min_confidence = patterns.get("min_confidence", 0.5)
            exclude_patterns = patterns.get("exclude_patterns", [])
            
            # Check for exclusion patterns first (especially for "confused")
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if re.search(exclude_pattern, text_lower):
                    should_exclude = True
                    break
            
            if should_exclude:
                continue
            
            # Check keywords
            found_keyword = False
            for keyword in patterns.get("keywords", []):
                if keyword in text_lower:
                    found_keyword = True
                    break
            
            if found_keyword:
                tone_map = {
                    "urgent": EmotionalTone.URGENT,
                    "confused": EmotionalTone.CONFUSED,
                    "frustrated": EmotionalTone.NEGATIVE,
                    "happy": EmotionalTone.POSITIVE,
                    "stressed": EmotionalTone.NEGATIVE,
                }
                return tone_map.get(tone_name, EmotionalTone.NEUTRAL)
            
            # Check regex patterns
            found_pattern = False
            for pattern in patterns.get("patterns", []):
                if re.search(pattern, text_lower):
                    found_pattern = True
                    break
            
            if found_pattern:
                tone_map = {
                    "urgent": EmotionalTone.URGENT,
                    "confused": EmotionalTone.CONFUSED,
                    "frustrated": EmotionalTone.NEGATIVE,
                    "happy": EmotionalTone.POSITIVE,
                    "stressed": EmotionalTone.NEGATIVE,
                }
                return tone_map.get(tone_name, EmotionalTone.NEUTRAL)
        
        # Fall back to emotion-based mapping
        emotion_to_tone = {
            "happy": EmotionalTone.POSITIVE,
            "stressed": EmotionalTone.NEGATIVE,
            "angry": EmotionalTone.VERY_NEGATIVE,
            "sad": EmotionalTone.NEGATIVE,
            "confused": EmotionalTone.CONFUSED,
            "neutral": EmotionalTone.NEUTRAL
        }
        
        return emotion_to_tone.get(emotion, EmotionalTone.NEUTRAL)
    
    def _calculate_tone_confidence(self, text: str, tone: EmotionalTone, urgency: float) -> float:
        """Calculate confidence in tone detection with minimum thresholds."""
        base_confidence = 0.6
        
        # Boost confidence for urgent tone with high urgency score
        if tone == EmotionalTone.URGENT and urgency > 0.7:
            base_confidence = 0.9
        
        # Boost confidence for longer contextualized text
        if len(text) > 50:
            base_confidence += 0.15
        
        # IMPROVED: Reduce confidence for confused tone (was too high)
        # to prevent false positives on simple questions
        if tone == EmotionalTone.CONFUSED:
            # Only high confidence for genuine confusion signals
            text_lower = text.lower()
            has_genuine_confusion = any(
                phrase in text_lower 
                for phrase in ["don't understand", "unclear", "confused", "lost", "bewildered"]
            )
            if not has_genuine_confusion and len(text) <= 10:
                # Short messages without explicit confusion keyword
                base_confidence = 0.3
        
        return min(base_confidence, 1.0)
    
    def _get_tone_explanation(self, tone: EmotionalTone, urgency: float) -> str:
        """Get explanation for detected tone."""
        explanations = {
            EmotionalTone.URGENT: f"Urgent tone detected (urgency: {urgency:.0%}). Quick, direct response needed.",
            EmotionalTone.CONFUSED: "User seems confused. Clear, detailed explanation needed.",
            EmotionalTone.NEGATIVE: "User appears frustrated. Supportive, solution-focused response needed.",
            EmotionalTone.POSITIVE: "Positive tone detected. Warm, encouraging response appropriate.",
            EmotionalTone.VERY_NEGATIVE: "Very negative tone. Highly supportive response required.",
            EmotionalTone.NEUTRAL: "Neutral tone. Informative response appropriate."
        }
        
        return explanations.get(tone, "Neutral tone.")
    
    def _get_recommended_length(self, tone_name: str) -> str:
        """Get recommended response length."""
        length_map = {
            "urgent": "brief",
            "confused": "detailed",
            "frustrated": "medium",
            "happy": "medium",
            "stressed": "medium",
            "positive": "medium",
            "negative": "medium",
            "very_negative": "detailed",
            "neutral": "standard"
        }
        return length_map.get(tone_name, "standard")
    
    def _get_formality_level(self, tone_name: str) -> str:
        """Get recommended formality level."""
        formality_map = {
            "urgent": "formal_professional",
            "confused": "casual_clear",
            "frustrated": "warm_professional",
            "happy": "friendly_professional",
            "stressed": "empathetic_professional",
            "positive": "friendly",
            "negative": "supportive",
            "very_negative": "highly_supportive",
            "neutral": "professional"
        }
        return formality_map.get(tone_name, "professional")
    
    def _get_detail_level(self, tone_name: str) -> str:
        """Get recommended detail level."""
        detail_map = {
            "urgent": "high_level_actionable",
            "confused": "very_detailed_step_by_step",
            "frustrated": "focused_relevant",
            "happy": "balanced",
            "stressed": "reassuring_concise",
            "positive": "balanced",
            "negative": "comprehensive",
            "very_negative": "very_comprehensive",
            "neutral": "appropriate"
        }
        return detail_map.get(tone_name, "appropriate")
    
    def _get_response_structure(self, tone_name: str) -> List[str]:
        """Get recommended response structure."""
        structures = {
            "urgent": ["action_item", "key_info", "next_step"],
            "confused": ["simple_explanation", "example", "clarification", "offer_help"],
            "frustrated": ["acknowledgment", "solution", "support_offer"],
            "happy": ["positive_response", "info", "encouragement"],
            "stressed": ["reassurance", "information", "support"],
            "positive": ["positive_response", "information"],
            "negative": ["empathy", "solution", "encouragement"],
            "very_negative": ["empathy", "support", "solution", "resources"],
            "neutral": ["information", "context"]
        }
        return structures.get(tone_name, ["information"])
    
    def get_tone_history(self) -> List[Dict]:
        """Get history of detected tones."""
        return self.detected_tones
