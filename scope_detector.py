"""
Domain scope detector for college assistant.
Determines if a query is within the college domain and returns scope status.
"""

import re
from typing import Tuple


class ScopeDetector:
    """Detects if a query is within college domain scope."""
    
    # College domain keywords organized by category
    DOMAIN_KEYWORDS = {
        "fees": ["fee", "tuition", "cost", "payment", "payment plan", "scholarship", "loan", 
                 "installment", "financial", "expense", "money", "budget", "deposit"],
        "exams": ["exam", "test", "quiz", "assessment", "midterm", "final", "score", 
                  "mark", "result", "pass", "fail", "fail", "grading", "gpa"],
        "timetable": ["schedule", "timetable", "timing", "class time", "when", "time", 
                      "term", "semester", "session", "calendar", "dates"],
        "placements": ["placement", "job", "company", "recruitment", "salary", "internship", 
                       "interview", "campus drive", "hiring", "career", "interview"],
        "faculty": ["faculty", "professor", "teacher", "lecturer", "instructor", "staff", 
                    "contact", "office", "department", "department head"],
        "library": ["library", "book", "resource", "database", "journal", "research", 
                    "e-book", "reference", "study"],
        "admission": ["admission", "apply", "application", "cutoff", "entrance", "merit", 
                      "requirement", "eligibility", "document", "rejection"],
        "college_info": ["college", "institution", "university", "campus", "about", 
                         "facility", "location", "history"],
        "hostel": ["hostel", "accommodation", "room", "boarding", "residence", "stay", 
                   "mess", "food"],
        "sports": ["sports", "cricket", "basketball", "gym", "athletics", "tournament", 
                   "sports day", "physical", "fitness"]
    }
    
    # Out-of-scope keywords that trigger generic "out of scope" responses
    OUT_OF_SCOPE_INDICATORS = [
        # Technology/Science
        "quantum", "relativity", "physics", "chemistry", "biology", "programming language", 
        "algorithm", "machine learning", "artificial intelligence", "deep learning", 
        "neural network", "python", "java", "javascript", "react", "angular",
        
        # Politics/Current affairs
        "election", "government", "politics", "politician", "president", "minister", 
        "parliament", "war", "conflict", "international",
        
        # Entertainment
        "movie", "actor", "actress", "music", "song", "cricket match", "football", 
        "celebrity", "bollywood", "hollywood", "netflix", "song",
        
        # Personal/Medical
        "health", "disease", "medicine", "doctor", "hospital", "symptom", "diet", 
        "weight", "exercise", "mental health",
        
        # Philosophy/Abstract
        "meaning of life", "god", "religion", "spirituality", "consciousness", 
        "metaphysics", "existential",
        
        # Other domains
        "gardening", "cooking", "travel recommendation", "real estate", "law", 
        "taxes", "investment", "stock"
    ]
    
    def __init__(self):
        """Initialize scope detector."""
        self.confidence_threshold = 0.3
    
    def is_in_scope(self, query: str, detected_intent: str = "", intent_confidence: float = 0.5) -> Tuple[bool, str, float]:
        """
        Determine if query is within college domain scope.
        
        Args:
            query (str): User's query
            detected_intent (str): Detected intent from intent classifier
            intent_confidence (float): Confidence of intent detection
        
        Returns:
            Tuple[bool, str, float]: (is_in_scope, reason, scope_confidence)
        """
        query_lower = query.lower()
        
        # Check for explicit out-of-scope indicators
        out_of_scope_score = self._check_out_of_scope(query_lower)
        if out_of_scope_score > 0.7:
            return False, "out_of_domain", out_of_scope_score
        
        # Check for college domain keywords
        domain_score = self._check_domain_keywords(query_lower)
        
        # Check detected intent relevance
        if detected_intent and detected_intent in ["college_info", "fees", "exams", "timetable", 
                                                     "placements", "faculty", "library", "admission", 
                                                     "hostel", "sports"]:
            # If intent classifier is confident, trust it
            if intent_confidence > 0.4:
                return True, f"detected_intent_{detected_intent}", intent_confidence
        
        # Final decision based on keyword analysis
        if domain_score > self.confidence_threshold:
            return True, "domain_keywords", domain_score
        
        # If still ambiguous and low keyword match, default to out of scope
        return False, "low_domain_confidence", domain_score
    
    def _check_out_of_scope(self, query: str) -> float:
        """
        Check if query contains out-of-scope indicators.
        Returns a confidence score for being out-of-scope.
        """
        out_of_scope_count = 0
        total_indicators_checked = len(self.OUT_OF_SCOPE_INDICATORS)
        
        for indicator in self.OUT_OF_SCOPE_INDICATORS:
            if indicator.lower() in query:
                out_of_scope_count += 1
        
        # Return confidence that this is out of scope
        if out_of_scope_count == 0:
            return 0.0
        
        return min(out_of_scope_count / total_indicators_checked, 1.0)
    
    def _check_domain_keywords(self, query: str) -> float:
        """
        Check how many domain keywords match the query.
        Returns a confidence score for being in-domain.
        """
        max_matches = 0
        best_category = None
        
        for category, keywords in self.DOMAIN_KEYWORDS.items():
            category_matches = sum(1 for keyword in keywords if keyword.lower() in query)
            
            if category_matches > max_matches:
                max_matches = category_matches
                best_category = category
        
        # Normalize: 1 match = high confidence, more matches = even higher
        if max_matches == 0:
            return 0.0
        elif max_matches == 1:
            return 0.5
        elif max_matches <= 3:
            return 0.7
        else:
            return 0.9
    
    def get_scope_info(self, query: str, detected_intent: str = "", intent_confidence: float = 0.5) -> dict:
        """
        Get detailed scope information for a query.
        
        Returns:
            dict: Detailed scope analysis including in_scope, confidence, category, etc.
        """
        is_in_scope, reason, confidence = self.is_in_scope(query, detected_intent, intent_confidence)
        
        return {
            "is_in_scope": is_in_scope,
            "reason": reason,
            "confidence": confidence,
            "should_clarify": confidence < 0.5 and is_in_scope,
            "out_of_scope_response": "This is beyond my scope as a college assistant. I can help with fees, exams, placements, faculty, library, hostel, admission queries, and more college-related information."
        }
