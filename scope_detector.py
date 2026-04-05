"""
Domain scope detector for college assistant.
Determines if a query is within the college domain and returns scope status.
Enhanced with contextual rules, word boundaries, and confidence scoring.
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
                  "mark", "result", "pass", "fail", "grading", "gpa"],
        "timetable": ["schedule", "timetable", "timing", "class time", "term", 
                      "semester", "session", "calendar", "dates", "when"],
        "placements": ["placement", "job", "company", "recruitment", "salary", "internship", 
                       "interview", "campus drive", "hiring", "career"],
        "faculty": ["faculty", "professor", "teacher", "lecturer", "instructor", "staff", 
                    "contact", "office", "department", "department head"],
        "library": ["library", "book", "resource", "database", "journal", "research", 
                    "reference", "study room"],
        "admission": ["admission", "apply", "application", "cutoff", "entrance", "merit", 
                      "requirement", "eligibility", "document"],
        "college_info": ["college", "institution", "university", "campus", "curriculum",
                         "course", "degree", "faculty", "facility", "facilities", "location", "history"],
        "hostel": ["hostel", "accommodation", "room", "boarding", "residence", "stay", 
                   "mess"],
        "sports": ["sports", "cricket", "basketball", "gym", "athletics", "tournament"]
    }
    
    # DEFINITE out-of-scope: Always out of scope (high confidence)
    DEFINITE_OUT_OF_SCOPE = [
        # Action-based (asking for help with non-college tasks)
        r"\bwrite\s+(my\s+)?code\b", r"\bdo\s+my\s+(assignment|homework|project)\b",
        r"\bcomplete\s+my\b", r"\bwrite\s+my\b", r"\bhelp\s+me\s+(write|code|debug)\b",
        
        # Cooking/Food (not college-related)
        r"\b(cook|recipe|cooking|how\s+to\s+cook)\b",
        
        # Entertainment (movies, celebrities, music)
        r"\b(movie|film|actor|actress|celebrity|bollywood|hollywood|netflix)\b",
        r"\b(music|concert|singer|band)\s+(?!college)",
        
        # Hobby/Personal
        r"\bgarden(ing)?\b", r"\btravel\s+recommendation\b", r"\bhow\s+to\s+(fix|repair|build|make)(?!\s+a\s+study)",
        
        # Finance/Investment
        r"\b(stock|bitcoin|investment|tax|trading)\b",
        r"\b(real\s+estate|property|mortgage)\b",
    ]
    
    # AMBIGUOUS out-of-scope: May be out of scope depending on context
    AMBIGUOUS_OUT_OF_SCOPE = [
        r"\bmachine\s+learning\b", r"\bdeep\s+learning\b", r"\bneural\s+network\b",
        r"\bartificial\s+intelligence\b", r"\bpython\s+(programming|code)\b", r"\bjava\s+(programming)\b",
        r"\bquantum\b", r"\bphysics\b", r"\bchemistry\b", r"\bbiology\b",
    ]
    
    # College context prefixes that make ambiguous queries in-scope
    COLLEGE_CONTEXT_PREFIXES = [
        "in college", "in campus", "at college", "at university", 
        "college", "campus", "course", "student", "professor tells",
        "my professor", "my college", "our college", "the college"
    ]
    
    def __init__(self):
        """Initialize scope detector."""
        self.confidence_threshold = 0.3
        self.definite_oos_pattern = re.compile("|".join(self.DEFINITE_OUT_OF_SCOPE), re.IGNORECASE)
        self.ambiguous_oos_pattern = re.compile("|".join(self.AMBIGUOUS_OUT_OF_SCOPE), re.IGNORECASE)
    
    def is_in_scope(self, query: str, detected_intent: str = "", intent_confidence: float = 0.5) -> Tuple[bool, str, float]:
        """
        Determine if query is within college domain scope.
        Uses multi-level analysis: definite rules → contextual analysis → keyword matching.
        
        Args:
            query (str): User's query
            detected_intent (str): Detected intent from intent classifier
            intent_confidence (float): Confidence of intent detection
        
        Returns:
            Tuple[bool, str, float]: (is_in_scope, reason, scope_confidence)
        """
        # Safety check: handle None or empty queries
        if not query or not isinstance(query, str):
            return True, "empty_query", 0.5
        
        query_lower = query.lower()
        
        # Level 0: Check for college context FIRST (override ambiguous queries)
        has_college_context = any(prefix in query_lower for prefix in self.COLLEGE_CONTEXT_PREFIXES)
        if has_college_context:
            return True, "college_context_detected", 0.85
        
        # Level 1: Check for DEFINITE out-of-scope indicators (high confidence)
        if self.definite_oos_pattern.search(query_lower):
            return False, "definite_out_of_domain", 0.95
        
        # Level 2: Check for ambiguous out-of-scope indicators
        if self.ambiguous_oos_pattern.search(query_lower):
            return False, "ambiguous_out_of_domain", 0.65
        
        # Level 3: Check for college domain keywords
        domain_score, matched_category = self._check_domain_keywords(query_lower)
        
        # Level 4: Use intent classifier if confident
        if detected_intent and detected_intent in [
            "college_info", "fees", "exams", "timetable", "placements", 
            "faculty", "library", "admission", "hostel", "sports"
        ]:
            if intent_confidence > 0.4:
                return True, f"detected_intent_{detected_intent}", intent_confidence
        
        # Level 5: Final decision based on keyword analysis
        if domain_score > self.confidence_threshold:
            return True, f"domain_keywords_{matched_category}", domain_score
        
        # Level 6: Default to out-of-scope if no matches
        return False, "low_domain_confidence", max(domain_score, intent_confidence * 0.3)
    
    def _check_domain_keywords(self, query: str) -> Tuple[float, str]:
        """
        Check how many domain keywords match the query.
        Returns confidence score and best matching category.
        Uses substring matching to handle plurals and word variations.
        
        Args:
            query (str): Lowercase query text
        
        Returns:
            Tuple[float, str]: (confidence_score, category_name)
        """
        max_matches = 0
        best_category = "unknown"
        
        for category, keywords in self.DOMAIN_KEYWORDS.items():
            # Count matches using substring approach (handles plurals better)
            category_matches = 0
            for keyword in keywords:
                # Use word boundaries at start, but allow natural word endings
                if re.search(r'\b' + re.escape(keyword.lower()), query):
                    category_matches += 1
            
            if category_matches > max_matches:
                max_matches = category_matches
                best_category = category
        
        # Convert matches to confidence score
        if max_matches == 0:
            # Special case: empty query might still be in-scope (user just wants help)
            if query.strip() == "":
                return 0.5, "empty_query"
            return 0.0, best_category
        elif max_matches == 1:
            return 0.7, best_category  # Increased from 0.6
        elif max_matches == 2:
            return 0.85, best_category  # Increased from 0.8
        else:
            return 0.95, best_category
    
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
