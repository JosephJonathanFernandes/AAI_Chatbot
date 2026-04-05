"""
Domain scope detector for college assistant.
Determines if a query is within the college domain and returns scope status.
Enhanced with:
- Expanded domain keywords (10+ categories)
- Semantic similarity for fuzzy matching
- Context-aware scope checking
- Optimized confidence thresholds
"""

import re
from typing import Tuple, Optional, List
import warnings

# Optional semantic similarity (graceful fallback if not available)
try:
    from sentence_transformers import util
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    warnings.warn("Semantic similarity unavailable. Install: pip install sentence-transformers")


class ScopeDetector:
    """Detects if a query is within college domain scope with semantic and keyword analysis."""
    
    # College domain keywords organized by category (EXPANDED)
    DOMAIN_KEYWORDS = {
        # Original categories
        "fees": ["fee", "tuition", "cost", "payment", "payment plan", "scholarship", "loan", 
                 "installment", "financial", "expense", "money", "budget", "deposit", "refund",
                 "grant", "aid", "subsidy", "waiver"],
        
        "exams": ["exam", "test", "quiz", "assessment", "midterm", "final", "score", 
                  "mark", "result", "pass", "fail", "grading", "gpa", "grade", "marks",
                  "examination", "paper", "syllabus", "study"],
        
        "timetable": ["schedule", "timetable", "timing", "class time", "term", 
                      "semester", "session", "calendar", "dates", "when", "time",
                      "class timing", "lecture timing", "start time", "end time"],
        
        "placements": ["placement", "job", "company", "recruitment", "salary", "internship", 
                       "interview", "campus drive", "hiring", "career", "package",
                       "placement rate", "company visit", "selection"],
        
        "faculty": ["faculty", "professor", "teacher", "lecturer", "instructor", "staff", 
                    "contact", "office", "department", "department head", "head of department",
                    "mentor", "guide", "advisor", "counselor"],
        
        "library": ["library", "book", "resource", "database", "journal", "research", 
                    "reference", "study room", "reading room", "archive", "e-book"],
        
        "admission": ["admission", "apply", "application", "cutoff", "entrance", "merit", 
                      "requirement", "eligibility", "document", "enroll", "enrollment",
                      "admission process", "merit list", "selection process"],
        
        "college_info": ["college", "institution", "university", "campus", "curriculum",
                         "course", "degree", "faculty", "facility", "facilities", "location", "history",
                         "about college", "college about", "about university"],
        
        "hostel": ["hostel", "accommodation", "room", "boarding", "residence", "stay", 
                   "mess", "dorm", "dorms", "dormitory", "living", "housing"],
        
        "sports": ["sports", "cricket", "basketball", "gym", "athletics", "tournament",
                   "physical", "exercise", "training", "competition", "ground"],
        
        # NEW: Academic/Academic Structure
        "academics": ["major", "minor", "degree", "specialization", "branch", "program",
                      "graduation", "graduate", "undergraduate", "bachelor", "master",
                      "curriculum", "course structure", "course load", "credit", "prerequisite"],
        
        # NEW: Student Facilities & Services
        "student_facilities": ["club", "society", "activity", "student center", "café",
                               "cafeteria", "restaurant", "food", "medical", "health center",
                               "health", "clinic", "doctor", "emergency", "counseling"],
        
        # NEW: Campus & Amenities
        "campus_amenities": ["parking", "transport", "bus", "vehicle", "commute",
                             "wifi", "internet", "laptop", "computer", "lab",
                             "dress code", "uniform", "code of conduct", "rules"],
        
        # NEW: Student Documents & Records
        "student_records": ["transcript", "certificate", "document", "mark sheet",
                            "result sheet", "verification", "degree", "diploma",
                            "letter of recommendation", "reference letter"],
    }
    
    # Semantic keyword patterns for fuzzy matching
    SEMANTIC_PATTERNS = {
        "academics": ["change major", "switch major", "declare major", "choose specialization",
                      "graduation date", "when graduate", "graduate process", "honors"],
        "student_facilities": ["student clubs", "campus activities", "where eat", "food options",
                               "health services", "medical help", "counseling services"],
        "campus_amenities": ["where park", "parking available", "transportation", "how commute",
                             "wifi available", "internet", "dress requirement"],
        "student_records": ["get transcript", "request document", "transcript copy", "official document"],
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
    
    def __init__(self, use_semantic: bool = True):
        """Initialize scope detector with optional semantic similarity."""
        self.confidence_threshold = 0.3
        self.use_semantic = use_semantic and SEMANTIC_AVAILABLE
        self.definite_oos_pattern = re.compile("|".join(self.DEFINITE_OUT_OF_SCOPE), re.IGNORECASE)
        self.ambiguous_oos_pattern = re.compile("|".join(self.AMBIGUOUS_OUT_OF_SCOPE), re.IGNORECASE)
        
        # Initialize semantic model if available
        self.semantic_model = None
        if self.use_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._encode_semantic_keywords()
            except Exception as e:
                warnings.warn(f"Failed to load semantic model: {e}")
                self.use_semantic = False
    
    def _encode_semantic_keywords(self):
        """Pre-encode semantic patterns for similarity matching."""
        if not self.semantic_model:
            return
        
        # Flatten all semantic patterns
        self.semantic_phrases = []
        self.semantic_categories = []
        
        for category, phrases in self.SEMANTIC_PATTERNS.items():
            for phrase in phrases:
                self.semantic_phrases.append(phrase)
                self.semantic_categories.append(category)
        
        # Encode all phrases
        if self.semantic_phrases:
            self.semantic_embeddings = self.semantic_model.encode(
                self.semantic_phrases, convert_to_tensor=True
            )
    
    def is_in_scope(self, query: str, detected_intent: str = "", intent_confidence: float = 0.5,
                    conversation_history: Optional[List[str]] = None) -> Tuple[bool, str, float]:
        """
        Determine if query is within college domain scope.
        Uses multi-level analysis: definite rules → contextual analysis → keyword matching → semantic similarity.
        
        Args:
            query (str): User's query
            detected_intent (str): Detected intent from intent classifier
            intent_confidence (float): Confidence of intent detection
            conversation_history (List[str]): Previous queries for context awareness
        
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
        
        # Level 4: Check semantic similarity (NEW)
        if self.use_semantic and domain_score == 0.0:
            semantic_score, semantic_category = self._check_semantic_similarity(query)
            if semantic_score > domain_score:
                domain_score, matched_category = semantic_score, semantic_category
        
        # Level 5: Context-aware checking (NEW - use conversation history)
        context_boost = 0.0
        if conversation_history:
            context_boost = self._compute_context_score(query_lower, conversation_history)
        
        # Apply context boost if available
        if context_boost > 0:
            domain_score = min(0.95, domain_score + context_boost)
        
        # Level 6: Use intent classifier if confident
        if detected_intent and detected_intent in [
            "college_info", "fees", "exams", "timetable", "placements", 
            "faculty", "library", "admission", "hostel", "sports"
        ]:
            if intent_confidence > 0.4:
                return True, f"detected_intent_{detected_intent}", intent_confidence
        
        # Level 7: Final decision based on keyword analysis
        if domain_score > self.confidence_threshold:
            return True, f"domain_keywords_{matched_category}", domain_score
        
        # Level 8: Default to out-of-scope if no matches
        return False, "low_domain_confidence", max(domain_score, intent_confidence * 0.3)
    
    def _check_semantic_similarity(self, query: str) -> Tuple[float, str]:
        """
        Check semantic similarity with college domain patterns.
        Uses pre-encoded semantic patterns for fuzzy matching.
        
        Args:
            query (str): User's query
        
        Returns:
            Tuple[float, str]: (confidence_score, category_name)
        """
        if not self.use_semantic or not self.semantic_model:
            return 0.0, "unknown"
        
        try:
            # Encode the query
            query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
            
            # Compute similarities
            similarities = util.pytorch_cos_sim(query_embedding, self.semantic_embeddings)[0]
            
            # Find best match
            max_sim = max(similarities).item() if len(similarities) > 0 else 0.0
            
            if max_sim > 0.5:  # Similarity threshold
                best_idx = similarities.argmax().item()
                category = self.semantic_categories[best_idx]
                # Map semantic score to confidence (0.5-0.95 range)
                confidence = 0.55 + (max_sim - 0.5) * 0.8  # Scale to 0.55-0.95
                return confidence, category
            
            return 0.0, "unknown"
        
        except Exception as e:
            warnings.warn(f"Semantic similarity error: {e}")
            return 0.0, "unknown"
    
    def _compute_context_score(self, query: str, history: List[str]) -> float:
        """
        Compute context-aware confidence boost based on conversation history.
        
        Args:
            query (str): Current query
            history (List[str]): Previous queries
        
        Returns:
            float: Context boost (0.0-0.3)
        """
        if not history:
            return 0.0
        
        # Check if previous queries were college-related
        college_keywords_in_history = 0
        for prev_query in history[-3:]:  # Check last 3 queries
            prev_lower = prev_query.lower()
            for category_keywords in self.DOMAIN_KEYWORDS.values():
                if any(kw in prev_lower for kw in category_keywords):
                    college_keywords_in_history += 1
                    break
        
        # If recent history is college-related, boost confidence
        if college_keywords_in_history > 0:
            return min(0.3, college_keywords_in_history * 0.1)
        
        return 0.0
    
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
            return 0.7, best_category
        elif max_matches == 2:
            return 0.85, best_category
        else:
            return 0.95, best_category
    
    def get_scope_info(self, query: str, detected_intent: str = "", intent_confidence: float = 0.5,
                      conversation_history: Optional[List[str]] = None) -> dict:
        """
        Get detailed scope information for a query.
        
        Args:
            query (str): User's query
            detected_intent (str): Detected intent
            intent_confidence (float): Intent confidence
            conversation_history (List[str]): Previous queries for context
        
        Returns:
            dict: Detailed scope analysis including in_scope, confidence, category, etc.
        """
        is_in_scope, reason, confidence = self.is_in_scope(
            query, detected_intent, intent_confidence, conversation_history
        )
        
        return {
            "is_in_scope": is_in_scope,
            "reason": reason,
            "confidence": confidence,
            "should_clarify": confidence < 0.5 and is_in_scope,
            "semantic_available": self.use_semantic,
            "out_of_scope_response": "This is beyond my scope as a college assistant. I can help with admissions, fees, exams, placements, faculty, library, hostel, sports, academics, student facilities, campus amenities, and student records."
        }

