"""
Text preprocessing module for robust intent classification.
Handles typos, Hinglish, slang, and casual variations.
"""

import re
import string
from typing import List


class TextPreprocessor:
    """Preprocesses text to handle typos, Hinglish, slang, and casual variations."""
    
    # Common typos -> correct mapping
    TYPO_MAP = {
        "fe": "fee",
        "fes": "fee",
        "fees?": "fee",
        "tution": "tuition",
        "expam": "exam",
        "exams?": "exam",
        "placment": "placement",
        "placements?": "placement",
        "admision": "admission",
        "admissions?": "admission",
        "hostl": "hostel",
        "hostels?": "hostel",
        "libary": "library",
        "librarys?": "library",
        "scholorship": "scholarship",
        "scholarships?": "scholarship",
        "collage": "college",
        "colleges?": "college",
        "university": "university",
        "campous": "campus",
        "departmnt": "department",
        "departments?": "department",
        "enginering": "engineering",
        "scince": "science",
        "commerce": "commerce",
        "arts": "arts",
        "bn": "been",
        "wont": "won't",
        "cant": "can't",
        "dont": "don't",
        "nt": "not",
        "pls": "please",
        "plz": "please",
        "u": "you",
        "ur": "your",
        "wud": "would",
        "shud": "should",
        "wht": "what",
        "abt": "about",
        "n": "and",
        "nd": "and",
        "b4": "before",
        "b/4": "before",
        "2": "to",
        "2day": "today",
        "4": "for",
        "plmnt": "placement",
        "qual": "qualification",
        "comapny": "company",
        "copany": "company",
        "compny": "company",
    }
    
    # Hinglish mappings (common Hindi-English code-switching)
    HINGLISH_MAP = {
        "kya": "what",  # क्या
        "hai": "is",    # है
        "hain": "are",  # हैं
        "kitna": "how much",  # कितना
        "paisa": "money",  # पैसा
        "paise": "money",  # पैसे
        "rupe": "rupees",  # रुपये
        "rupees": "rupees",
        "kharche": "cost",  # खर्चे
        "padhai": "study",  # पढ़ाई
        "padhnee": "study",
        "exam": "exam",
        "pariksha": "exam",  # परीक्षा
        "takk": "about",  # तक
        "tha": "was",  # था
        "dost": "friend",  # दोस्त
        "accha": "good",  # अच्छा
        "badha": "bad",  # बुरा
        "samjha": "understand",  # समझा
        "dhundho": "find",  # ढूंढो
        "batao": "tell",  # बताओ
        "sunao": "listen",  # सुनाओ
        "jaao": "go",  # जाओ
        "aao": "come",  # आओ
        "mat": "don't",  # मत
        "hoga": "will be",  # होगा
        "hua": "happened",  # हुआ
        "aur": "and",  # और
        "bc": "",  # slang - remove
        "yaar": "friend",  # यार
        "theek": "okay",  # ठीक
        "theek hai": "okay",
        "nahi": "no",  # नहीं
        "haan": "yes",  # हाँ
        "jaldi": "fast",  # जल्दी
        "chalega": "okay",  # चलेगा
        "dekhna": "see",  # देखना
    }
    
    @staticmethod
    def normalize_case(text: str) -> str:
        """Convert to lowercase for uniform processing."""
        return text.lower()
    
    @staticmethod
    def remove_special_chars(text: str) -> str:
        """Remove special characters but keep spaces and common punctuation."""
        # Keep alphanumeric, spaces, and ? !
        text = re.sub(r"[^a-z0-9\s\?!]", "", text)
        # Clean multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    @staticmethod
    def fix_common_typos(text: str) -> str:
        """Fix common typos and abbreviations."""
        words = text.split()
        corrected = []
        
        for word in words:
            # Remove trailing ? or ! for matching
            clean_word = word.rstrip("?!")
            ending = word[len(clean_word):]  # Store punctuation
            
            # Check if it's a known typo
            if clean_word in TextPreprocessor.TYPO_MAP:
                corrected.append(TextPreprocessor.TYPO_MAP[clean_word] + ending)
            else:
                corrected.append(word)
        
        return " ".join(corrected)
    
    @staticmethod
    def normalize_hinglish(text: str) -> str:
        """Normalize Hinglish (Hindi-English code-mixing) to English."""
        words = text.split()
        converted = []
        
        for word in words:
            # Remove trailing punctuation
            clean_word = word.rstrip("?!")
            ending = word[len(clean_word):]
            
            # Check if it's a Hinglish word
            if clean_word in TextPreprocessor.HINGLISH_MAP:
                mapped = TextPreprocessor.HINGLISH_MAP[clean_word]
                if mapped:  # Skip empty strings (like "bc" slang)
                    converted.append(mapped + ending)
            else:
                converted.append(word)
        
        return " ".join(converted)
    
    @staticmethod
    def remove_slang_filler(text: str) -> str:
        """Remove common slang and filler words."""
        slang_words = {"bc", "yaar", "right", "ryt", "lol", "haha", "hmm", "uh", "er"}
        words = text.split()
        filtered = [w for w in words if w.lower() not in slang_words and w]
        return " ".join(filtered)
    
    @staticmethod
    def preprocess(text: str) -> str:
        """
        Apply full preprocessing pipeline.
        
        Args:
            text (str): Raw input text
        
        Returns:
            str: Preprocessed text
        """
        # Step 1: Normalize case
        text = TextPreprocessor.normalize_case(text)
        
        # Step 2: Remove special characters
        text = TextPreprocessor.remove_special_chars(text)
        
        # Step 3: Normalize Hinglish (before fixing typos, as Hinglish words might be typos)
        text = TextPreprocessor.normalize_hinglish(text)
        
        # Step 4: Fix common typos
        text = TextPreprocessor.fix_common_typos(text)
        
        # Step 5: Remove slang/filler
        text = TextPreprocessor.remove_slang_filler(text)
        
        # Step 6: Clean multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    @staticmethod
    def preprocess_batch(texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts (List[str]): List of texts to preprocess
        
        Returns:
            List[str]: List of preprocessed texts
        """
        return [TextPreprocessor.preprocess(text) for text in texts]


# Test the preprocessor
if __name__ == "__main__":
    test_cases = [
        "Kya fees hain?",
        "cn i gt a scolarship",
        "wht bout placements bc",
        "Tell me about the college",
        "FEES???",
        "exams expam exam?",
        "when r exams?",
        "paisa kitna hoga",
        "Difference B.Tech vs B.Sc",
        "hostl facilities?",
    ]
    
    print("TEXT PREPROCESSING TEST")
    print("=" * 60)
    for text in test_cases:
        preprocessed = TextPreprocessor.preprocess(text)
        print(f"Original:      {text}")
        print(f"Preprocessed:  {preprocessed}")
        print()
