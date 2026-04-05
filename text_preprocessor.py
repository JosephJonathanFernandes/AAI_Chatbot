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
        # Fees & Payment related
        "fe": "fee",
        "fes": "fee",
        "fees?": "fee",
        "fee?": "fee",
        "tution": "tuition",
        "tutn": "tuition",
        "tuition fees": "fees",
        
        # Exam related
        "expam": "exam",
        "exams?": "exam",
        "exam?": "exam",
        "exm": "exam",
        "teste": "test",
        "tests?": "test",
        
        # Placement related
        "placment": "placement",
        "placements?": "placement",
        "placement?": "placement",
        "plcement": "placement",
        "plmnt": "placement",
        "plmnt": "placement",
        "place": "placement",
        "placement": "placement",
        
        # Admission related
        "admision": "admission",
        "admissions?": "admission",
        "admission?": "admission",
        "admn": "admission",
        "admisn": "admission",
        
        # Campus & Hostel
        "hostl": "hostel",
        "hostels?": "hostel",
        "hostel?": "hostel",
        "hostels": "hostel",
        "libary": "library",
        "librarys?": "library",
        "library?": "library",
        "campous": "campus",
        "campus?": "campus",
        
        # Scholarship & Financial
        "scholorship": "scholarship",
        "scholarships?": "scholarship",
        "scholarship?": "scholarship",
        "scholar": "scholarship",
        
        # College & Department
        "collage": "college",
        "colleges?": "college",
        "college?": "college",
        "univ": "university",
        "university": "university",
        "departmnt": "department",
        "departments?": "department",
        "department?": "department",
        "dept": "department",
        
        # Streams/Disciplines
        "enginering": "engineering",
        "engr": "engineering",
        "engineering?": "engineering",
        "scince": "science",
        "science?": "science",
        "commerce": "commerce",
        "arts": "arts",
        "arts?": "arts",
        "cse": "cse",
        "ece": "ece",
        "mechanical": "mechanical",
        
        # Faculty
        "faculty?": "faculty",
        "facult": "faculty",
        "prof": "professor",
        "professor?": "professor",
        "teacher?": "teacher",
        
        # Common abbreviations
        "bn": "been",
        "wont": "won't",
        "cant": "can't",
        "dont": "don't",
        "didnt": "did not",
        "isnt": "is not",
        "nt": "not",
        "pls": "please",
        "plz": "please",
        "req": "required",
        "reqd": "required",
        "conf": "confirm",
        
        # Pronouns & Common words
        "u": "you",
        "u?": "you",
        "ur": "your",
        "ur?": "your",
        "urs": "yours",
        "wud": "would",
        "shud": "should",
        "wud": "would",
        "cd": "could",
        "cud": "could",
        
        # Questions & Conversational
        "wht": "what",
        "wht?": "what",
        "wat": "what",
        "y": "why",
        "y?": "why",
        "y": "why",
        "hw": "how",
        "hw?": "how",
        "abt": "about",
        "abt?": "about",
        "bout": "about",
        "n": "and",
        "nd": "and",
        "nd?": "and",
        
        # Time & Dates
        "b4": "before",
        "b/4": "before",
        "aftr": "after",
        "tmrw": "tomorrow",
        "2day": "today",
        "yest": "yesterday",
        "mon": "monday",
        
        # Numbers as words
        "2": "to",
        "2day": "today",
        "4": "for",
        "4u": "for you",
        "8": "ate",
        
        # Company/Job related
        "plmnt": "placement",
        "job?": "job",
        "internship?": "internship",
        "intern?": "intern",
        "company?": "company",
        "compny": "company",
        "comapny": "company",
        "copany": "company",
        "comapny": "company",
        "salary?": "salary",
        "sal": "salary",
        "ctc": "ctc",
        "package?": "package",
        
        # Qualification related
        "qual": "qualification",
        "qualification?": "qualification",
        "qualif": "qualification",
        "9thpass": "9th pass",
        "10thpass": "10th pass",
        "12thpass": "12th pass",
        
        # Quality/Clarity related
        "gud": "good",
        "bad?": "bad",
        "worse": "worse",
        "best?": "best",
        "ok": "okay",
        "ok?": "okay",
        "okk": "okay",
        "perfect?": "perfect",
        
        # Response related
        "thx": "thanks",
        "thnx": "thanks",
        "ty": "thank you",
        "tyvm": "thank you very much",
    }
    
    # Hinglish mappings (common Hindi-English code-switching)
    HINGLISH_MAP = {
        # Questions
        "kya": "what",  # क्या
        "kya?": "what",
        "kyaa": "what",
        "kaun": "who",  # कौन
        "kaun?": "who",
        "kahan": "where",  # कहाँ
        "kahan?": "where",
        "kaise": "how",  # कैसे
        "kaise?": "how",
        "kaunsa": "which",  # कौनसा
        "kaunsa?": "which",
        "konsa": "which",
        
        # Verbs & State
        "hai": "is",    # है
        "hain": "are",  # हैं
        "tha": "was",  # था
        "the": "were",  # थे
        "ho": "be",  # हो
        "honge": "will be",  # होंगे
        "hua": "happened",  # हुआ
        "hoge": "will be",  # होगे
        "hoga": "will be",  # होगा
        "hogaa": "will be",
        "hogey": "will be",
        
        # Quantities
        "kitna": "how much",  # कितना
        "kitne": "how many",  # कितने
        "kitni": "how much",  # कितनी
        "adhik": "more",  # अधिक
        "kam": "less",  # कम
        "sab": "all",  # सब
        "kuch": "some",  # कुछ
        
        # Money related
        "paisa": "money",  # पैसा
        "paise": "money",  # पैसे
        "rupee": "rupee",  # रुपया
        "rupees": "rupees",
        "rupe": "rupees",
        "kharche": "cost",  # खर्चे
        "kharc": "cost",
        "kharch": "cost",
        
        # Study related
        "padhai": "study",  # पढ़ाई
        "padhnee": "study",
        "pdhae": "study",
        "padhungi": "will study",
        "exam": "exam",
        "pariksha": "exam",  # परीक्षा
        "pareeksha": "exam",
        "puraiksha": "exam",
        
        # Time
        "takk": "about",  # तक
        "tak": "till",  # तक
        "jab": "when",  # जब
        "tab": "then",  # तब
        "abhi": "now",  # अभी
        "kal": "tomorrow",  # कल
        "aaj": "today",  # आज
        "kal": "yesterday",  # कल
        "subah": "morning",  # सुबह
        "shaam": "evening",  # शाम
        "raat": "night",  # रात
        
        # Feelings
        "accha": "good",  # अच्छा
        "achha": "good",
        "acha": "good",
        "badha": "bad",  # बुरा
        "bura": "bad",
        "bhadey": "bad",
        
        # Action verbs
        "dhundho": "find",  # ढूंढो
        "dhundo": "find",
        "batao": "tell",  # बताओ
        "bataye": "tell",
        "sunao": "listen",  # सुनाओ
        "sunao": "listen",
        "jaao": "go",  # जाओ
        "jao": "go",
        "aao": "come",  # आओ
        "aa": "come",
        "chalo": "come",  # चलो
        "chalo": "let's go",
        "samjha": "understand",  # समझा
        "samajh": "understand",
        "samajhta": "understand",
        "dekho": "see",  # देखो
        "dekha": "saw",  # देखा
        
        # Negation & Modals
        "mat": "don't",  # मत
        "nahi": "no",  # नहीं
        "nhi": "no",
        "naa": "no",
        "na": "no",
        "bilkul": "absolutely",  # बिलकुल
        
        # Affirmation
        "haan": "yes",  # हाँ
        "ha": "yes",
        "theek": "okay",  # ठीक
        "theek hai": "okay",
        "thik": "okay",
        "chalega": "okay",  # चलेगा
        "chalega": "fine let's go",
        
        # Conjunction
        "aur": "and",  # और
        "aur?": "and",
        "par": "but",  # पर
        "par?": "but",
        "yaa": "or",  # या
        "ya": "or",
        
        # Relationships
        "dost": "friend",  # दोस्त
        "bhai": "brother",  # भाई
        "bhen": "sister",  # बहन
        "ma": "mother",  # माँ
        "baap": "father",  # बाप
        
        # General slang
        "yaar": "friend",  # यार
        "jaldi": "fast",  # जल्दी
        "jaldhi": "fast",
        "slow": "slow",  # स्लो
        "thoda": "a bit",  # थोड़ा
        "bahut": "very",  # बहुत
        "dekhna": "see",  # देखना
        "sun": "listen",  # सुन
        "sunna": "listen",
        "jaanna": "know",  # जानना
        
        # Remove slang  
        "bc": "",  # slang - remove
        "arre": "",  # slang - remove
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
