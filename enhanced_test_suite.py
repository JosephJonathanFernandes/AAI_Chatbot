"""
Enhanced Chatbot Test Suite with Metrics
Runs comprehensive test cases against the chatbot and measures performance metrics.
Outputs results to TEST_RESULTS_*.txt file.
"""

import time
import os
from typing import Dict, List, Optional
from datetime import datetime
from statistics import mean, stdev, median
import sys

# Import chatbot modules
try:
    from intent_model import IntentClassifier
    from emotion_detector import EmotionDetector
    from llm_handler import LLMHandler
    from context_manager import ConversationContext
    from database import ChatbotDatabase
    from time_context import TimeContext
    from scope_detector import ScopeDetector
    from prompt_engineering import PromptEngineer
    from session_greeter import SessionGreeter
    from error_recovery import ErrorRecovery
    from emotional_tone_detector import EmotionalToneDetector
    from intent_refiner import IntentRefiner
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Test cases with edge cases and recent optimization tests
TEST_CASES = [
    # FEES - Original + Edge Cases
    {"id": "FEES-001", "query": "What are the admission fees?", "category": "fees", "type": "direct"},
    {"id": "FEES-004", "query": "How much does it cost?", "category": "fees", "type": "vague"},
    {"id": "FEES-012", "query": "Wht r the fess?", "category": "fees", "type": "typo"},
    {"id": "FEES-013", "query": "How much money do I gotta pay?", "category": "fees", "type": "casual"},
    {"id": "FEES-014", "query": "This is way too expensive! Why are fees so high?!", "category": "fees", "type": "emotion_angry"},
    {"id": "FEES-EDGE-1", "query": "fees?", "category": "fees", "type": "ultra_short"},
    {"id": "FEES-EDGE-2", "query": "FEES FEES FEES FEES", "category": "fees", "type": "repetition"},
    
    # EXAMS - Original + Edge Cases
    {"id": "EXAM-001", "query": "When are the exams scheduled?", "category": "exams", "type": "direct"},
    {"id": "EXAM-002", "query": "When is the math exam?", "category": "exams", "type": "specific"},
    {"id": "EXAM-008", "query": "I didn't understand the exam schedule", "category": "exams", "type": "clarification"},
    {"id": "EXAM-EDGE-1", "query": "exam exam exam exam", "category": "exams", "type": "repetition"},
    {"id": "EXAM-EDGE-2", "query": "when when when", "category": "exams", "type": "ultra_short"},
    
    # PLACEMENTS - Original + Edge Cases
    {"id": "PLACE-001", "query": "Tell me about placements", "category": "placements", "type": "direct"},
    {"id": "PLACE-002", "query": "What's the average salary for placements?", "category": "placements", "type": "specific"},
    {"id": "PLACE-010", "query": "I'm worried the package might be too low for me", "category": "placements", "type": "emotion_worried"},
    {"id": "PLACE-EDGE-1", "query": "placements?", "category": "placements", "type": "ultra_short"},
    
    # HOSTEL - Original + Edge Cases
    {"id": "HOSTEL-001", "query": "Is hostel accommodation available?", "category": "hostel", "type": "direct"},
    {"id": "HOSTEL-004", "query": "What facilities are provided in the hostel?", "category": "hostel", "type": "facilities"},
    {"id": "HOSTEL-EDGE-1", "query": "hostel hostel hostel", "category": "hostel", "type": "repetition"},
    
    # OUT-OF-SCOPE - Critical Edge Cases
    {"id": "OOS-001", "query": "What's the weather like?", "category": "out_of_scope", "type": "weather"},
    {"id": "OOS-002", "query": "Tell me a joke", "category": "out_of_scope", "type": "entertainment"},
    {"id": "OOS-003", "query": "What's your favorite color?", "category": "out_of_scope", "type": "personal"},
    {"id": "OOS-004", "query": "Can you help me with calculus?", "category": "out_of_scope", "type": "general_help"},
    
    # MULTI-TURN - Context preservation
    {"id": "CONTEXT-001", "query": "What are placements?", "category": "placements", "type": "context_first"},
    {"id": "CONTEXT-002", "query": "And what about fees?", "category": "fees", "type": "context_second"},
    
    # TYPOS & VARIATIONS
    {"id": "TYPO-001", "query": "wot r the fees", "category": "fees", "type": "typo_lowercase"},
    {"id": "TYPO-002", "query": "FEES", "category": "fees", "type": "uppercase"},
    {"id": "TYPO-003", "query": "tution fees", "category": "fees", "type": "typo_misspell"},
    
    # EMPTY/NULL - Boundary cases
    {"id": "BOUNDARY-001", "query": "   ", "category": "boundary", "type": "whitespace"},
    {"id": "BOUNDARY-002", "query": "...", "category": "boundary", "type": "punctuation"},
    
    # HYBRID QUERIES
    {"id": "HYBRID-001", "query": "What are the fees and when are exams?", "category": "hybrid", "type": "multi_intent"},
    {"id": "HYBRID-002", "query": "Tell me about placements and hostel facilities", "category": "hybrid", "type": "multi_intent"},
    
    # RECENT OPTIMIZATION TESTS
    {"id": "OPTIM-001", "query": "admission process engineering", "category": "admission", "type": "keywords_only"},
    {"id": "OPTIM-002", "query": "placement average package salary", "category": "placements", "type": "keywords_only"},
    {"id": "OPTIM-003", "query": "Can I know about fees?", "category": "fees", "type": "polite"},
]

DEFAULT_EXPANDED_SOURCE_FILE = "CHATBOT_QUESTION_TEST_CASES.txt"
DEFAULT_MAX_TEST_CASES = 120


def _safe_strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def _infer_category_from_expected_intent(expected_intent: str) -> str:
    intent = (expected_intent or "").strip().lower()
    if not intent:
        return "unknown"
    if "out" in intent and "scope" in intent:
        return "out_of_scope"
    if "library" in intent:
        return "library"
    if "faculty" in intent or "professor" in intent or "teacher" in intent:
        return "faculty"
    if "timetable" in intent or "schedule" in intent:
        return "timetable"
    if "campus" in intent or "clubs" in intent or "facility" in intent:
        return "campus_life"
    if "compar" in intent:
        return "comparison"
    if "general" in intent or "info" in intent:
        return "general_info"
    if "fees" in intent or "tuition" in intent or "payment" in intent or "refund" in intent:
        return "fees"
    if "exam" in intent or "result" in intent:
        return "exams"
    if "placement" in intent or "intern" in intent:
        return "placements"
    if "hostel" in intent:
        return "hostel"
    if "admission" in intent or "eligib" in intent:
        return "admission"
    if "greet" in intent:
        return "greetings"
    if "thanks" in intent or "grat" in intent:
        return "gratitude"
    return "unknown"


def load_expanded_test_cases(
    source_file: str = DEFAULT_EXPANDED_SOURCE_FILE,
    max_cases: int = DEFAULT_MAX_TEST_CASES,
) -> Optional[List[Dict]]:
    """Load test cases from the human-authored test-case document.

    The document is expected to contain repeated blocks with:
      - 'TEST ID: ...'
      - 'Test Query: "..."'
      - 'Expected Intent: ...'

    Returns None if the source file is missing or cannot be parsed.
    """
    if not os.path.exists(source_file):
        return None

    try:
        with open(source_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"[WARN] Could not read expanded test source '{source_file}': {e}")
        return None

    lines = text.splitlines()
    cases: List[Dict] = []
    current_id: Optional[str] = None
    current_query: Optional[str] = None
    current_expected_intent: Optional[str] = None
    current_question_type: Optional[str] = None

    def flush_current():
        nonlocal current_id, current_query, current_expected_intent, current_question_type
        if not current_id or not current_query:
            current_id = None
            current_query = None
            current_expected_intent = None
            current_question_type = None
            return

        expected_intent = current_expected_intent or ""
        category = _infer_category_from_expected_intent(expected_intent)
        case_type = (current_question_type or "doc_case").strip().lower().replace(" ", "_")
        cases.append(
            {
                "id": current_id.strip(),
                "query": current_query,
                "category": category,
                "type": case_type,
                "expected_intent_raw": expected_intent.strip(),
            }
        )

        current_id = None
        current_query = None
        current_expected_intent = None
        current_question_type = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.upper().startswith("TEST ID:"):
            flush_current()
            current_id = line.split(":", 1)[1].strip()
            continue

        if line.lower().startswith("question type:"):
            current_question_type = line.split(":", 1)[1].strip()
            continue

        if line.lower().startswith("test query:"):
            value = line.split(":", 1)[1].strip()
            current_query = _safe_strip_quotes(value)
            continue

        if line.lower().startswith("expected intent:"):
            current_expected_intent = line.split(":", 1)[1].strip()
            continue

    flush_current()

    if not cases:
        return None

    # De-duplicate by ID then by query.
    by_id = {}
    for c in cases:
        by_id[c["id"]] = c
    unique = list(by_id.values())

    seen_queries = set()
    deduped = []
    for c in unique:
        q = (c.get("query") or "").strip().lower()
        if not q or q in seen_queries:
            continue
        seen_queries.add(q)
        deduped.append(c)

    return deduped[: max(1, int(max_cases))]

class ChatbotTestSuite:
    def __init__(self):
        """Initialize test suite and chatbot components."""
        print("[Initializing Chatbot Test Suite...]")
        self.intent_classifier = IntentClassifier()
        self.emotion_detector = EmotionDetector()
        self.llm_handler = LLMHandler()
        self.scope_detector = ScopeDetector()
        self.prompt_engineer = PromptEngineer()
        self.database = ChatbotDatabase()
        
        # Metrics
        self.results = []
        self.latencies = []
        self.successful_count = 0
        self.failed_count = 0
        self.out_of_scope_count = 0

        # Controls
        self.strict_acceptance = os.getenv("STRICT_ACCEPTANCE", "1") not in {"0", "false", "False"}

    def _acceptance_pass(self, result: Dict) -> bool:
        """Determine PASS/FAIL for scenario tests.

        The original suite treated any non-error LLM response as PASS; for expanded scenario
        runs, strict acceptance is enabled by default to incorporate expected category.
        """
        if result.get("error"):
            return False

        category = (result.get("category") or "unknown").strip().lower()

        query = (result.get("query") or "").strip()
        if not query or query == "...":
            return False

        if not self.strict_acceptance:
            return bool(result.get("response"))

        predicted_intent = (result.get("intent") or "unknown").strip().lower()
        is_in_scope = bool(result.get("is_in_scope", True))
        has_response = bool(result.get("response"))

        if category == "out_of_scope":
            oos_detected = (not is_in_scope) or (predicted_intent == "out_of_scope")
            return oos_detected and has_response

        # For categories that are not directly mapped to an intent label, fall back to
        # response-based acceptance to keep the expanded run usable/reproducible.
        if category in {"unknown", "hybrid"}:
            return is_in_scope and has_response

        # In-scope categories: require in-scope + correct intent label.
        return is_in_scope and (predicted_intent == category) and has_response
        
    def run_test(self, test_case: Dict) -> Dict:
        """Run single test case and measure metrics."""
        query = test_case["query"]
        test_id = test_case["id"]
        
        start_time = time.perf_counter()
        result = {
            "test_id": test_id,
            "query": query,
            "category": test_case["category"],
            "type": test_case["type"],
            "success": False,
            "error": None,
            "latency_ms": 0,
            "intent": None,
            "confidence": 0.0,
            "emotion": "neutral",
            "is_in_scope": True,
            "response": None
        }
        
        try:
            # PHASE 1: Intent Classification
            if not query or query.strip() == "" or query.strip() == "...":
                result["error"] = "Empty or invalid query"
                result["latency_ms"] = (time.perf_counter() - start_time) * 1000
                result["success"] = False
                return result
            
            intent_result = self.intent_classifier.predict(query)
            if isinstance(intent_result, dict):
                result["intent"] = intent_result.get("intent", "unknown")
                result["confidence"] = intent_result.get("confidence", 0.0)
            
            # PHASE 2: Emotion Detection
            try:
                emotion_result = self.emotion_detector.detect(query)
                if isinstance(emotion_result, dict):
                    result["emotion"] = emotion_result.get("emotion", "neutral")
            except:
                result["emotion"] = "neutral"
            
            # PHASE 3: Scope Detection
            try:
                scope_result = self.scope_detector.get_scope_info(query, result["intent"], result["confidence"])
                if isinstance(scope_result, dict):
                    result["is_in_scope"] = scope_result.get("is_in_scope", True)
            except:
                result["is_in_scope"] = True
            
            # PHASE 4: LLM Response
            try:
                llm_result = self.llm_handler.generate_response(
                    user_input=query,
                    intent=result["intent"],
                    confidence=result["confidence"],
                    emotion=result["emotion"]
                )
                
                if isinstance(llm_result, dict) and not llm_result.get("error"):
                    result["response"] = llm_result.get("response", "")
                    result["success"] = True
                else:
                    result["error"] = llm_result.get("error", "LLM failed") if isinstance(llm_result, dict) else str(llm_result)
            except Exception as llm_error:
                result["error"] = f"LLM Error: {str(llm_error)[:50]}"
            
        except Exception as e:
            result["error"] = f"Test Error: {str(e)[:50]}"
            result["success"] = False
        
        # METRICS
        result["latency_ms"] = (time.perf_counter() - start_time) * 1000
        # Override success with acceptance criteria when running scenario evaluation.
        result["success"] = self._acceptance_pass(result)
        return result
    
    def run_all_tests(self) -> List[Dict]:
        """Run all test cases."""
        print(f"\n[Running {len(TEST_CASES)} Test Cases]")
        print("-" * 80)
        
        for idx, test_case in enumerate(TEST_CASES, 1):
            result = self.run_test(test_case)
            self.results.append(result)
            self.latencies.append(result["latency_ms"])
            
            # Update counters
            if result["success"]:
                self.successful_count += 1
            else:
                self.failed_count += 1
            
            if not result["is_in_scope"]:
                self.out_of_scope_count += 1
            
            # Progress indicator
            status = "[PASS]" if result["success"] else "[FAIL]"
            print(f"{idx:3}. {status} {result['test_id']:15} | {result['query'][:40]:40} | {result['latency_ms']:6.0f}ms")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 100)
        report.append("CHATBOT TEST EXECUTION REPORT")
        report.append("=" * 100)
        report.append(f"\nExecution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {len(self.results)}")
        report.append(f"Successful: {self.successful_count} ({self.successful_count/len(self.results)*100:.1f}%)")
        report.append(f"Failed: {self.failed_count} ({self.failed_count/len(self.results)*100:.1f}%)")
        report.append(f"Out-of-Scope Detected: {self.out_of_scope_count}")
        
        # LATENCY ANALYSIS
        if self.latencies:
            report.append("\n[LATENCY METRICS (milliseconds)]")
            report.append(f"  Min:    {min(self.latencies):8.1f}ms")
            report.append(f"  Max:    {max(self.latencies):8.1f}ms")
            report.append(f"  Avg:    {mean(self.latencies):8.1f}ms")
            report.append(f"  Median: {median(self.latencies):8.1f}ms")
            if len(self.latencies) > 1:
                report.append(f"  Stdev:  {stdev(self.latencies):8.1f}ms")
        
        # CATEGORY BREAKDOWN
        report.append("\n[TEST RESULTS BY CATEGORY]")
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"pass": 0, "fail": 0, "latencies": []}
            if result["success"]:
                categories[cat]["pass"] += 1
            else:
                categories[cat]["fail"] += 1
            categories[cat]["latencies"].append(result["latency_ms"])
        
        for cat in sorted(categories.keys()):
            stats = categories[cat]
            total = stats["pass"] + stats["fail"]
            pass_rate = stats["pass"] / total * 100 if total > 0 else 0
            avg_latency = mean(stats["latencies"]) if stats["latencies"] else 0
            report.append(f"  {cat:15} | Pass: {stats['pass']:2}/{total:2} ({pass_rate:5.1f}%) | Avg: {avg_latency:6.0f}ms")
        
        # DETAILED RESULTS
        report.append("\n[DETAILED TEST RESULTS]")
        report.append("-" * 100)
        
        for result in self.results:
            report.append(f"\nTest ID: {result['test_id']}")
            report.append(f"  Query:      {result['query'][:70]}")
            report.append(f"  Type:       {result['type']}")
            report.append(f"  Status:     {'PASS' if result['success'] else 'FAIL'}")
            report.append(f"  Latency:    {result['latency_ms']:.1f}ms")
            report.append(f"  Intent:     {result['intent']} ({result['confidence']:.1%})")
            report.append(f"  Emotion:    {result['emotion']}")
            report.append(f"  In-Scope:   {result['is_in_scope']}")
            
            if result["error"]:
                report.append(f"  Error:      {result['error']}")
            
            if result["response"]:
                response_preview = result["response"][:100] + "..." if len(result["response"]) > 100 else result["response"]
                report.append(f"  Response:   {response_preview}")
        
        # EDGE CASE ANALYSIS
        report.append("\n[EDGE CASE ANALYSIS]")
        report.append("-" * 100)
        
        edge_cases = {
            "typo": [r for r in self.results if r["type"].startswith("typo")],
            "out_of_scope": [r for r in self.results if r["category"] == "out_of_scope"],
            "emotion": [r for r in self.results if "emotion_" in r["type"]],
            "boundary": [r for r in self.results if r["category"] == "boundary"],
            "ultra_short": [r for r in self.results if r["type"] == "ultra_short"],
        }
        
        for case_type, cases in edge_cases.items():
            if cases:
                pass_count = sum(1 for c in cases if c["success"])
                report.append(f"\n{case_type.upper()}: {pass_count}/{len(cases)} passed")
                for case in cases:
                    status = "[PASS]" if case["success"] else "[FAIL]"
                    report.append(f"  {status} {case['test_id']:15} | {case['query'][:50]:50}")
        
        # SUMMARY STATS
        report.append("\n[SUMMARY STATISTICS]")
        report.append("-" * 100)
        report.append(f"Average Response Latency: {mean(self.latencies):.1f}ms")
        report.append(f"P95 Latency: {sorted(self.latencies)[int(len(self.latencies)*0.95)]:.1f}ms" if len(self.latencies) > 20 else "P95 Latency: N/A (< 20 samples)")
        report.append(f"Success Rate: {self.successful_count}/{len(self.results)} ({self.successful_count/len(self.results)*100:.1f}%)")
        report.append(f"Out-of-Scope Detection Rate: {self.out_of_scope_count/len(self.results)*100:.1f}%")
        
        report.append("\n" + "=" * 100)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = None) -> str:
        """Save report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"TEST_RESULTS_{timestamp}.txt"
        
        report_content = self.generate_report()
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"\n[OK] Report saved: {filename}")
            return filename
        except Exception as e:
            print(f"[ERROR] Failed to save report: {e}")
            return None

def main():
    """Main test execution."""
    print("\n" + "=" * 100)
    print("ENHANCED CHATBOT TEST SUITE WITH METRICS")
    print("=" * 100)
    
    suite = ChatbotTestSuite()

    # Expand test cases (100--150) from the human-authored document when available.
    expanded_source = os.getenv("EXPANDED_TEST_SOURCE", DEFAULT_EXPANDED_SOURCE_FILE)
    max_cases = int(os.getenv("MAX_TEST_CASES", str(DEFAULT_MAX_TEST_CASES)))
    expanded_cases = load_expanded_test_cases(expanded_source, max_cases=max_cases)
    if expanded_cases:
        global TEST_CASES
        TEST_CASES = expanded_cases
        print(f"[OK] Loaded expanded test cases: {len(TEST_CASES)} from {expanded_source}")
        if suite.strict_acceptance:
            print("[OK] Strict acceptance enabled: PASS requires intent/scope match")
        else:
            print("[WARN] Strict acceptance disabled: PASS requires only LLM response")
    else:
        print(f"[WARN] Using built-in TEST_CASES ({len(TEST_CASES)}) - expanded source not available")
    
    # Run tests
    print("\nInitializing chatbot components...")
    if suite.intent_classifier.is_trained:
        print("[OK] Intent classifier loaded")
    else:
        print("[TRAINING] Intent classifier training...")
        suite.intent_classifier.train()
    
    # Execute all tests
    results = suite.run_all_tests()
    
    # Generate and save report
    report = suite.generate_report()
    
    # Force UTF-8 encoding for output
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("\n" + report)
    
    # Save to file
    output_file = suite.save_report()
    
    print(f"\n[OK] Test suite complete: {suite.successful_count}/{len(results)} passed")

if __name__ == "__main__":
    main()
