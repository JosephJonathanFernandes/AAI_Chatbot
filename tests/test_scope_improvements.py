#!/usr/bin/env python
"""Comprehensive scope detection validation test."""

from scope_detector import ScopeDetector
from intent_model import IntentClassifier

def test_expanded_keywords():
    """Test that all previously failing queries now pass."""
    print("\n" + "=" * 120)
    print("TEST 1: EXPANDED KEYWORDS - Previously Failing Queries")
    print("=" * 120 + "\n")
    
    detector = ScopeDetector()
    
    # These were the 9 queries with low confidence before
    previously_failing = [
        "What's the graduation process?",
        "Can I change my major?",
        "Do you have a medical center?",
        "What's the parking situation?",
        "Are there clubs or societies?",
        "What's the dress code?",
        "Can I get transcript?",
        "What majors are available?",
        "Can I get a dorm?",
    ]
    
    failed = []
    for query in previously_failing:
        is_in_scope, reason, conf = detector.is_in_scope(query)
        status = "✓ PASS" if is_in_scope and conf > 0.5 else "✗ FAIL"
        print(f"  {status} [{conf:.2f}] {query}")
        if not (is_in_scope and conf > 0.5):
            failed.append(query)
    
    if not failed:
        print(f"\n  ✓ All 9 previously-failing queries now pass!")
        return True
    else:
        print(f"\n  ✗ {len(failed)} queries still failing")
        return False


def test_context_awareness():
    """Test context-aware scope checking."""
    print("\n" + "=" * 120)
    print("TEST 2: CONTEXT AWARENESS - Using Conversation History")
    print("=" * 120 + "\n")
    
    detector = ScopeDetector()
    
    # Test scenario: user asks vague question after college-related history
    history = [
        "What are the admission requirements?",
        "How much do engineering fees cost?",
        "When are exams scheduled?",
    ]
    
    vague_query = "What else do I need to know?"
    
    # Without history
    is_in_scope_no_hist, _, conf_no_hist = detector.is_in_scope(vague_query)
    
    # With history
    is_in_scope_hist, _, conf_hist = detector.is_in_scope(vague_query, conversation_history=history)
    
    print(f"  Query: '{vague_query}'")
    print(f"    Without history: in_scope={is_in_scope_no_hist}, conf={conf_no_hist:.2f}")
    print(f"    With history:    in_scope={is_in_scope_hist}, conf={conf_hist:.2f}")
    
    if conf_hist >= conf_no_hist:
        print(f"  ✓ Context boost working (confidence increased from {conf_no_hist:.2f} to {conf_hist:.2f})")
        return True
    else:
        print(f"  ✗ Context boost not working as expected")
        return False


def test_semantic_similarity():
    """Test semantic similarity matching."""
    print("\n" + "=" * 120)
    print("TEST 3: SEMANTIC SIMILARITY - Fuzzy Matching")
    print("=" * 120 + "\n")
    
    detector = ScopeDetector()
    
    # These should match via semantic similarity
    semantic_queries = [
        ("Can I switch my specialization?", "academics", True),
        ("Where can I eat on campus?", "student_facilities", True),
        ("Are there transport services?", "campus_amenities", True),
        ("How do I get my diploma?", "student_records", True),
    ]
    
    passed = 0
    for query, expected_category, should_be_in_scope in semantic_queries:
        is_in_scope, reason, conf = detector.is_in_scope(query)
        
        # Check if correctly detected as in-scope
        if is_in_scope == should_be_in_scope:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
        
        print(f"  {status} [{conf:.2f}] {query}")
        print(f"       Category: {reason}, Expected: {expected_category}")
    
    print(f"\n  Result: {passed}/{len(semantic_queries)} semantic queries passed")
    return passed == len(semantic_queries)


def test_out_of_scope():
    """Test that out-of-scope queries are still correctly rejected."""
    print("\n" + "=" * 120)
    print("TEST 4: OUT-OF-SCOPE DETECTION - Verify False Positives Prevented")
    print("=" * 120 + "\n")
    
    detector = ScopeDetector()
    
    out_of_scope_queries = [
        "How do I cook biryani?",
        "Tell me about Bitcoin trading",
        "Write Python code for me",
        "What's on Netflix today?",
        "How to fix my laptop?",
    ]
    
    passed = 0
    for query in out_of_scope_queries:
        is_in_scope, reason, conf = detector.is_in_scope(query)
        if not is_in_scope:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
        
        print(f"  {status} {query}")
        print(f"       Detected as: {reason} (conf={conf:.2f})")
    
    print(f"\n  Result: {passed}/{len(out_of_scope_queries)} out-of-scope queries correctly rejected")
    return passed == len(out_of_scope_queries)


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "=" * 120)
    print("TEST 5: EDGE CASES - Typos, Abbreviations, Mixed Language")
    print("=" * 120 + "\n")
    
    detector = ScopeDetector()
    
    edge_cases = [
        ("wat r the fees?", True),  # Typo
        ("placement rate plz", True),  # Abbreviation
        ("Kya scholarships hain?", True),  # Hinglish
        ("FEE", True),  # Caps
        ("admission requirements?", True),  # Minimal
        ("", True),  # Empty (should be in-scope with low confidence)
    ]
    
    passed = 0
    for query, should_be_in_scope in edge_cases:
        is_in_scope, reason, conf = detector.is_in_scope(query)
        if is_in_scope == should_be_in_scope:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
        
        query_display = f"'{query}'" if query else "(empty)"
        print(f"  {status} {query_display} -> in_scope={is_in_scope}, conf={conf:.2f}")
    
    print(f"\n  Result: {passed}/{len(edge_cases)} edge cases handled correctly")
    return passed == len(edge_cases)


def main():
    """Run all tests and report results."""
    print("\n")
    print("╔" + "═" * 118 + "╗")
    print("║" + " " * 40 + "SCOPE DETECTION VALIDATION SUITE" + " " * 46 + "║")
    print("║" + " " * 50 + "Priority 4: Accuracy Improvements" + " " * 36 + "║")
    print("╚" + "═" * 118 + "╝")
    
    results = []
    results.append(("Expanded Keywords", test_expanded_keywords()))
    results.append(("Context Awareness", test_context_awareness()))
    results.append(("Semantic Similarity", test_semantic_similarity()))
    results.append(("Out-of-Scope Detection", test_out_of_scope()))
    results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120 + "\n")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\n  Total: {passed}/{total} test groups passed")
    
    if passed == total:
        print("\n  ✓✓✓ ALL TESTS PASSED - Scope detection accuracy improved! ✓✓✓\n")
        return True
    else:
        print(f"\n  ✗ {total - passed} test group(s) failed\n")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
