#!/usr/bin/env python
"""Diagnose scope detection issues."""

from scope_detector import ScopeDetector
from intent_model import IntentClassifier

detector = ScopeDetector()
intent_c = IntentClassifier()

test_queries = [
    "What are the engineering fees?",
    "When are the exams scheduled?",
    "Tell me about placements",
    "What's the hostel fee?",
    "How much does it cost?",
    "Can I use the library late?",
    "What's the campus location?",
    "Tell me about the campus culture",
    "What are the admissions requirements?",
    "Are there any sports facilities?",
    "What's the graduation process?",
    "How do I apply for scholarships?",
    "What are the course prerequisites?",
    "Can I change my major?",
    "What's the class schedule?",
    "Do you have a medical center?",
    "What's the parking situation?",
    "Are there clubs or societies?",
    "What's the dress code?",
    "Can I get transcript?",
    "What about student clubs?",
    "Tell me about scholarship",
    "Placement statistics please",
    "Faculty contact info?",
    "Library opening hours?",
    "Hostel rules and regulations?",
    "How to apply?",
    "Requirements for admission?",
    "What majors are available?",
    "Can I get a dorm?",
]

print("=" * 120)
print("SCOPE DETECTION DIAGNOSTIC")
print("=" * 120)
print()
print(f"{'Query':<45} | {'Scope':<5} | {'Reason':<30} | {'Conf':<6} | {'Intent':<20}")
print("-" * 120)

low_conf = []
for query in test_queries:
    is_in_scope, reason, conf = detector.is_in_scope(query)
    intent, intent_conf = intent_c.predict(query)
    if conf < 0.5:
        low_conf.append((query, conf, reason, intent))
    status = "IN" if is_in_scope else "OUT"
    print(f"{query:<45} | {status:<5} | {reason:<30} | {conf:<6.2f} | {intent:<20}")

print()
print("=" * 120)
print(f"SUMMARY: {len(low_conf)} queries with LOW confidence (<0.5)")
print("=" * 120)
for query, conf, reason, intent in low_conf:
    print(f"  [{conf:.2f}] {query}")
    print(f"        → Reason: {reason} | Intent: {intent}")
