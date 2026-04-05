"""
Performance profiling and latency analysis for chatbot.
Identifies bottlenecks and measures component latencies.
"""
import time
import json
from intent_model import IntentClassifier
from emotion_detector import EmotionDetector
from llm_handler import LLMHandler
from database import ChatbotDatabase

def profile_component(name, func, *args, **kwargs):
    """Measure component execution time."""
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        print(f"[OK] {name:30} | {elapsed:7.2f}ms | Success")
        return elapsed, result
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[FAIL] {name:30} | {elapsed:7.2f}ms | Error: {str(e)[:40]}")
        return elapsed, None

print("\n" + "="*80)
print("LATENCY PROFILING: Component Breakdown")
print("="*80 + "\n")

# 1. Intent Classifier
print("1. INTENT CLASSIFIER")
print("-" * 80)
classifier = IntentClassifier()

# Training (one-time)
elapsed, _ = profile_component(
    "Ensemble Training (first run)",
    classifier.train
)
training_time = elapsed

# Semantic prediction
test_query = "What are the engineering fees?"
elapsed, pred = profile_component(
    "Ensemble Prediction",
    classifier.predict,
    test_query
)
prediction_time = elapsed

# 2. Emotion Detector
print("\n2. EMOTION DETECTOR")
print("-" * 80)
detector = EmotionDetector()

elapsed, emotion = profile_component(
    "Emotion Detection",
    detector.detect_emotion,
    test_query
)
emotion_time = elapsed

# 3. LLM Handler (Groq API)
print("\n3. LLM HANDLER (Groq API)")
print("-" * 80)
llm = LLMHandler()

# System prompt construction
start = time.perf_counter()
system_prompt = llm.prompt_engineer.build_system_prompt(
    intent=pred[0],
    confidence=pred[1],
    emotion=emotion
)
prompt_time = (time.perf_counter() - start) * 1000

profile_component(
    "System Prompt Construction",
    lambda: system_prompt  # Already done above
)

# LLM Response Generation (actual API call)
print("\n  Attempting Groq API call (first actual network request)...")
user_prompt = llm.prompt_engineer.build_user_prompt(
    user_input=test_query,
    intent=pred[0],
    confidence=pred[1]
)

try:
    start = time.perf_counter()
    response_result = llm.generate_response(
        user_input=test_query,
        intent=pred[0],
        confidence=pred[1],
        emotion=emotion
    )
    llm_time = (time.perf_counter() - start) * 1000
    print(f"✓ {'LLM Response Generation':30} | {llm_time:7.2f}ms | Success")
except Exception as e:
    llm_time = 5000  # timeout estimate
    print(f"⚠ {'LLM Response Generation':30} | ~5000ms+ | {str(e)[:40]}")

# 4. Database Logging
print("\n4. DATABASE LOGGING")
print("-" * 80)
db = ChatbotDatabase()

elapsed, _ = profile_component(
    "Database Log Write",
    db.log_interaction,
    session_id="test_session",
    user_input=test_query,
    intent=pred[0],
    confidence=pred[1],
    emotion=emotion,
    response="Test response",
    llm_source="groq",
    response_time=llm_time
)
db_time = elapsed

# 5. Summary
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)

metrics = {
    "Intent Classification": prediction_time,
    "Emotion Detection": emotion_time,
    "Prompt Construction": prompt_time,
    "LLM Generation": llm_time,
    "Database Logging": db_time,
}

total = sum(metrics.values())
print(f"\n{'Component':<30} | {'Time (ms)':>10} | {'% of Total':>10}")
print("-" * 55)
for component, time_ms in metrics.items():
    percentage = (time_ms / total) * 100
    print(f"{component:<30} | {time_ms:>10.2f} | {percentage:>9.1f}%")

print("-" * 55)
print(f"{'TOTAL LATENCY':<30} | {total:>10.2f}ms | {'100.0%':>10}")

print(f"\n📊 Bottleneck: {max(metrics, key=metrics.get)} ({max(metrics.values()):.0f}ms)")
print(f"🎯 Target: <500ms (excluding first-time LLM calls)")
print("="*80 + "\n")
