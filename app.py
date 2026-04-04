"""
Streamlit UI for the College AI Assistant - Production-Ready Edition.
Enhanced with scope detection, prompt engineering, emotion-aware responses,
and knowledge grounding (RAG-lite).

Features:
- Integrated scope detection for domain filtering
- Smart clarification for low confidence
- Emotion-aware response toning
- Knowledge grounding with relevant college data
- Enhanced logging with scope & clarification tracking
- Beautiful chat-like interface similar to ChatGPT
"""

import streamlit as st
import time
from datetime import datetime
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
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
from utils import (
    get_time_greeting, get_time_of_day, calculate_confidence_percentage,
    is_college_domain_query, truncate_text, get_current_timestamp
)


# Page configuration
st.set_page_config(
    page_title="College AI Assistant - Production Ready",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0px); }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 1.25rem;
        margin-left: 3rem;
        border: none;
    }
    .assistant-message {
        background-color: #f0f2f6;
        border-left: 4px solid #4caf50;
        border-radius: 0.75rem;
        margin-right: 3rem;
    }
    .debug-panel {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .scope-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .in-scope {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .out-of-scope {
        background-color: #ffcdd2;
        color: #c62828;
    }
    .clarification-badge {
        background-color: #ffe082;
        color: #f57f17;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.75rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_models():
    """Initialize all models (cached to avoid reloading)."""
    try:
        # Suppress warnings
        import warnings
        warnings.filterwarnings('ignore')
        
        # Initialize components
        intent_classifier = IntentClassifier()
        
        # Train if needed
        if not intent_classifier.is_trained:
            training_result = intent_classifier.train()
            if not training_result.get("success"):
                st.warning("⚠️ Intent classifier training in progress...")
        
        emotion_detector = EmotionDetector()
        llm_handler = LLMHandler()
        database = ChatbotDatabase()
        scope_detector = ScopeDetector()
        prompt_engineer = PromptEngineer()
        
        # Phase 2: Enhanced modules for faster, more accurate responses
        session_greeter = SessionGreeter()
        error_recovery = ErrorRecovery()
        tone_detector = EmotionalToneDetector()
        intent_refiner = IntentRefiner()
        
        return {
            "intent_classifier": intent_classifier,
            "emotion_detector": emotion_detector,
            "llm_handler": llm_handler,
            "database": database,
            "scope_detector": scope_detector,
            "prompt_engineer": prompt_engineer,
            "session_greeter": session_greeter,
            "error_recovery": error_recovery,
            "tone_detector": tone_detector,
            "intent_refiner": intent_refiner
        }
    
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None


def chat_interface():
    """Main enhanced chatbot interface."""
    
    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ConversationContext()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    
    if "show_logs" not in st.session_state:
        st.session_state.show_logs = False
    
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    
    if "greeting_shown" not in st.session_state:
        st.session_state.greeting_shown = False
    
    # Initialize models
    models = initialize_models()
    if not models:
        st.error("Failed to initialize models.")
        return
    
    intent_classifier = models["intent_classifier"]
    emotion_detector = models["emotion_detector"]
    llm_handler = models["llm_handler"]
    database = models["database"]
    scope_detector = models["scope_detector"]
    prompt_engineer = models["prompt_engineer"]
    session_greeter = models["session_greeter"]
    error_recovery = models["error_recovery"]
    tone_detector = models["tone_detector"]
    intent_refiner = models["intent_refiner"]
    
    time_context = TimeContext()
    
    # Header
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown("# 🎓 College AI Assistant")
        st.markdown("*Production-Ready Edition with Advanced AI Features*")
    with col2:
        st.markdown(f"**{get_time_of_day().upper()}** ⏰")
    
    st.caption(f"Session ID: {st.session_state.session_id} | Time: {get_current_timestamp()}")
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.markdown("## ⚙️ Controls & Analytics")
        
        # Debug toggle
        st.session_state.show_debug = st.checkbox(
            "🔍 Show Advanced Debug Info",
            value=st.session_state.show_debug,
            help="Display intent, emotion, confidence, scope detection, and more"
        )
        
        # Clear chat
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_context = ConversationContext()
            st.success("✅ Chat cleared!")
            st.rerun()
        
        # View logs
        if st.button("📊 View Session Logs", use_container_width=True):
            st.session_state.show_logs = not st.session_state.show_logs
        
        st.divider()
        
        # Analytics
        st.markdown("### 📈 Analytics")
        analytics = database.get_analytics_summary()
        
        if analytics:
            col1, col2 = st.columns(2)
            with col1:
                total = analytics.get("total_interactions", 0)
                st.metric("📝 Total Interactions", total, delta=None)
                
                conf = analytics.get("average_confidence", 0)
                st.metric("🎯 Avg Confidence", f"{conf:.1%}", delta=None)
            
            with col2:
                time_m = analytics.get("average_response_time", 0)
                st.metric("⚡ Avg Response Time", f"{time_m:.2f}s", delta=None)
                
                llm_stats = llm_handler.get_stats()
                fallback = llm_stats.get('fallback_rate', 0)
                st.metric("🔄 Fallback Rate", f"{fallback:.0f}%", delta=None)
            
            # Top intents
            if analytics.get("top_intents"):
                st.markdown("**🏆 Top Intents**")
                for intent, count in list(analytics["top_intents"].items())[:3]:
                    st.caption(f"• {intent}: {count} queries")
            
            # Out of scope & Clarifications
            out_of_scope = analytics.get("out_of_scope_count", 0)
            clarifications = analytics.get("clarification_count", 0)
            if out_of_scope > 0 or clarifications > 0:
                st.markdown("**⚠️ Special Handling**")
                if out_of_scope > 0:
                    st.caption(f"• Out-of-scope: {out_of_scope}")
                if clarifications > 0:
                    st.caption(f"• Clarifications: {clarifications}")
        
        st.divider()
        
        # Model status
        st.markdown("### ℹ️ System Status")
        
        model_info = intent_classifier.get_model_info()
        status = "✅" if model_info["is_trained"] else "⏳"
        st.caption(f"{status} Intent Classifier: {model_info.get('intents_count', 0)} intents")
        
        groq_status = "✅ Available" if llm_handler.groq_api_key else "⚠️ Fallback Mode"
        st.caption(f"🤖 Groq API: {groq_status}")
        
        st.caption(f"🌍 Emotion Detector: Ready")
        st.caption(f"🎯 Scope Detection: Active")
    
    # Main content area
    st.markdown("---")
    
    # First-time greeting and name collection (Claude-like, warm, no hallucinations)
    if not st.session_state.greeting_shown and len(st.session_state.messages) == 0:
        col_greeting, col_info = st.columns([0.6, 0.4])
        
        with col_greeting:
            st.markdown("### 👋 Welcome!")
            greeting_text = session_greeter.greet(include_prompt=False)
            st.markdown(greeting_text)
        
        with col_info:
            st.info(session_greeter.quick_help())
        
        st.markdown("---")
        
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            user_name = st.text_input(
                "Your name (optional):",
                placeholder="Enter your name...",
                key="name_input"
            )
        with col2:
            if st.button("✨ Start Chat", use_container_width=True):
                st.session_state.user_name = user_name.strip() if user_name else "there"
                st.session_state.greeting_shown = True
                st.session_state.session_start_time = time.time()
                
                # Use SessionGreeter for personalized greeting
                session_greeter_instance = SessionGreeter(
                    user_name=st.session_state.user_name,
                    is_returning=False
                )
                greeting = session_greeter_instance.greet(include_prompt=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": greeting,
                    "debug_info": {
                        "intent": "greeting",
                        "confidence": 1.0,
                        "emotion": "friendly",
                        "llm_source": "session_greeter",
                        "response_time": 0,
                        "is_in_scope": True,
                        "should_clarify": False,
                        "scope_reason": "system_greeting"
                    }
                })
                st.rerun()
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            debug = message.get("debug_info", {})
            in_scope_badge = ""
            if debug.get("is_in_scope") is not None:
                badge_class = "in-scope" if debug["is_in_scope"] else "out-of-scope"
                badge_text = "IN-SCOPE" if debug["is_in_scope"] else "OUT-OF-SCOPE"
                in_scope_badge = f'<span class="scope-badge {badge_class}">{badge_text}</span>'
            
            clarif_badge = ""
            if debug.get("should_clarify"):
                clarif_badge = '<span class="scope-badge clarification-badge">🤔 CLARIFYING</span>'
            
            st.markdown(
                f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]} {in_scope_badge}{clarif_badge}</div>',
                unsafe_allow_html=True
            )
            
            # Show debug info if enabled
            if st.session_state.show_debug and debug:
                with st.expander("🔍 Debug Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Intent", debug.get("intent", "N/A"))
                        st.metric("In-Scope", "✅ Yes" if debug.get("is_in_scope") else "❌ No")
                    with col2:
                        st.metric("Confidence", f"{debug.get('confidence', 0):.0%}")
                        st.metric("Emotion", debug.get("emotion", "N/A"))
                    with col3:
                        st.metric("LLM Source", debug.get("llm_source", "N/A"))
                        st.metric("Response Time", f"{debug.get('response_time', 0):.2f}s")
                    
                    if debug.get("scope_reason"):
                        st.caption(f"📌 Scope Reason: {debug['scope_reason']}")
                    if debug.get("entities"):
                        st.caption(f"🏷️ Entities: {debug['entities']}")
    
    # User input area
    st.markdown("---")
    user_input = st.chat_input(
        placeholder=f"Hi {st.session_state.user_name or 'there'}! Ask me about fees, exams, placements, faculty, or anything college-related...",
        key="user_input"
    )
    
    # Process input
    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        with st.spinner("⚡ Processing your question..."):
            start_time = time.time()
            
            # STEP 1: Intent classification
            classification_result = intent_classifier.predict(user_input)
            intent = classification_result.get("intent", "unknown")
            confidence = classification_result.get("confidence", 0.0)
            
            # STEP 2: Emotion detection
            emotion_result = emotion_detector.detect_emotion(user_input)
            emotion = emotion_result.get("emotion", "neutral")
            
            # STEP 3: PHASE 2 - Intent refinement using context (FAST & ACCURATE)
            history = st.session_state.conversation_context.get_formatted_history()
            refined_intent_result = intent_refiner.refine_intent(
                predicted_intent=intent,
                confidence=confidence,
                user_input=user_input,
                conversation_history=history,
                emotion=emotion
            )
            intent = refined_intent_result["intent"]
            confidence = refined_intent_result["confidence"]
            was_refined = refined_intent_result["refined"]
            
            # STEP 4: PHASE 2 - Emotional tone detection for adaptive responses
            tone_result = tone_detector.detect_tone(user_input, emotion, intent)
            emotional_tone = tone_result["tone_name"]
            tone_guidelines = tone_detector.get_response_guidelines(tone_result)
            
            # STEP 5: Scope detection
            scope_info = scope_detector.get_scope_info(user_input, intent, confidence)
            is_in_scope = scope_info["is_in_scope"]
            scope_reason = scope_info["reason"]
            
            # STEP 6: Error recovery check (for confidence issues)
            if confidence < 0.4:
                error_info = error_recovery.handle_confidence_error(confidence, intent)
                if not error_info.get("should_proceed", True):
                    emotion = "confused"  # Adjust emotion for clarification
            
            # STEP 7: Generate response with ALL enhancements (LLM ALWAYS USED)
            try:
                llm_result = llm_handler.generate_response(
                    user_input=user_input,
                    intent=intent,
                    confidence=confidence,
                    emotion=emotion,
                    conversation_history=history,
                    tone_guidelines=tone_guidelines  # Pass tone for response adaptation
                )
                
                response = llm_result["response"]
                llm_source = llm_result.get("source", "unknown")
                response_time = llm_result.get("time", 0.0)
                should_clarify = llm_result.get("should_clarify", False)
                is_in_scope_llm = llm_result.get("is_in_scope", True)
            
            except Exception as e:
                # Error recovery: graceful fallback
                error_recovery_result = error_recovery.handle_api_error(
                    error=e,
                    operation="llm_response_generation",
                    context={"intent": intent, "user_input": user_input[:50]}
                )
                response = error_recovery_result.get("message", "Let me reconsider that. Could you rephrase?")
                llm_source = "fallback"
                response_time = 0.0
                should_clarify = True
                is_in_scope_llm = True
            
            total_time = time.time() - start_time
        
        # Add assistant message with metadata
        debug_info = {
            "intent": intent,
            "confidence": confidence,
            "emotion": emotion,
            "emotional_tone": emotional_tone,  # PHASE 2: Tone awareness
            "intent_refined": was_refined,     # PHASE 2: Intent refinement
            "tone_emphasis": tone_guidelines.get("emphasis"),  # PHASE 2: Response tone
            "llm_source": llm_source,
            "response_time": response_time,
            "total_processing_time": f"{total_time:.2f}s",
            "is_in_scope": is_in_scope_llm,
            "should_clarify": should_clarify,
            "scope_reason": scope_reason,
            "entities": {}
        }
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "debug_info": debug_info
        })
        
        # Update conversation context
        st.session_state.conversation_context.add_turn(
            user_input,
            response,
            intent,
            confidence,
            emotion
        )
        
        # Log to database with NEW fields
        database.log_interaction(
            user_input=user_input,
            intent=intent,
            confidence=confidence,
            emotion=emotion,
            response=response,
            response_time=response_time,
            llm_source=llm_source,
            is_in_scope=is_in_scope_llm,
            should_clarify=should_clarify,
            scope_reason=scope_reason,
            session_id=st.session_state.session_id
        )
        
        st.rerun()
    
    # Show logs section if requested
    if st.session_state.show_logs:
        st.markdown("---")
        st.markdown("## 📋 Recent Logs & Analytics")
        
        logs = database.get_logs(limit=20)
        if logs:
            log_data = []
            for log in logs:
                scope_icon = "✅" if log.get("is_in_scope") else "❌"
                clarify_icon = "🤔" if log.get("should_clarify") else "✓"
                
                log_data.append({
                    "Time": log.get("timestamp", "N/A")[-8:],
                    "Intent": log.get("intent", "N/A"),
                    "Confidence": f"{log.get('confidence', 0):.0%}",
                    "Emotion": log.get("emotion", "N/A"),
                    "In-Scope": scope_icon,
                    "Clarify": clarify_icon,
                    "LLM": log.get("llm_source", "N/A")
                })
            
            st.dataframe(log_data, use_container_width=True, hide_index=True)
        else:
            st.info("No logs yet. Start a conversation to see logs!")


def main():
    """Main function."""
    try:
        chat_interface()
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")


if __name__ == "__main__":
    main()
