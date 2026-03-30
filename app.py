"""
Streamlit UI for the college chatbot.
Main entry point for the application.
"""

import streamlit as st
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from intent_model import IntentClassifier
from emotion_detector import EmotionDetector
from llm_handler import LLMHandler
from context_manager import ConversationContext
from database import ChatbotDatabase
from utils import (
    get_time_greeting, get_time_of_day, calculate_confidence_percentage,
    is_college_domain_query, truncate_text, get_current_timestamp
)


# Page configuration
st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .debug-panel {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        border-radius: 0.5rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_models():
    """Initialize all models (cached to avoid reloading). Emotion detector lazy-loads on first use."""
    try:
        # Suppress model initialization logs
        import warnings
        warnings.filterwarnings('ignore')
        
        # Initialize intent classifier
        intent_classifier = IntentClassifier()
        
        if not intent_classifier.is_trained:
            training_result = intent_classifier.train()
            if not training_result.get("success"):
                st.warning("⚠️ Intent classifier training failed")
        
        # Initialize emotion detector (lazy-loads on first use, NOT on startup)
        emotion_detector = EmotionDetector()
        
        # Initialize LLM handler
        llm_handler = LLMHandler()
        
        # Initialize database
        database = ChatbotDatabase()
        
        return {
            "intent_classifier": intent_classifier,
            "emotion_detector": emotion_detector,
            "llm_handler": llm_handler,
            "database": database
        }
    
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None


def chat_interface():
    """Main chatbot interface with persistent session memory."""
    
    # Initialize session state (with persistence)
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ConversationContext()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    
    # Initialize models
    models = initialize_models()
    if not models:
        st.error("Failed to initialize models. Please check your setup.")
        return
    
    intent_classifier = models["intent_classifier"]
    emotion_detector = models["emotion_detector"]
    llm_handler = models["llm_handler"]
    database = models["database"]
    
    # Start fresh each session (previous chat history available but not auto-loaded)
    # To restore previous session, user can click "📊 View Logs Summary" in sidebar
    
    # Header
    st.markdown("# 🎓 College AI Assistant")
    st.markdown(f"**{get_time_greeting()}** Welcome to your intelligent college assistant!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        
        # Debug toggle
        st.session_state.show_debug = st.checkbox("Show Debug Info", value=st.session_state.show_debug)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_context.clear_history()
            st.success("Chat history cleared!")
            st.rerun()
        
        # View logs button
        if st.button("📊 View Logs Summary", use_container_width=True):
            st.session_state.show_logs = True
        
        st.divider()
        
        # Statistics
        st.markdown("## 📈 Statistics")
        
        analytics = database.get_analytics_summary()
        if analytics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Interactions", analytics.get("total_interactions", 0))
                st.metric("Avg Confidence", f"{analytics.get('average_confidence', 0):.2%}")
            
            with col2:
                st.metric("Avg Response Time", f"{analytics.get('average_response_time', 0):.2f}s")
                llm_stats = llm_handler.get_stats()
                st.metric("Fallback Rate", f"{llm_stats.get('fallback_rate', 0):.2%}")
            
            if analytics.get("top_intents"):
                st.markdown("### Top Intents")
                for intent, count in list(analytics["top_intents"].items())[:3]:
                    st.write(f"- {intent}: {count}")
        
        st.divider()
        
        # Model info
        st.markdown("## ℹ️ Model Status")
        st.caption("Intent Classifier")
        model_info = intent_classifier.get_model_info()
        st.write(f"- Trained: ✅" if model_info["is_trained"] else "- Trained: ❌")
        st.write(f"- Intents: {model_info['intents_count']}")
        
        st.caption("LLM Handler")
        llm_available = "✅ Available" if llm_handler.groq_api_key else "⚠️ Using Ollama fallback"
        st.write(f"- Groq API: {llm_available}")
        
        st.caption(f"Session: {st.session_state.session_id[:4]}...")
    
    # Main chat area
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', 
                       unsafe_allow_html=True)
            
            # Show debug info if enabled
            if st.session_state.show_debug and message.get("debug_info"):
                debug = message["debug_info"]
                entities_str = ", ".join([f"{k}={v}" for k, v in debug.get('entities', {}).items()])
                if not entities_str:
                    entities_str = "None"
                clarity_status = "[NEEDS CLARITY]" if debug.get('should_clarify') else "[CLEAR]"
                st.markdown(f"""<div class="debug-panel">
                <strong>Debug Info:</strong><br>
                Intent: {debug.get('intent')} {clarity_status} | Confidence: {debug.get('confidence'):.0%}<br>
                Emotion: {debug.get('emotion')} | Source: {debug.get('llm_source')}<br>
                Entities: {entities_str}<br>
                Response Time: {debug.get('response_time'):.2f}s
                </div>""", unsafe_allow_html=True)
    
    # User input
    st.markdown("---")
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_input = st.text_input(
            "Type your message:",
            placeholder="Ask me about fees, exams, placements, or anything about college..."
        )
    with col2:
        submit_button = st.button("Send", use_container_width=True)
    
    # Process user input
    if submit_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Show typing indicator
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            # Step 1: Classify intent
            classification_result = intent_classifier.predict(user_input)
            intent = classification_result.get("intent")
            confidence = classification_result.get("confidence", 0.0)
            
            # Step 2: Detect emotion
            emotion_result = emotion_detector.detect_emotion(user_input)
            emotion = emotion_result.get("emotion", "neutral")
            
            # Step 3: Extract entities and update context
            entities = st.session_state.conversation_context.extract_entities(user_input, intent)
            st.session_state.conversation_context.last_entities = entities
            
            # Step 4: Check if within college domain
            is_in_domain = is_college_domain_query(intent, confidence)
            
            # Step 5: Generate context
            context = st.session_state.conversation_context.get_prompt_context()
            
            # Step 6: Generate response (LLM always generates, no strict domain blocking)
            llm_result = llm_handler.generate_response(
                user_input,
                intent,
                confidence,
                emotion,
                context
            )
            response = llm_result["response"]
            llm_source = llm_result.get("source", "unknown")
            response_time = llm_result.get("time", 0.0)
            should_clarify = llm_result.get("should_clarify", False)
            
            total_time = time.time() - start_time
        
        # Add assistant message to chat
        debug_info = {
            "intent": intent,
            "confidence": confidence,
            "emotion": emotion,
            "llm_source": llm_source,
            "response_time": response_time,
            "is_in_domain": is_in_domain,
            "should_clarify": should_clarify,
            "entities": entities
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
        
        # Log to database
        database.log_interaction(
            user_input,
            intent,
            confidence,
            emotion,
            response,
            response_time,
            llm_source
        )
        
        # Save session for persistence
        database.save_session(
            session_id=st.session_state.session_id,
            messages=st.session_state.messages,
            metadata={
                "total_turns": len(st.session_state.messages) // 2,
                "last_intent": intent,
                "last_emotion": emotion
            }
        )
        
        # Rerun to refresh the display
        st.rerun()
    
    # Show logs summary if requested
    if st.session_state.get("show_logs"):
        st.markdown("---")
        st.markdown("## 📋 Recent Logs")
        
        logs = database.get_logs(limit=10)
        if logs:
            for log in logs:
                timestamp = log.get("timestamp", "N/A")
                intent = log.get("intent", "N/A")
                confidence = log.get("confidence", 0)
                emotion = log.get("emotion", "N/A")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.caption(f"🕐 {timestamp}")
                col2.caption(f"🎯 {intent}")
                col3.caption(f"📊 {confidence:.0%}")
                col4.caption(f"😊 {emotion}")
        
        if st.button("Close Logs"):
            st.session_state.show_logs = False
            st.rerun()


def main():
    """Main function to run the Streamlit app."""
    try:
        chat_interface()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
