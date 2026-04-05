"""
Streamlit UI for the College AI Assistant - PREMIUM REDESIGNED EDITION
Modern, Minimalist, ChatGPT-like Interface

Features:
- Premium gradient design with soft colors
- Smooth animations and transitions
- Emotion-aware visual feedback
- Modern chat bubble design
- Dark mode support
- Quick action buttons
- Responsive layout
- Analytics dashboard
- Enhanced debug UI
- Typing indicator animation
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
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/JosephJonathanFernandes/AAI_Chatbot",
        "Report a bug": "https://github.com/JosephJonathanFernandes/AAI_Chatbot/issues",
    }
)

# ============================================================================
# PREMIUM CSS STYLING - MODERN DESIGN
# ============================================================================

MODERN_CSS = """
<style>
    /* Root color variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --neutral-bg: #ffffff;
        --neutral-text: #1f2937;
        --light-bg: #f9fafb;
        --border-color: #e5e7eb;
    }

    /* Overall layout */
    .main {
        background: #ffffff;
    }

    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }

    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 0px;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.95;
        margin: 0;
    }

    /* Chat Message Containers */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .message-wrapper {
        display: flex;
        flex-direction: column;
        animation: slideInUp 0.3s ease-out;
    }

    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* User Message */
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        gap: 0.75rem;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
        font-size: 0.95rem;
        line-height: 1.5;
        animation: fadeInRight 0.3s ease-out;
    }

    @keyframes fadeInRight {
        from {
            opacity: 0;
            transform: translateX(10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Assistant Message */
    .assistant-message-container {
        display: flex;
        justify-content: flex-start;
        gap: 0.75rem;
        align-items: flex-start;
    }

    .assistant-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.2rem;
        flex-shrink: 0;
    }

    .assistant-message {
        background-color: #f3f4f6;
        color: #1f2937;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        font-size: 0.95rem;
        line-height: 1.5;
        animation: fadeInLeft 0.3s ease-out;
        border: 1px solid #e5e7eb;
    }

    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 1rem 1.5rem;
        background-color: #f3f4f6;
        border-radius: 18px;
        width: fit-content;
    }

    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #9ca3af;
        animation: typing 1.4s infinite;
    }

    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }

    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }

    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }

    /* Badge Styling */
    .badge-container {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .badge-in-scope {
        background: linear-gradient(135deg, #d1fae5 0%, #bbf7d0 100%);
        color: #065f46;
        border: 1px solid #6ee7b7;
    }

    .badge-out-of-scope {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #7f1d1d;
        border: 1px solid #fca5a5;
    }

    .badge-clarifying {
        background: linear-gradient(135deg, #fef3c7 0%, #fcd34d 100%);
        color: #92400e;
        border: 1px solid #fbbf24;
    }

    /* Emotion Indicator */
    .emotion-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        padding: 0.35rem 0.75rem;
        background: #f0f9ff;
        border-radius: 8px;
        color: #0369a1;
    }

    /* Debug Panel */
    .debug-panel {
        background: linear-gradient(135deg, #fef3c7 0%, #fef08a 100%);
        border: 1px solid #fcd34d;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.8rem;
        font-size: 0.85rem;
        color: #92400e;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.15);
    }

    .debug-label {
        font-weight: 600;
        color: #b45309;
        margin-bottom: 0.35rem;
    }

    .debug-item {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(245, 158, 11, 0.2);
    }

    .debug-item:last-child {
        border-bottom: none;
    }

    /* Sidebar Styling */
    .sidebar-section {
        background: #f9fafb;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }

    .sidebar-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        letter-spacing: -0.3px;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 0.75rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.25);
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.75rem;
        opacity: 0.9;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25) !important;
        font-size: 0.95rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Input Area */
    .stChatInputContainer {
        background: white;
        border-top: 1px solid #e5e7eb;
        padding: 1.5rem 0;
    }

    .stChatInput {
        border: 2px solid #e5e7eb !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }

    .stChatInput:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }

    /* Welcome Card */
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
    }

    .welcome-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .welcome-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-bottom: 1.5rem;
    }

    /* Quick Action Buttons */
    .quick-actions {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-bottom: 2rem;
        justify-content: center;
    }

    .quick-action-btn {
        background: #f3f4f6;
        border: 2px solid #e5e7eb;
        color: #1f2937;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }

    .quick-action-btn:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25);
    }

    /* Status Indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-online {
        background: #dcfce7;
        color: #166534;
    }

    .status-offline {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }

    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Confidence Bar */
    .confidence-bar {
        height: 6px;
        background: #e5e7eb;
        border-radius: 3px;
        margin-top: 0.5rem;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 2rem 0;
    }

    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        .assistant-message {
            background-color: #2d3748;
            color: #e2e8f0;
            border-color: #4a5568;
        }

        .metric-card {
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .sidebar-section {
            background: #1a202c;
            border-color: #2d3748;
        }

        .debug-panel {
            background: rgba(245, 158, 11, 0.1);
            border-color: rgba(245, 158, 11, 0.3);
        }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .user-message, .assistant-message {
            max-width: 85%;
        }

        .header-title {
            font-size: 2rem;
        }

        .metric-card {
            padding: 0.75rem;
        }

        .metric-value {
            font-size: 1.25rem;
        }
    }

    /* Loading State */
    .loading-pulse {
        animation: pulse 1.5s ease-in-out infinite;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)


@st.cache_resource
def initialize_models():
    """Initialize all models (cached to avoid reloading)."""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        intent_classifier = IntentClassifier()
        if not intent_classifier.is_trained:
            training_result = intent_classifier.train()
            if not training_result.get("success"):
                st.warning("⚠️ Intent classifier training in progress...")
        
        emotion_detector = EmotionDetector()
        llm_handler = LLMHandler()
        database = ChatbotDatabase()
        scope_detector = ScopeDetector()
        prompt_engineer = PromptEngineer()
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


def get_emotion_emoji(emotion: str) -> str:
    """Get emoji for emotion type."""
    emotion_map = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "confused": "😕",
        "neutral": "😐",
        "excited": "🤩",
        "worried": "😟",
        "frustrated": "😤"
    }
    return emotion_map.get(emotion.lower(), "😐")


def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= 0.8:
        return "🟢"
    elif confidence >= 0.6:
        return "🟡"
    else:
        return "🔴"


def render_message_badges(debug_info: dict) -> str:
    """Render badges for message."""
    badges_html = '<div class="badge-container">'
    
    # Scope badge
    if debug_info.get("is_in_scope") is not None:
        if debug_info["is_in_scope"]:
            badges_html += '<span class="badge badge-in-scope">✓ In-Scope</span>'
        else:
            badges_html += '<span class="badge badge-out-of-scope">✗ Out-of-Scope</span>'
    
    # Clarification badge
    if debug_info.get("should_clarify"):
        badges_html += '<span class="badge badge-clarifying">🤔 Clarifying</span>'
    
    badges_html += '</div>'
    return badges_html


def render_debug_panel(debug_info: dict) -> str:
    """Render enhanced debug panel."""
    confidence = debug_info.get("confidence", 0)
    confidence_pct = f"{confidence * 100:.0f}%"
    emotion = debug_info.get("emotion", "neutral")
    
    html = f"""
    <div class="debug-panel">
        <div class="debug-label">🔍 Processing Details</div>
        <div class="debug-item">
            <span>Intent</span>
            <span><strong>{debug_info.get('intent', 'N/A')}</strong></span>
        </div>
        <div class="debug-item">
            <span>Confidence</span>
            <span><strong>{get_confidence_color(confidence)} {confidence_pct}</strong></span>
        </div>
        <div class="debug-item">
            <span>Emotion</span>
            <span><strong>{get_emotion_emoji(emotion)} {emotion.capitalize()}</strong></span>
        </div>
        <div class="debug-item">
            <span>Response Time</span>
            <span><strong>{debug_info.get('response_time', 0):.2f}s</strong></span>
        </div>
        <div class="debug-item">
            <span>LLM Source</span>
            <span><strong>{debug_info.get('llm_source', 'N/A')}</strong></span>
        </div>
        <div class="debug-item">
            <span>Scope Reason</span>
            <span><strong>{debug_info.get('scope_reason', 'N/A')}</strong></span>
        </div>
    </div>
    """
    return html


def get_emotion_aware_tone(emotion: str) -> str:
    """Get response tone prefix based on user emotion for empathetic UX."""
    tone_prefixes = {
        "sad": "💙 I understand. ",
        "angry": "😌 I appreciate your concern. ",
        "confused": "🤔 Let me clarify: ",
        "worried": "✨ Relax, here's the info: ",
        "frustrated": "😌 Let me help: ",
        "excited": "🎉 Excellent! ",
        "happy": "😊 Great! ",
    }
    return tone_prefixes.get(emotion.lower(), "")


def display_streaming_response(response_text: str, llm_handler: LLMHandler, emotion: str) -> None:
    """
    Display response with word-by-word streaming for better UX.
    Uses st.write_stream() for progressive rendering (Streamlit 1.23+).
    
    Args:
        response_text (str): Full response to stream
        llm_handler: LLM handler instance (for access to stream_response_tokens)
        emotion (str): User emotion for styling
    """
    # Create a container for the response
    response_container = st.container()
    
    with response_container:
        # Display as streaming text using Streamlit's write_stream
        try:
            # Use write_stream if available (Streamlit 1.23+)
            if hasattr(st, 'write_stream'):
                st.write_stream(llm_handler.stream_response_tokens(response_text))
            else:
                # Fallback for older Streamlit versions
                st.write(response_text)
        except Exception as e:
            # Fallback if streaming fails
            st.write(response_text)


def render_out_of_scope_handler(scope_reason: str) -> str:
    """Render attractive out-of-scope message with smart suggestions."""
    suggestions_map = {
        "personal": "• 📱 **Student Support Office** - Personal help\n• 📧 support@college.edu",
        "general_knowledge": "• 🔍 Try Google or Wikipedia\n• 📚 General knowledge resources",
        "technical": "• 💻 **IT Helpdesk** - Tech issues\n• 📞 ext. 5555 or it@college.edu",
        "unrelated": "• 🎓 I'm here for college topics\n• 💡 Ask about admissions, exams, placements, or campus life",
        "inappropriate": "• 📋 Keep it professional please\n• 🎓 Focus on campus-related questions",
    }
    
    suggestion_text = suggestions_map.get(scope_reason, suggestions_map["unrelated"])
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.08), rgba(244, 67, 54, 0.05));
        border-left: 4px solid #F44336;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #C62828;
    ">
        <div style="font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            ⚠️ Outside College Scope
        </div>
        <div style="font-size: 0.9rem; opacity: 0.85; margin-bottom: 0.75rem;">
            I'm specialized in college topics. Here's where you can go instead:
        </div>
        <div style="font-size: 0.85rem; line-height: 1.7; color: #D32F2F;">
            {suggestion_text}
        </div>
    </div>
    """
    return html


def chat_interface():
    """Main enhanced chatbot interface."""
    
    # ========== Initialize Session State ==========
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = ConversationContext()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "greeting_shown" not in st.session_state:
        st.session_state.greeting_shown = False
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    
    # Apply dynamic dark mode CSS based on session state
    if st.session_state.dark_mode:
        DARK_MODE_CSS = """
        <style>
            /* Override light mode for dark mode */
            .main { background: #1a1a1a; }
            .stChatMessage { background: #2d2d2d !important; }
            
            .assistant-message {
                background-color: #2d2d2d !important;
                color: #e0e0e0 !important;
                border-color: #404040 !important;
            }
            
            .assistant-avatar {
                background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            }
            
            .badge-container { background: rgba(0,0,0,0.3); }
            
            .debug-panel {
                background: rgba(245, 158, 11, 0.15) !important;
                border-color: rgba(245, 158, 11, 0.4) !important;
                color: #fbbf24 !important;
            }
            
            .stChatInput {
                background-color: #2d2d2d !important;
                color: #e0e0e0 !important;
                border-color: #404040 !important;
            }
            
            .stChatInput:focus {
                background-color: #333333 !important;
            }
            
            .sidebar-section {
                background: #2d2d2d !important;
                border-color: #404040 !important;
                color: #e0e0e0 !important;
            }
            
            .sidebar-title { color: #e0e0e0 !important; }
            
            /* Keep header gradient in both modes */
            .header-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
        </style>
        """
        st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
    
    # ========== Initialize Models ==========
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
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        # Sidebar Header
        st.markdown("## ⚙️ Settings")
        
        # Toggle controls
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_debug = st.checkbox(
                "Debug",
                value=st.session_state.show_debug,
                help="Show detailed processing info"
            )
        with col2:
            st.session_state.dark_mode = st.checkbox(
                "Dark",
                value=st.session_state.dark_mode,
                help="Toggle dark mode"
            )
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_context = ConversationContext()
                st.success("Chat cleared!")
                st.rerun()
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.greeting_shown = False
                st.rerun()
        
        st.divider()
        
        # Analytics Section
        st.markdown("### 📊 Analytics")
        analytics = database.get_analytics_summary()
        
        if analytics:
            # Metrics in cards
            col1, col2 = st.columns(2)
            with col1:
                total = analytics.get("total_interactions", 0)
                st.metric("Queries", total, label_visibility="collapsed")
            with col2:
                conf = analytics.get("average_confidence", 0)
                st.metric("Confidence", f"{conf:.0%}", label_visibility="collapsed")
            
            col1, col2 = st.columns(2)
            with col1:
                time_m = analytics.get("average_response_time", 0)
                st.metric("Avg Time", f"{time_m:.1f}s", label_visibility="collapsed")
            with col2:
                llm_stats = llm_handler.get_stats()
                fallback = llm_stats.get('fallback_rate', 0)
                st.metric("Fallback", f"{fallback:.0f}%", label_visibility="collapsed")
            
            # Top intents
            if analytics.get("top_intents"):
                st.markdown("**Top Intents**")
                for intent, count in list(analytics["top_intents"].items())[:3]:
                    st.caption(f"📌 {intent}: {count}")
        
        st.divider()
        
        # System Status
        st.markdown("### ℹ️ System")
        
        model_info = intent_classifier.get_model_info()
        status = "✅" if model_info["is_trained"] else "⏳"
        st.caption(f"{status} Classifier: {model_info.get('intents_count', 0)} intents")
        
        groq_status = "✅" if llm_handler.groq_api_key else "⚠️"
        st.caption(f"{groq_status} Groq API")
        
        st.caption(f"🌍 Emotion Detector: Active")
        st.caption(f"🎯 Scope Detection: Active")
        
        st.divider()
        
        # Session info
        st.markdown("### 🔑 Session")
        st.caption(f"ID: `{st.session_state.session_id}`")
        st.caption(f"Time: {get_current_timestamp()}")
    
    # ========== MAIN CONTENT ==========
    
    # Header
    st.markdown(f"""
    <div class="header-container">
        <div class="header-title">🎓 College AI Assistant</div>
        <div class="header-subtitle">Smart, Context-Aware Campus Companion</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== WELCOME SCREEN ==========
    if not st.session_state.greeting_shown and len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to Your Campus Assistant</div>
            <div class="welcome-subtitle">Ask me anything about fees, exams, placements, or campus life</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Name input
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            user_name = st.text_input(
                "What's your name?",
                placeholder="Enter your name (optional)...",
                key="name_input"
            )
        with col2:
            if st.button("Start", use_container_width=True, type="primary"):
                st.session_state.user_name = user_name.strip() if user_name else "there"
                st.session_state.greeting_shown = True
                st.session_state.session_start_time = time.time()
                
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
        
        # Quick action suggestions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**💡 Try asking about:**")
        
        quick_topics = [
            ("💰 Fees", "What are the admission fees?"),
            ("📅 Exams", "When are the exams scheduled?"),
            ("🎯 Placements", "Tell me about placements"),
            ("🏢 Campus", "What facilities are available?"),
        ]
        
        cols = st.columns(len(quick_topics))
        for idx, (label, query) in enumerate(quick_topics):
            if cols[idx].button(label, use_container_width=True):
                st.session_state.user_name = "there"
                st.session_state.greeting_shown = True
                session_greeter_instance = SessionGreeter(user_name="there", is_returning=False)
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
    
    # ========== CHAT HISTORY ==========
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-wrapper">
                    <div class="user-message-container">
                        <div class="user-message">{message["content"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                debug = message.get("debug_info", {})
                emotion = debug.get("emotion", "neutral")
                is_in_scope = debug.get("is_in_scope", True)
                scope_reason = debug.get("scope_reason", "")
                
                # Assistant message with emotion-aware styling
                st.markdown(f"""
                <div class="message-wrapper">
                    <div class="assistant-message-container">
                        <div class="assistant-avatar">{get_emotion_emoji(emotion)}</div>
                        <div class="assistant-message">{message["content"]}</div>
                    </div>
                    {render_message_badges(debug)}
                </div>
                """, unsafe_allow_html=True)
                
                # Out-of-scope handler with suggestions
                if not is_in_scope and scope_reason:
                    st.markdown(render_out_of_scope_handler(scope_reason), unsafe_allow_html=True)
                
                # Debug panel
                if st.session_state.show_debug and debug:
                    st.markdown(render_debug_panel(debug), unsafe_allow_html=True)
    
    # ========== INPUT AREA ==========
    st.markdown("---")
    user_input = st.chat_input(
        placeholder=f"Hi {st.session_state.user_name}! Ask about fees, exams, placements..."
    )
    
    # ========== SUGGESTION CHIPS ==========
    # Show smart suggestions based on last message or default suggestions
    if st.session_state.greeting_shown and len(st.session_state.messages) > 0:
        st.markdown(
            """
            <div style="margin-top: 1rem; margin-bottom: 0.5rem; text-align: center; color: #6B7280; font-size: 0.85rem;">
                💡 Quick Questions:
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        suggestions = [
            ("💰 Fees", "Tell me about tuition fees and payment options"),
            ("📝 Exams", "What's the exam schedule?"),
            ("💼 Placements", "Tell me about placements"),
            ("🏢 Campus", "What facilities are available?"),
        ]
        
        cols = [col1, col2, col3, col4]
        for idx, (label, query) in enumerate(suggestions):
            if cols[idx].button(label, use_container_width=True, key=f"suggest_{idx}"):
                st.session_state.messages.append({
                    "role": "user",
                    "content": query
                })
                st.rerun()
    
    # ========== PROCESS INPUT ==========
    # Check if there's an unprocessed user message (from quick questions button)
    last_is_user = len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user"
    should_process = bool(user_input) or last_is_user
    
    if should_process:
        # If user typed something new, add it to messages
        if user_input:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            message_to_process = user_input
        else:
            # Use the last unprocessed user message
            message_to_process = st.session_state.messages[-1]["content"]
        
        # Show typing indicator
        with st.spinner("✨ Thinking..."):
            start_time = time.time()
            
            # Intent classification
            classification_result = intent_classifier.predict(message_to_process)
            intent = classification_result.get("intent", "unknown")
            confidence = classification_result.get("confidence", 0.0)
            
            # Emotion detection
            emotion_result = emotion_detector.detect_emotion(message_to_process)
            emotion = emotion_result.get("emotion", "neutral")
            
            # Intent refinement
            history = st.session_state.conversation_context.get_history()  # Get list of dicts
            refined_intent_result = intent_refiner.refine_intent(
                predicted_intent=intent,
                confidence=confidence,
                user_input=message_to_process,
                conversation_history=history,
                emotion=emotion
            )
            intent = refined_intent_result["intent"]
            confidence = refined_intent_result["confidence"]
            was_refined = refined_intent_result["refined"]
            
            # Tone detection
            tone_result = tone_detector.detect_tone(message_to_process, emotion, intent)
            emotional_tone = tone_result.get("tone_name", "informative") if isinstance(tone_result, dict) else "informative"
            tone_guidelines = tone_detector.get_response_guidelines(tone_result) if isinstance(tone_result, dict) else {}
            
            # Ensure tone_guidelines is a dict
            if not isinstance(tone_guidelines, dict):
                tone_guidelines = {}
            
            # Scope detection
            scope_info = scope_detector.get_scope_info(message_to_process, intent, confidence)
            is_in_scope = scope_info["is_in_scope"]
            scope_reason = scope_info["reason"]
            
            # Error recovery
            if confidence < 0.4:
                error_info = error_recovery.handle_confidence_error(confidence, intent)
                if not error_info.get("should_proceed", True):
                    emotion = "confused"
            
            # Generate response
            try:
                llm_result = llm_handler.generate_response(
                    user_input=message_to_process,
                    intent=intent,
                    confidence=confidence,
                    emotion=emotion,
                    conversation_history=history,
                    tone_guidelines=tone_guidelines
                )
                
                response = llm_result["response"]
                llm_source = llm_result.get("source", "unknown")
                response_time = llm_result.get("time", 0.0)
                should_clarify = llm_result.get("should_clarify", False)
                is_in_scope_llm = llm_result.get("is_in_scope", True)
            
            except Exception as e:
                error_recovery_result = error_recovery.handle_api_error(
                    error=e,
                    operation="llm_response_generation",
                    context={"intent": intent, "user_input": message_to_process[:50]}
                )
                # Ensure error_recovery_result is a dict before calling .get()
                if not isinstance(error_recovery_result, dict):
                    error_recovery_result = {"message": "Let me reconsider that. Could you rephrase?"}
                response = error_recovery_result.get("message", "Let me reconsider that. Could you rephrase?")
                llm_source = "fallback"
                response_time = 0.0
                should_clarify = True
                is_in_scope_llm = True
            
            total_time = time.time() - start_time
        
        # Add assistant response
        debug_info = {
            "intent": intent,
            "confidence": confidence,
            "emotion": emotion,
            "emotional_tone": emotional_tone,
            "intent_refined": was_refined,
            "tone_emphasis": tone_guidelines.get("emphasis"),
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
        
        # Update context
        st.session_state.conversation_context.add_turn(
            message_to_process,
            response,
            intent,
            confidence,
            emotion
        )
        
        # Log to database
        database.log_interaction(
            user_input=message_to_process,
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


def main():
    """Main function."""
    try:
        chat_interface()
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")


if __name__ == "__main__":
    main()
