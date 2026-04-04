"""
Session greeter for college chatbot.
Provides contextual, friendly greetings when session starts.
Fast responses with Claude-like personality, no hallucinations.
"""

import random
from datetime import datetime
from utils import get_time_of_day, get_time_greeting


class SessionGreeter:
    """Generates warm, contextual session greetings."""
    
    # Morning greetings
    MORNING_GREETINGS = [
        "Good morning! 🌅 I'm your College AI Assistant. Ready to help with admissions, courses, or campus info?",
        "Rise and shine! ☀️ What college questions can I help you with today?",
        "Good morning! 🎓 Welcome back! How can I assist you today?",
        "Morning! 📚 Let's tackle your college questions. What's on your mind?",
    ]
    
    # Afternoon greetings
    AFTERNOON_GREETINGS = [
        "Good afternoon! ☀️ I'm here to help with all your college questions.",
        "Afternoon! 🎓 What would you like to know about our college?",
        "Hey there! 👋 Welcome to your College AI Assistant. How can I help?",
        "Good afternoon! 📚 Ready to answer your college questions!",
    ]
    
    # Evening greetings
    EVENING_GREETINGS = [
        "Good evening! 🌙 I'm your College AI Assistant. What can I help you with?",
        "Evening! 🌟 Let's explore your college questions together.",
        "Good evening! 📖 What college info can I provide for you?",
        "Hey! 🎓 Burning college questions? I'm here to answer them!",
    ]
    
    # Context-based opening prompts
    CONTEXT_PROMPTS = {
        "first_time": "I'm Claude-like, accurate, and here to help with college info without any nonsense. Fire away! 💡",
        "returning": "Welcome back! I remember our chat. What else would you like to know?",
        "help": "I can help with:\n• Admissions & fees\n• Courses & faculty\n• Placements & campus life\n• Exams & grades\n\nWhat interests you?",
    }
    
    def __init__(self, user_name: str = None, is_returning: bool = False):
        """
        Initialize session greeter.
        
        Args:
            user_name (str): User's name (optional)
            is_returning (bool): Whether user is returning
        """
        self.user_name = user_name
        self.is_returning = is_returning
        self.session_start = datetime.now()
    
    def greet(self, include_prompt: bool = True) -> str:
        """
        Generate a warm contextual greeting.
        
        Args:
            include_prompt (bool): Include context-based prompt
            
        Returns:
            str: Greeting message
        """
        # Get time-appropriate greeting
        time_of_day = get_time_of_day()
        
        if time_of_day == "morning":
            greeting = random.choice(self.MORNING_GREETINGS)
        elif time_of_day == "afternoon":
            greeting = random.choice(self.AFTERNOON_GREETINGS)
        else:  # evening/night
            greeting = random.choice(self.EVENING_GREETINGS)
        
        # Personalize with name if provided
        if self.user_name:
            greeting = greeting.replace("!", f", {self.user_name}!").replace("?", f", {self.user_name}?")
        
        # Add context prompt
        if include_prompt:
            context_key = "returning" if self.is_returning else "first_time"
            prompt = self.CONTEXT_PROMPTS[context_key]
            greeting = f"{greeting}\n\n{prompt}"
        
        return greeting
    
    def quick_help(self) -> str:
        """Get quick help/options menu."""
        return self.CONTEXT_PROMPTS["help"]
    
    def get_session_info(self) -> dict:
        """Get session metadata."""
        return {
            "session_start": self.session_start.isoformat(),
            "user_name": self.user_name,
            "is_returning": self.is_returning,
            "time_of_day": get_time_of_day(),
        }
