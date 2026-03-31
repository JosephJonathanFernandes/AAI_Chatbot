"""
Time-aware context management for intelligent, context-aware responses.
Provides time-based greetings, schedule awareness, and prevents hallucinations.
"""

from datetime import datetime
from typing import Dict, List, Tuple
import json
from utils import load_json_file


class TimeContext:
    """Manages time-aware context for smarter, faster responses."""
    
    def __init__(self, college_data_path: str = "data/college_data.json"):
        """
        Initialize time context manager.
        
        Args:
            college_data_path (str): Path to college knowledge base
        """
        self.college_data = load_json_file(college_data_path)
        self.current_time = datetime.now()
        
        # Define college schedules (can be customized)
        self.college_schedules = {
            "office_hours": "9:00 AM - 5:00 PM",
            "class_timings": "8:00 AM - 4:00 PM",
            "library_hours": "7:00 AM - 9:00 PM",
            "canteen_hours": "8:00 AM - 7:00 PM",
            "lab_sessions": "1:00 PM - 5:00 PM"
        }
    
    def get_time_of_day(self) -> str:
        """
        Determine time of day with finer granularity.
        
        Returns:
            str: One of 'early_morning', 'morning', 'late_morning', 
                 'afternoon', 'evening', 'night', 'late_night'
        """
        hour = self.current_time.hour
        
        if 5 <= hour < 8:
            return "early_morning"
        elif 8 <= hour < 11:
            return "morning"
        elif 11 <= hour < 13:
            return "late_morning"
        elif 13 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 20:
            return "evening"
        elif 20 <= hour < 23:
            return "night"
        else:
            return "late_night"
    
    def get_intelligent_greeting(self, user_name: str = None) -> str:
        """
        Generate an intelligent, time-aware greeting with context.
        
        Args:
            user_name (str): Optional user name for personalization
        
        Returns:
            str: Context-aware greeting message
        """
        time_of_day = self.get_time_of_day()
        hour = self.current_time.hour
        day_name = self.current_time.strftime("%A")
        
        # Creative greetings based on time and context
        greetings_map = {
            "early_morning": [
                "You're an early bird! ☀️ Good morning! What can I help you with?",
                "Rise and shine! ☀️ How can I assist you this morning?",
                "Good morning! The day is just beginning. 🌅",
            ],
            "morning": [
                "Good morning! ☀️ Ready to ace your day? What do you want to know?",
                "Morning! ☀️ How can I help you with college info today?",
                "Hey there! Good morning! 😊",
            ],
            "late_morning": [
                f"Good late morning! 🌤️ It's {self.current_time.strftime('%I:%M %p')}. How can I assist?",
                "Morning's almost over! ☀️ Last minute questions? I'm here to help!",
                "Good morning! 🌤️",
            ],
            "afternoon": [
                "Good afternoon! 👋 Hope you're having a great day!",
                f"Afternoon! 👋 It's {self.current_time.strftime('%I:%M %p')}. What do you need?",
                "Good afternoon! ☕ What's on your mind?",
            ],
            "evening": [
                "Good evening! 🌙 How can I help as the day winds down?",
                f"Evening vibes! 🌙 It's {self.current_time.strftime('%I:%M %p')}. Let's tackle your questions!",
                "Good evening! 🌙 What do you want to know?",
            ],
            "night": [
                "Burning the midnight oil, I see! 🌙 How can I help?",
                "Late night study session? 🌙 I'm here to help!",
                "Night owl mode activated! 🦉 What's your question?",
            ],
            "late_night": [
                "Wow, you're up late! 🌙 Need urgent college info?",
                "It's quite late! 🌙 But I'm always here to help!",
                "Late night learner! 📚🌙 What can I assist with?",
            ]
        }
        
        base_greeting = greetings_map.get(time_of_day, ["Hello! 👋"])[0]
        
        # Add personalization if name is provided
        if user_name:
            base_greeting = base_greeting.replace("Good", f"Good, {user_name}!") \
                                        .replace("Hey there", f"Hey {user_name}")
        
        # Add day-specific context
        if day_name == "Monday":
            base_greeting += " New week, new questions! 💪"
        elif day_name == "Friday":
            base_greeting += " Almost Friday! 🎉"
        elif day_name in ["Saturday", "Sunday"]:
            base_greeting += " Weekend planning? 📚"
        
        return base_greeting
    
    def get_context_awareness_prompt(self) -> str:
        """
        Generate a context-aware system prompt segment based on current time.
        
        Returns:
            str: Prompt text for LLM to be aware of time context
        """
        time_of_day = self.get_time_of_day()
        hour = self.current_time.hour
        day_name = self.current_time.strftime("%A")
        date_str = self.current_time.strftime("%B %d, %Y")
        
        context_prompts = {
            "early_morning": (
                f"It's early morning ({self.current_time.strftime('%I:%M %A')}). "
                "College offices may not be open yet. Avoid suggesting immediate office visits. "
                "If asked about urgent matters, suggest emailing or visiting during office hours."
            ),
            "morning": (
                f"It's {self.current_time.strftime('%I:%M %A')}. College is actively operational. "
                "Classes are likely happening. Office hours are active. "
                "You can reference ongoing activities and suggest real-time interactions."
            ),
            "late_morning": (
                f"It's late morning/early afternoon ({self.current_time.strftime('%I:%M')}). "
                "Most classes are ongoing. Lunch time is approaching. "
                "Reference this timing when giving advice."
            ),
            "afternoon": (
                f"It's afternoon ({self.current_time.strftime('%I:%M %p')}). "
                "Lab sessions and afternoon classes are ongoing. "
                "It's a good time for campus visits or office inquiries."
            ),
            "evening": (
                f"It's evening ({self.current_time.strftime('%I:%M %p')}). "
                "College offices are closing. Library is still open. "
                "For urgent matters, suggest next morning or online resources."
            ),
            "night": (
                f"It's night ({self.current_time.strftime('%I:%M %p')}). "
                "College is mostly closed. Emergency contacts may be limited. "
                "Suggest email, website, or next business day contact."
            ),
            "late_night": (
                f"It's very late ({self.current_time.strftime('%I:%M %p')}). "
                "College facilities are completely closed. "
                "Only provide information; suggest visiting during business hours for immediate help."
            )
        }
        
        time_context = context_prompts.get(time_of_day, "")
        
        # Add weekend context
        if day_name in ["Saturday", "Sunday"]:
            time_context += (
                f" Note: It's {day_name}, a weekend day. "
                "Most college offices are typically closed. "
                "Direct users to online resources, emergency contact numbers, or suggest visiting on weekdays."
            )
        
        return time_context
    
    def get_relevant_schedule_info(self, intent: str) -> str:
        """
        Return relevant schedule/timing information based on query intent.
        Quick contextual fact that prevents hallucination.
        
        Args:
            intent (str): User's query intent (e.g., 'library', 'office', 'classes')
        
        Returns:
            str: Relevant schedule information or empty string
        """
        schedule_map = {
            "library": f"Library hours: {self.college_schedules.get('library_hours')}",
            "classes": f"Class timings: {self.college_schedules.get('class_timings')}",
            "office": f"Office hours: {self.college_schedules.get('office_hours')}",
            "office_hours": f"Office hours: {self.college_schedules.get('office_hours')}",
            "canteen": f"Canteen hours: {self.college_schedules.get('canteen_hours')}",
            "lab": f"Lab sessions: {self.college_schedules.get('lab_sessions')}",
            "admin": f"Administrative office hours: {self.college_schedules.get('office_hours')}"
        }
        
        return schedule_map.get(intent.lower(), "")
    
    def get_hallucination_check_prompt(self) -> str:
        """
        Get anti-hallucination directive for LLM.
        
        Returns:
            str: Instruction to prevent hallucination
        """
        return (
            "CRITICAL: Only answer based on the college data provided. "
            "If information is not in the college database, explicitly state: "
            "'I don't have this information in my college database. "
            "Please contact the college office directly or visit the official website.' "
            "Never invent or guess faculty names, phone numbers, email addresses, or specific details. "
            "If unsure, ask for clarification or direct to official channels."
        )
    
    def is_office_open(self) -> Tuple[bool, str]:
        """
        Check if office is currently open based on time.
        
        Returns:
            Tuple[bool, str]: (is_open, reason)
        """
        hour = self.current_time.hour
        day = self.current_time.weekday()
        
        # Assume office open Mon-Fri, 9 AM - 5 PM
        is_weekday = day < 5  # 0-4 are Mon-Fri
        is_business_hours = 9 <= hour < 17
        
        if not is_weekday:
            return False, "Office is closed on weekends"
        elif not is_business_hours:
            return False, f"Office is closed (current time: {self.current_time.strftime('%I:%M %p')})"
        else:
            return True, "Office is currently open"
    
    def get_college_data_snippet(self, key: str) -> str:
        """
        Safely retrieve college data to prevent hallucination.
        
        Args:
            key (str): Key to retrieve from college data
        
        Returns:
            str: Data value or "Information not available"
        """
        if not self.college_data:
            return "College data not loaded"
        
        value = self.college_data.get(key)
        if value:
            return str(value)
        else:
            return f"'{key}' information not available in database"
    
    def get_context_summary(self) -> Dict:
        """
        Get a complete time and context summary.
        
        Returns:
            Dict: Summary of all time-based context
        """
        office_open, office_status = self.is_office_open()
        
        return {
            "current_time": self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_of_day": self.get_time_of_day(),
            "day_of_week": self.current_time.strftime("%A"),
            "is_weekend": self.current_time.weekday() >= 5,
            "office_open": office_open,
            "office_status": office_status,
            "context_awareness": self.get_context_awareness_prompt(),
            "schedules": self.college_schedules
        }
