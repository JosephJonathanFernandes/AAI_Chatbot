"""
Utility functions for the college chatbot.
Includes time awareness, helpers, and common utilities.
"""

from datetime import datetime
import json
from pathlib import Path


def get_time_of_day():
    """
    Determine the time of day based on current hour.
    
    Returns:
        str: One of 'morning', 'afternoon', 'evening', 'night'
    """
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def get_time_greeting():
    """
    Get a time-appropriate greeting.
    
    Returns:
        str: A greeting message based on current time
    """
    time_of_day = get_time_of_day()
    greetings = {
        "morning": "Good morning! ☀️",
        "afternoon": "Good afternoon! 👋",
        "evening": "Good evening! 🌙",
        "night": "Hello, it's quite late! 🌙"
    }
    return greetings.get(time_of_day, "Hello!")


def get_current_timestamp():
    """
    Get the current timestamp in a standardized format.
    
    Returns:
        str: Timestamp in format 'YYYY-MM-DD HH:MM:SS'
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json_file(filepath):
    """
    Load a JSON file safely.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON content, or empty dict if file not found
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in file: {filepath}")
        return {}


def save_json_file(filepath, data):
    """
    Save data to a JSON file.
    
    Args:
        filepath (str): Path to save the file
        data (dict): Data to save
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def is_weekend():
    """
    Check if today is a weekend.
    
    Returns:
        bool: True if Saturday or Sunday, False otherwise
    """
    weekday = datetime.now().weekday()
    return weekday >= 5  # 5=Saturday, 6=Sunday


def normalize_text(text):
    """
    Normalize user input text.
    
    Args:
        text (str): Raw user input
        
    Returns:
        str: Normalized text (lowercase, stripped)
    """
    return text.lower().strip()


def truncate_text(text, max_length=500):
    """
    Truncate text to a maximum length.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text with ellipsis if needed
    """
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def calculate_confidence_percentage(confidence_score):
    """
    Convert confidence score to percentage.
    
    Args:
        confidence_score (float): Score between 0 and 1
        
    Returns:
        float: Percentage between 0 and 100
    """
    return round(confidence_score * 100, 2)


def is_college_domain_query(intent_tag, confidence):
    """
    Determine if a query is within college domain based on intent and confidence.
    
    Args:
        intent_tag (str): The classified intent tag
        confidence (float): Confidence score of the classification
        
    Returns:
        bool: True if query is within college domain, False otherwise
    """
    college_intents = [
        "fees", "exams", "timetable", "placements", "faculty",
        "holidays", "library", "admission", "departments", "greeter", "gratitude"
    ]
    return intent_tag in college_intents and confidence >= 0.5


def format_response_for_display(response, max_display_length=1000):
    """
    Format response for UI display.
    
    Args:
        response (str): The full response text
        max_display_length (int): Maximum length for display
        
    Returns:
        str: Formatted response
    """
    if len(response) > max_display_length:
        return response[:max_display_length] + "\n\n[Response truncated. Full content available in logs.]"
    return response


def get_context_aware_prompt_prefix(intent, confidence, emotion, time_of_day):
    """
    Build a context-aware prompt prefix for the LLM.
    
    Args:
        intent (str): Detected intent
        confidence (float): Intent confidence score
        emotion (str): Detected emotion
        time_of_day (str): Current time of day
        
    Returns:
        str: Prompt prefix with context
    """
    prefix = f"""You are a helpful college assistant.
Current context:
- Intent: {intent}
- Confidence: {confidence:.2%}
- User emotion: {emotion}
- Time of day: {time_of_day}

Respond as a knowledgeable, helpful college assistant. If the query is outside the college domain, respond with:
"This is beyond my scope as a college assistant."
"""
    return prefix
