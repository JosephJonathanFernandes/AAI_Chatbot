"""
SQLite database handler for logging chatbot interactions.
Manages schema creation, logging, and analytics.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from utils import get_current_timestamp


class ChatbotDatabase:
    """Handles SQLite logging for chatbot interactions."""
    
    def __init__(self, db_path="chatbot.db"):
        """
        Initialize database connection.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.create_tables()
    
    def get_connection(self):
        """
        Get a database connection.
        
        Returns:
            sqlite3.Connection: Database connection
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None
    
    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        try:
            conn = self.get_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # Main logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    intent TEXT,
                    confidence REAL,
                    emotion TEXT,
                    response TEXT,
                    response_time REAL,
                    llm_source TEXT DEFAULT 'groq'
                )
            """)
            
            # Session tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_messages INTEGER DEFAULT 0,
                    primary_intents TEXT
                )
            """)
            
            # Analytics cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    total_interactions INTEGER,
                    average_confidence REAL,
                    top_intent TEXT,
                    avg_response_time REAL
                )
            """)
            
            # Chat sessions table for persistence
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    total_turns INTEGER,
                    metadata TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            print(f"Database initialized successfully: {self.db_path}")
        
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
    
    def log_interaction(self, user_input, intent, confidence, emotion, response, response_time=0, llm_source="groq"):
        """
        Log a chatbot interaction.
        
        Args:
            user_input (str): User's input message
            intent (str): Detected intent
            confidence (float): Intent confidence score
            emotion (str): Detected emotion
            response (str): Chatbot's response
            response_time (float): Time taken to generate response (seconds)
            llm_source (str): Source of LLM response ('groq' or 'ollama')
        
        Returns:
            bool: True if logging successful, False otherwise
        """
        try:
            conn = self.get_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            timestamp = get_current_timestamp()
            
            cursor.execute("""
                INSERT INTO logs 
                (timestamp, user_input, intent, confidence, emotion, response, response_time, llm_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, user_input, intent, confidence, emotion, response, response_time, llm_source))
            
            conn.commit()
            conn.close()
            return True
        
        except sqlite3.Error as e:
            print(f"Error logging interaction: {e}")
            return False
    
    def get_logs(self, limit=50, intent_filter=None):
        """
        Retrieve chat logs.
        
        Args:
            limit (int): Maximum number of logs to retrieve
            intent_filter (str): Filter by specific intent (optional)
        
        Returns:
            list: List of log records
        """
        try:
            conn = self.get_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            
            if intent_filter:
                cursor.execute("""
                    SELECT * FROM logs 
                    WHERE intent = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (intent_filter, limit))
            else:
                cursor.execute("""
                    SELECT * FROM logs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        
        except sqlite3.Error as e:
            print(f"Error retrieving logs: {e}")
            return []
    
    def get_analytics_summary(self):
        """
        Get analytics summary of interactions.
        
        Returns:
            dict: Analytics summary including counts and statistics
        """
        try:
            conn = self.get_connection()
            if not conn:
                return {}
            
            cursor = conn.cursor()
            
            # Total interactions
            cursor.execute("SELECT COUNT(*) as count FROM logs")
            total_interactions = cursor.fetchone()['count']
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) as avg_conf FROM logs")
            avg_confidence = cursor.fetchone()['avg_conf'] or 0
            
            # Intent distribution
            cursor.execute("""
                SELECT intent, COUNT(*) as count 
                FROM logs 
                GROUP BY intent 
                ORDER BY count DESC 
                LIMIT 5
            """)
            top_intents = {row['intent']: row['count'] for row in cursor.fetchall()}
            
            # Emotion distribution
            cursor.execute("""
                SELECT emotion, COUNT(*) as count 
                FROM logs 
                GROUP BY emotion 
                ORDER BY count DESC
            """)
            emotions = {row['emotion']: row['count'] for row in cursor.fetchall()}
            
            # Average response time
            cursor.execute("SELECT AVG(response_time) as avg_time FROM logs")
            avg_response_time = cursor.fetchone()['avg_time'] or 0
            
            # LLM source distribution
            cursor.execute("""
                SELECT llm_source, COUNT(*) as count 
                FROM logs 
                GROUP BY llm_source
            """)
            llm_sources = {row['llm_source']: row['count'] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                "total_interactions": total_interactions,
                "average_confidence": round(avg_confidence, 3),
                "top_intents": top_intents,
                "emotion_distribution": emotions,
                "average_response_time": round(avg_response_time, 3),
                "llm_source_distribution": llm_sources
            }
        
        except sqlite3.Error as e:
            print(f"Error retrieving analytics: {e}")
            return {}
    
    def get_intent_count(self, intent):
        """
        Get count of interactions for a specific intent.
        
        Args:
            intent (str): Intent tag
        
        Returns:
            int: Number of interactions with this intent
        """
        try:
            conn = self.get_connection()
            if not conn:
                return 0
            
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM logs WHERE intent = ?", (intent,))
            count = cursor.fetchone()['count']
            conn.close()
            return count
        
        except sqlite3.Error as e:
            print(f"Error getting intent count: {e}")
            return 0
    
    def clear_logs(self, older_than_days=None):
        """
        Clear old logs from database.
        
        Args:
            older_than_days (int): Delete logs older than N days (None = delete all)
        
        Returns:
            int: Number of records deleted
        """
        try:
            conn = self.get_connection()
            if not conn:
                return 0
            
            cursor = conn.cursor()
            
            if older_than_days:
                cursor.execute("""
                    DELETE FROM logs 
                    WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
                """, (older_than_days,))
            else:
                cursor.execute("DELETE FROM logs")
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            return deleted_count
        
        except sqlite3.Error as e:
            print(f"Error clearing logs: {e}")
            return 0
    
    def export_logs_csv(self, output_path="chatbot_logs.csv"):
        """
        Export logs to CSV file.
        
        Args:
            output_path (str): Path to save CSV file
        
        Returns:
            bool: True if export successful
        """
        try:
            import csv
            
            logs = self.get_logs(limit=None)  # Get all logs
            
            if not logs:
                print("No logs to export")
                return False
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            
            print(f"Logs exported to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error exporting logs: {e}")
            return False
    
    def save_session(self, session_id: str, messages: list, metadata: dict = None) -> bool:
        """
        Save chat session to persistent storage.
        
        Args:
            session_id (str): Unique session identifier
            messages (list): Chat messages
            metadata (dict): Additional session metadata
        
        Returns:
            bool: True if successful
        """
        try:
            import json
            conn = self.get_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO chat_sessions 
                (session_id, created_at, updated_at, messages, total_turns, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                json.dumps(messages),
                len(messages),
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False
    
    def load_session(self, session_id: str) -> dict:
        """
        Load chat session from persistent storage.
        
        Args:
            session_id (str): Unique session identifier
        
        Returns:
            dict: Session data with messages, or empty dict if not found
        """
        try:
            import json
            conn = self.get_connection()
            if not conn:
                return {}
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT messages, metadata FROM chat_sessions 
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "messages": json.loads(row['messages']),
                    "metadata": json.loads(row['metadata'])
                }
            return {}
        except Exception as e:
            print(f"Error loading session: {e}")
            return {}
    
    def list_sessions(self, limit: int = 10) -> list:
        """
        List recent chat sessions.
        
        Args:
            limit (int): Number of sessions to retrieve
        
        Returns:
            list: Session summaries
        """
        try:
            conn = self.get_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, created_at, updated_at, total_turns 
                FROM chat_sessions 
                ORDER BY updated_at DESC 
                LIMIT ?
            """, (limit,))
            
            sessions = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return sessions
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []
