"""
Script to view and analyze chatbot database logs.
Run: python view_database.py
"""

import sqlite3
from datetime import datetime
import json


class DatabaseViewer:
    
    @staticmethod
    def format_table(headers, data):
        """Simple table formatter (replaces tabulate)"""
        if not data:
            return "No data to display"
        
        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Build table
        lines = []
        
        # Header separator
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        lines.append(sep)
        
        # Header row
        header_row = "|" + "|".join(f" {str(h):<{col_widths[i]}} " 
                                     for i, h in enumerate(headers)) + "|"
        lines.append(header_row)
        lines.append(sep)
        
        # Data rows
        for row in data:
            data_row = "|" + "|".join(f" {str(cell):<{col_widths[i]}} " 
                                       for i, cell in enumerate(row)) + "|"
            lines.append(data_row)
        
        lines.append(sep)
        return "\n".join(lines)
    
    def __init__(self, db_path="chatbot.db"):
        self.db_path = db_path
    
    def get_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            print(f"❌ Database connection error: {e}")
            return None
    
    def view_all_logs(self, limit=10):
        """View recent logs"""
        conn = self.get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, user_input, intent, confidence, emotion, 
                   response_time, llm_source, is_in_scope, scope_reason
            FROM logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            print("❌ No logs found in database")
            return
        
        print(f"\n📊 Latest {limit} Interactions:")
        print("=" * 150)
        
        data = []
        for row in rows:
            data.append([
                row['id'],
                row['timestamp'][-8:],  # Show only time
                row['user_input'][:40],
                row['intent'],
                f"{row['confidence']:.0%}",
                row['emotion'][:8],
                f"{row['response_time']:.2f}s",
                row['llm_source'],
                "✓" if row['is_in_scope'] else "✗"
            ])
        
        headers = ["ID", "Time", "User Input", "Intent", "Conf", "Emotion", "Time", "LLM", "Scope"]
        print(self.format_table(headers, data))
    
    def view_intent_distribution(self):
        """View intent classification statistics"""
        conn = self.get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT intent, COUNT(*) as count
            FROM logs
            GROUP BY intent
            ORDER BY count DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            print("❌ No intent data found")
            return
        
        print(f"\n🎯 Intent Distribution:")
        print("=" * 60)
        
        data = []
        total = sum(row['count'] for row in rows)
        
        for row in rows:
            pct = (row['count'] / total) * 100
            bar = "█" * int(pct / 2)
            data.append([row['intent'][:20], row['count'], f"{pct:.1f}%", bar])
        
        headers = ["Intent", "Count", "Percentage", "Distribution"]
        print(self.format_table(headers, data))
    
    def view_emotion_distribution(self):
        """View emotion detection statistics"""
        conn = self.get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM logs
            GROUP BY emotion
            ORDER BY count DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            print("❌ No emotion data found")
            return
        
        print(f"\n😊 Emotion Distribution:")
        print("=" * 60)
        
        data = []
        total = sum(row['count'] for row in rows)
        
        for row in rows:
            pct = (row['count'] / total) * 100
            data.append([row['emotion'], row['count'], f"{pct:.1f}%"])
        
        headers = ["Emotion", "Count", "Percentage"]
        print(self.format_table(headers, data))
    
    def view_llm_source_distribution(self):
        """View LLM source statistics"""
        conn = self.get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT llm_source, COUNT(*) as count
                FROM logs
                GROUP BY llm_source
                ORDER BY count DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                print("❌ No LLM source data found")
                return
            
            print(f"\n🤖 LLM Source Distribution:")
            print("=" * 60)
            
            data = []
            total = sum(row['count'] for row in rows)
            
            for row in rows:
                pct = (row['count'] / total) * 100
                data.append([row['llm_source'], row['count'], f"{pct:.1f}%"])
            
            headers = ["LLM Source", "Count", "Percentage"]
            print(self.format_table(headers, data))
        except sqlite3.OperationalError:
            print("⚠️ LLM source column not available in logs table")
    
    def view_scope_analysis(self):
        """View scope detection analysis"""
        conn = self.get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                CASE WHEN is_in_scope = 1 THEN 'In-Scope' ELSE 'Out-of-Scope' END as scope,
                COUNT(*) as count
            FROM logs
            GROUP BY is_in_scope
        """)
        
        rows = cursor.fetchall()
        cursor.execute("""
            SELECT scope_reason, COUNT(*) as count
            FROM logs
            WHERE scope_reason IS NOT NULL
            GROUP BY scope_reason
            ORDER BY count DESC
        """)
        
        scope_reasons = cursor.fetchall()
        conn.close()
        
        print(f"\n🎯 Scope Analysis:")
        print("=" * 60)
        
        data = []
        for row in rows:
            data.append([row['scope'], row['count']])
        
        headers = ["Scope Status", "Count"]
        print(self.format_table(headers, data))
        
        if scope_reasons:
            print("\nScope Reasons Breakdown:")
            data = []
            for row in scope_reasons:
                data.append([row['scope_reason'], row['count']])
            print(self.format_table(["Reason", "Count"], data))
    
    def view_confidence_stats(self):
        """View confidence score statistics"""
        conn = self.get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(confidence) as avg_conf,
                MIN(confidence) as min_conf,
                MAX(confidence) as max_conf,
                ROUND(CAST(COUNT(CASE WHEN confidence >= 0.8 THEN 1 END) AS FLOAT) / COUNT(*) * 100, 1) as high_conf_pct
            FROM logs
        """)
        
        stats = cursor.fetchone()
        conn.close()
        
        print(f"\n📈 Confidence Statistics:")
        print("=" * 60)
        
        data = [
            ["Total Interactions", stats['total']],
            ["Average Confidence", f"{stats['avg_conf']:.1%}"],
            ["Minimum Confidence", f"{stats['min_conf']:.1%}"],
            ["Maximum Confidence", f"{stats['max_conf']:.1%}"],
            ["High Confidence (≥80%)", f"{stats['high_conf_pct']}%"],
        ]
        
        print(self.format_table(["Metric", "Value"], data))
    
    def view_response_time_stats(self):
        """View response time statistics"""
        conn = self.get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                ROUND(AVG(response_time), 2) as avg_time,
                ROUND(MIN(response_time), 2) as min_time,
                ROUND(MAX(response_time), 2) as max_time
            FROM logs
        """)
        
        stats = cursor.fetchone()
        conn.close()
        
        print(f"\n⏱️ Response Time Statistics:")
        print("=" * 60)
        
        data = [
            ["Total Interactions", stats['total']],
            ["Average Response Time", f"{stats['avg_time']}s"],
            ["Minimum Response Time", f"{stats['min_time']}s"],
            ["Maximum Response Time", f"{stats['max_time']}s"],
        ]
        
        print(self.format_table(["Metric", "Value"], data))
    
    def get_table_stats(self):
        """Get database table statistics"""
        conn = self.get_connection()
        if not conn:
            return
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' ORDER BY name
        """)
        
        tables = cursor.fetchall()
        
        print(f"\n📁 Database Tables:")
        print("=" * 60)
        
        data = []
        for table in tables:
            table_name = table['name']
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            count = cursor.fetchone()['count']
            data.append([table_name, count])
        
        print(self.format_table(["Table Name", "Row Count"], data))
        conn.close()


def main():
    print("\n" + "=" * 150)
    print("🤖 CHATBOT DATABASE VIEWER")
    print("=" * 150)
    
    viewer = DatabaseViewer()
    
    # Show table stats
    viewer.get_table_stats()
    
    # Show recent logs
    viewer.view_all_logs(limit=15)
    
    # Show distribution analysis
    viewer.view_intent_distribution()
    viewer.view_emotion_distribution()
    viewer.view_llm_source_distribution()
    viewer.view_scope_analysis()
    
    # Show statistics
    viewer.view_confidence_stats()
    viewer.view_response_time_stats()
    
    print("\n" + "=" * 150)
    print("✅ Database view complete!")
    print("=" * 150 + "\n")


if __name__ == "__main__":
    main()
