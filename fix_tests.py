#!/usr/bin/env python3
"""
Script to fix test file API mismatches
"""

import re
import os

def fix_test_file(filepath):
    """Fix a test file to match actual API."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Replace add_interaction with add_turn
    content = content.replace(
        'context.add_interaction(query, "response", "fees", 0.9)',
        'context.add_turn(user_input=query, bot_response="response", intent="fees", confidence=0.9, emotion="neutral")'
    )
    
    # Replace get_prompt_context with get_formatted_history  
    content = content.replace('context.get_prompt_context()', 'context.get_formatted_history()')
    
    # Fix generate_response calls - remove is_in_scope and conversation_context params
    # Pattern 1: , is_in_scope=...
    content = re.sub(r',\s*is_in_scope=[^,\n]+', '', content)
    # Pattern 2: , conversation_context=...
    content = re.sub(r',\s*conversation_context=[^,\n)]+', '', content)
    # Pattern 3: conversation_context= (at end of params)
    content = re.sub(r'\s*conversation_context=[^,\n)]+\)', '\n        )', content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

# Fix test files
files_to_fix = [
    'tests/test_new_features.py',
    'tests/test_scenarios_realistic.py',
    'tests/test_comprehensive_edge_cases.py',
    'tests/test_time_aware_comprehensive.py'
]

for filepath in files_to_fix:
    full_path = os.path.join('c:\\Users\\Joseph\\Desktop\\projects\\AAI_chatbot', filepath)
    if os.path.exists(full_path):
        if fix_test_file(full_path):
            print(f"✓ Fixed {filepath}")
        else:
            print(f"◯ No changes needed for {filepath}")
    else:
        print(f"✗ File not found: {filepath}")

print("\nTest files fixed!")
