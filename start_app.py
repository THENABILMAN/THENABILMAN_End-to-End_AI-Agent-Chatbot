#!/usr/bin/env python3
"""
Simple Windows batch runner for the AI Chatbot
Run both backend and frontend in separate terminal windows
"""

import subprocess
import sys
import os
from pathlib import Path

def run_in_new_terminal(command, title):
    """Run a command in a new PowerShell window"""
    # PowerShell command to open new window and run command
    ps_command = f'''
    Start-Process powershell -ArgumentList @(
        '-NoExit',
        '-Command',
        'cd "{Path(__file__).parent}"; {command}'
    ) -WindowStyle Normal
    '''
    
    subprocess.Popen(
        ['powershell', '-Command', ps_command],
        cwd=Path(__file__).parent
    )

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting THENABILMAN AI Chatbot Agents")
    print("=" * 60)
    print("\n📋 Launching backend and frontend in separate windows...\n")
    
    try:
        # Start backend in new terminal
        run_in_new_terminal(
            'python combined_app.py --backend',
            'Backend Server'
        )
        print("✅ Backend terminal opened (port 9999)")
        
        # Wait a moment for backend to start
        import time
        time.sleep(3)
        
        # Start frontend in new terminal
        run_in_new_terminal(
            'streamlit run combined_app.py',
            'Streamlit Frontend'
        )
        print("✅ Frontend terminal opened (port 8501)")
        
        print("\n" + "=" * 60)
        print("✨ Both services are running!")
        print("=" * 60)
        print("\n📌 Backend URL: http://127.0.0.1:9999")
        print("📌 Frontend URL: http://localhost:8501")
        print("\nYou can now use the chatbot in the frontend window.")
        print("Close any window to stop that service.\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
