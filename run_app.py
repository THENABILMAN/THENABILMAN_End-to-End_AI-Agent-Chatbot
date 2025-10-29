"""
Run both backend and frontend simultaneously using multiprocessing
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server"""
    print("🚀 Starting Backend Server...")
    subprocess.run(
        [sys.executable, "combined_app.py", "--backend"],
        cwd=Path(__file__).parent
    )

def run_frontend():
    """Run the Streamlit frontend"""
    print("🎨 Starting Streamlit Frontend...")
    time.sleep(3)  # Wait for backend to start
    subprocess.run(
        ["streamlit", "run", "combined_app.py"],
        cwd=Path(__file__).parent
    )

if __name__ == "__main__":
    from multiprocessing import Process
    
    # Create processes
    backend_process = Process(target=run_backend)
    frontend_process = Process(target=run_frontend)
    
    try:
        print("=" * 60)
        print("Starting THENABILMAN AI Chatbot Agents")
        print("=" * 60)
        print("\n✅ Backend will run on: http://127.0.0.1:9999")
        print("✅ Frontend will open on: http://localhost:8501\n")
        
        # Start both processes
        backend_process.start()
        frontend_process.start()
        
        # Wait for both processes
        backend_process.join()
        frontend_process.join()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping application...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.join()
        frontend_process.join()
        print("✅ Application stopped.")
