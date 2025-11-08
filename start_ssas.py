#!/usr/bin/env python3
"""
SSAS Startup Script
Starts both backend and frontend servers for the Smart Student Analytics System.
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def check_port(port):
    """Check if a port is available."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def start_backend():
    """Start the Django backend server."""
    print("🚀 Starting SSAS Backend Server...")
    
    if check_port(8000):
        print("⚠️  Port 8000 is already in use. Backend server may already be running.")
        return None
    
    backend_dir = Path(__file__).parent / "backend"
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return None
    
    try:
        process = subprocess.Popen(
            ["python", "manage.py", "runserver", "0.0.0.0:8000"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✅ Backend server started on http://localhost:8000")
        return process
    except Exception as e:
        print(f"❌ Failed to start backend server: {e}")
        return None

def start_frontend():
    """Start the frontend HTTP server."""
    print("🌐 Starting SSAS Frontend Server...")
    
    if check_port(3000):
        print("⚠️  Port 3000 is already in use. Frontend server may already be running.")
        return None
    
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return None
    
    try:
        process = subprocess.Popen(
            ["python3", "-m", "http.server", "3000"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✅ Frontend server started on http://localhost:3000")
        return process
    except Exception as e:
        print(f"❌ Failed to start frontend server: {e}")
        return None

def main():
    """Main startup function."""
    print("🎓 Smart Student Analytics System (SSAS)")
    print("=" * 50)
    
    # Start backend
    backend_process = start_backend()
    if backend_process:
        time.sleep(3)  # Wait for backend to start
    
    # Start frontend
    frontend_process = start_frontend()
    if frontend_process:
        time.sleep(2)  # Wait for frontend to start
    
    print("\n🎉 SSAS is now running!")
    print("=" * 50)
    print("📊 Backend API: http://localhost:8000")
    print("🌐 Frontend UI: http://localhost:3000")
    print("📚 API Docs: http://localhost:8000/api/docs/")
    print("\n💡 To stop the servers, press Ctrl+C")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down SSAS...")
        
        if backend_process:
            backend_process.terminate()
            print("✅ Backend server stopped")
        
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend server stopped")
        
        print("👋 SSAS shutdown complete!")

if __name__ == "__main__":
    main()
