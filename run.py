#!/usr/bin/env python3
"""
Run script for Deepfake Detection Web Server
This script starts the FastAPI backend server for the deepfake detection application.
"""

import os
import sys
import subprocess

def main():
    """Main function to run the web server"""
    print("=" * 60)
    print("🚀 STARTING DEEPFAKE DETECTION WEB SERVER")
    print("=" * 60)

    # Get the directory of this script (root directory)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "Backend")

    # Check if Backend directory exists
    if not os.path.exists(backend_dir):
        print(f"❌ Error: Backend directory not found at {backend_dir}")
        sys.exit(1)

    # Check if app.py exists
    app_file = os.path.join(backend_dir, "app.py")
    if not os.path.exists(app_file):
        print(f"❌ Error: app.py not found at {app_file}")
        sys.exit(1)

    print(f"📁 Root directory: {root_dir}")
    print(f"📁 Backend directory: {backend_dir}")
    print("🌐 Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print("-" * 60)


    try:
        # Change to Backend directory
        os.chdir(backend_dir)

        # Open the default web browser to the server URL
        import webbrowser
        webbrowser.open_new_tab("http://localhost:8000")

        # Use global python path explicitly
        global_python = r"C:\Program Files\Python311\python.exe"
        if os.path.exists(global_python):
            subprocess.run([
                global_python, "-m", "uvicorn",
                "app:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ], check=True)
        else:
            # Fallback to sys.executable
            subprocess.run([
                sys.executable, "-m", "uvicorn",
                "app:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()