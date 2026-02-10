#!/usr/bin/env python3
"""PerryPicks v3 - Automation Startup Script.

One-stop script to start the complete automation system:
- Install/check dependencies
- Start backend automation (CLI scheduler)
- Start frontend GUI (Streamlit)

Usage:
    python start_automation.py [--port 8501] [--backend-only] [--frontend-only]
"""

from __future__ import annotations
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.absolute()
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_AUTOMATION = PROJECT_ROOT / "requirements-automation.txt"

# Global process tracking
backend_process: Optional[subprocess.Popen] = None
frontend_process: Optional[subprocess.Popen] = None


def get_python_command() -> Tuple[str, List[str]]:
    """Get Python command (uv or system Python)."""
    if which("uv") is not None:
        return "uv", ["uv", "run", "python"]
    else:
        return "python", [sys.executable]

def check_and_install_dependencies() -> bool:
    """Check if dependencies are installed, install if needed."""
    logger.info("Checking dependencies...")
    
    python_cmd_type, python_cmd = get_python_command()
    
    # Install command
    if python_cmd_type == "uv":
        pip_cmd = ["uv", "pip", "install", "-q"]
    else:
        pip_cmd = [sys.executable, "-m", "pip", "install", "-q"]
    
    required_packages = [
        "streamlit",
        "tweepy",
        "atproto",
        "schedule",
    ]
    
    missing_packages = []
    for package in required_packages:
        if not is_package_installed(package):
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        logger.info("Installing dependencies...")
        
        # Install from requirements files (gracefully - don't fail if one file has issues)
        requirements_files = []
        if REQUIREMENTS_AUTOMATION.exists():
            requirements_files.append(REQUIREMENTS_AUTOMATION)
        if REQUIREMENTS_FILE.exists():
            requirements_files.append(REQUIREMENTS_FILE)
        
        for req_file in requirements_files:
            try:
                cmd = pip_cmd + ["-r", str(req_file)]
                logger.info(f"Installing from {req_file}...")
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"✅ Installed from {req_file}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to install from {req_file}: {e}")
                logger.warning("   Continuing with individual package installation...")
                # Don't return False - continue to try installing packages individually
        
        # Check if packages are now installed (they might be from requirements files)
        still_missing = []
        for package in missing_packages:
            if not is_package_installed(package):
                still_missing.append(package)
        
        # Install any remaining missing packages individually
        if still_missing:
            cmd = pip_cmd + still_missing
            logger.info(f"Installing remaining packages: {still_missing}")
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info("✅ All dependencies installed")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install packages: {e}")
                return False
        else:
            logger.info("✅ All dependencies are now installed")
    else:
        logger.info("✅ All dependencies are already installed")
    
    return True

def is_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed."""
    python_cmd_type, python_cmd = get_python_command()
    
    try:
        if python_cmd_type == "uv":
            cmd = python_cmd + ["-c", f"import {package_name}"]
            subprocess.run(cmd, check=True, capture_output=True, timeout=5)
        else:
            __import__(package_name)
        return True
    except (ImportError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False

def which(cmd: str) -> Optional[str]:
    """Find executable in PATH."""
    for path in os.environ.get("PATH", "").split(os.pathsep):
        cmd_path = Path(path) / cmd
        if cmd_path.exists() and os.access(cmd_path, os.X_OK):
            return str(cmd_path)
    return None

def start_backend(
    poll_interval: int = 15,
    dry_run: bool = False,
) -> Optional[subprocess.Popen]:
    """Start backend automation (CLI scheduler)."""
    
    python_cmd_type, python_cmd = get_python_command()
    
    # Build command
    if python_cmd_type == "uv":
        cmd = ["uv", "run", "python", "-m", "src.automation.game_state_service"]
    else:
        cmd = python_cmd + ["-m", "src.automation.game_state_service"]

    os.environ["GAME_STATE_POLL_INTERVAL"] = str(int(poll_interval) * 60)
    os.environ["GAME_STATE_DRY_RUN"] = "true" if dry_run else "false"
    
    logger.info(f"Starting backend automation: {' '.join(cmd)}")
    
    # Start process
    try:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=False,  # Don't create new session (helps with signal handling)
        )
        logger.info("✅ Backend automation started")
        return process
    except Exception as e:
        logger.error(f"❌ Failed to start backend: {e}")
        return None

def start_frontend(
    port: int = 8501,
    headless: bool = False,
) -> Optional[subprocess.Popen]:
    """Start frontend GUI (Streamlit)."""
    app_path = PROJECT_ROOT / "pages" / "04_Automation_Manager.py"
    
    if not app_path.exists():
        logger.error(f"❌ Frontend app not found: {app_path}")
        return None
    
    # Check if streamlit is installed
    if not is_package_installed("streamlit"):
        logger.error("❌ Streamlit is not installed")
        return None
    
    # Build command
    python_cmd_type, python_cmd = get_python_command()
    
    if python_cmd_type == "uv":
        cmd = ["uv", "run", "streamlit", "run", str(app_path)]
    else:
        cmd = python_cmd + ["-m", "streamlit", "run", str(app_path)]
    
    if headless:
        cmd.extend(["--server.headless", "true"])
    
    cmd.extend(["--server.port", str(port)])
    
    logger.info(f"Starting frontend GUI: {' '.join(cmd)}")
    
    # Start process
    try:
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        logger.info(f"✅ Frontend GUI started on http://localhost:{port}")
        return process
    except Exception as e:
        logger.error(f"❌ Failed to start frontend: {e}")
        return None

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global backend_process, frontend_process
    
    logger.info(f"\nReceived signal {signum}, shutting down...")
    
    # Stop frontend
    if frontend_process:
        logger.info("Stopping frontend GUI...")
        frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
    
    # Stop backend
    if backend_process:
        logger.info("Stopping backend automation...")
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    logger.info("✅ Shutdown complete")
    sys.exit(0)

def print_status():
    """Print current status."""
    print("\n" + "=" * 60)
    print("PerryPicks v3 - Automation System")
    print("=" * 60)
    print()
    print("Status:")
    print(f"  Backend: {'✅ Running' if backend_process else '❌ Not running'}")
    print(f"  Frontend: {'✅ Running' if frontend_process else '❌ Not running'}")
    print()
    if frontend_process:
        print(f"  Frontend URL: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

def wait_for_startup(processes: List[subprocess.Popen], timeout: int = 10):
    """Wait for processes to start."""
    logger.info("Waiting for services to start...")
    
    for i in range(timeout):
        time.sleep(1)
        
        # Check if any process has exited
        for i, proc in enumerate(processes):
            if proc.poll() is not None:
                logger.error(f"❌ Process {i+1} exited unexpectedly")
                return False
    
    return True

def main():
    """Main function."""
    global backend_process, frontend_process
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="PerryPicks v3 - Automation Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit GUI (default: 8501)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Backend poll interval in minutes (default: 15)",
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only backend automation",
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Start only frontend GUI",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run backend in dry-run mode",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run frontend in headless mode",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Skip dependency check",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print()
    print("=" * 60)
    print("""
   ╔═════════════════════════════════════════════════════════════╗
   ║                                                               ║
   ║    🤖 PerryPicks v3 - Automation System 🤖                  ║
   ║                                                               ║
   ║    Complete social media automation for NBA predictions            ║
   ║                                                               ║
   ╚═════════════════════════════════════════════════════════════╝
    """)
    print("=" * 60)
    print()
    
    # Check dependencies
    if not args.no_deps:
        if not check_and_install_dependencies():
            logger.error("❌ Failed to install dependencies")
            sys.exit(1)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start backend (if not frontend-only)
    if not args.frontend_only:
        backend_process = start_backend(
            poll_interval=args.poll_interval,
            dry_run=args.dry_run,
        )
        if not backend_process:
            logger.error("❌ Failed to start backend")
            sys.exit(1)
    
    # Start frontend (if not backend-only)
    if not args.backend_only:
        frontend_process = start_frontend(
            port=args.port,
            headless=args.headless,
        )
        if not frontend_process:
            logger.error("❌ Failed to start frontend")
            if backend_process:
                backend_process.terminate()
            sys.exit(1)
    
    # Wait for startup
    processes = [p for p in [backend_process, frontend_process] if p is not None]
    if not wait_for_startup(processes):
        logger.error("❌ Services failed to start")
        sys.exit(1)
    
    # Print status
    print_status()
    
    # Monitor processes
    try:
        while True:
            time.sleep(1)
            
            # Check backend
            if backend_process and backend_process.poll() is not None:
                logger.error("❌ Backend automation stopped unexpectedly")
                break
            
            # Check frontend
            if frontend_process and frontend_process.poll() is not None:
                logger.error("❌ Frontend GUI stopped unexpectedly")
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        signal_handler(signal.SIGTERM, None)

if __name__ == "__main__":
    main()
