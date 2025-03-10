"""
Command-line interface for cursor_clicker.

This module provides entry points for the cursor_clicker package.
"""

import sys
import argparse
import logging

from cursor_clicker.main import main as main_app
from cursor_clicker.continuous_monitor import main as monitor_app

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cursor Clicker - Automatic tool call limit handler"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Main application
    main_parser = subparsers.add_parser(
        "run", help="Run the main Cursor Clicker application"
    )
    
    # Monitor application
    monitor_parser = subparsers.add_parser(
        "monitor", help="Run continuous screenshot monitoring without ML model"
    )
    monitor_parser.add_argument(
        "--interval", type=int, default=5, 
        help="Seconds between screenshots"
    )
    monitor_parser.add_argument(
        "--timestamp", action="store_true", 
        help="Use timestamp in filenames instead of overwriting"
    )
    monitor_parser.add_argument(
        "--no-compress", action="store_true", 
        help="Disable compression"
    )
    monitor_parser.add_argument(
        "--width", type=int, default=800, 
        help="Maximum width for compression"
    )
    monitor_parser.add_argument(
        "--quality", type=int, default=75, 
        help="JPEG quality (1-100)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    try:
        if args.command == "run":
            main_app()
        elif args.command == "monitor":
            monitor_app()
        else:
            # Default to main app if no command specified
            main_app()
    except Exception as e:
        logger.exception(f"Error in CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 