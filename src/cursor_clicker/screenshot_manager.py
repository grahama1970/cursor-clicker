"""
Screenshot manager for cursor_clicker.
Handles capturing, storing, and managing screenshots.

Official docs:
- PyAutoGUI: https://pyautogui.readthedocs.io/
- PyGetWindow: https://pygetwindow.readthedocs.io/
- PIL: https://pillow.readthedocs.io/
"""

import os
import time
import datetime
import pyautogui
import pygetwindow as gw
from PIL import Image
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SCREENSHOT_DIR = "screenshots"
LATEST_SCREENSHOT_NAME = "latest_screenshot.png"

def ensure_dir_exists(directory):
    """Ensure the directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def get_timestamp():
    """Get a timestamp for filename use."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def capture_cursor_window(window_title="Cursor.sh", activation_timeout=2.0):
    """
    Capture a screenshot of the Cursor.sh window.
    
    Args:
        window_title (str): The title of the window to capture
        activation_timeout (float): Timeout for window activation
        
    Returns:
        PIL.Image or None: The captured screenshot, or None if failed
    """
    try:
        windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            logger.warning(f"Window with title '{window_title}' not found")
            return None

        window = windows[0]
        
        # Try to activate window with timeout
        activation_start = time.time()
        while time.time() - activation_start < activation_timeout:
            try:
                window.activate()
                break
            except Exception as e:
                logger.warning(f"Window activation attempt failed: {e}")
                time.sleep(0.1)
        
        # Verify window is active
        if not window.isActive:
            logger.warning("Failed to activate window")
            return None

        time.sleep(0.3)  # Wait for window to settle

        try:
            left, top = window.left, window.top
            width, height = window.width, window.height
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            return screenshot
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    except Exception as e:
        logger.error(f"Window handling error: {e}")
        return None

def save_screenshot(screenshot, directory=DEFAULT_SCREENSHOT_DIR, use_timestamp=False, 
                   compress=True, max_width=800, quality=75):
    """
    Save a screenshot to the specified directory.
    
    Args:
        screenshot (PIL.Image): The screenshot to save
        directory (str): Directory to save the screenshot
        use_timestamp (bool): Whether to use timestamp in filename
        compress (bool): Whether to compress the screenshot
        max_width (int): Maximum width for compression
        quality (int): JPEG quality for compression
        
    Returns:
        str: Path to the saved screenshot
    """
    ensure_dir_exists(directory)
    
    # Determine filename
    if use_timestamp:
        timestamp = get_timestamp()
        original_filename = f"screenshot_{timestamp}.png"
        compressed_filename = f"screenshot_{timestamp}_compressed.jpg"
    else:
        original_filename = LATEST_SCREENSHOT_NAME
        compressed_filename = "latest_screenshot_compressed.jpg"
    
    original_path = os.path.join(directory, original_filename)
    compressed_path = os.path.join(directory, compressed_filename)
    
    # Save original screenshot
    if not compress:
        screenshot.save(original_path)
        logger.info(f"Screenshot saved to {original_path}")
        return original_path
    
    # Save original before compression for record keeping
    screenshot.save(original_path)
    
    # Compress the screenshot
    start_time = time.time()
    
    # Convert RGBA to RGB for JPEG compatibility
    if screenshot.mode == 'RGBA':
        screenshot = screenshot.convert('RGB')
    
    # Resize if needed
    if screenshot.width > max_width:
        width, height = screenshot.size
        new_width = max_width
        new_height = int(height * (new_width / width))
        screenshot = screenshot.resize((new_width, new_height), Image.LANCZOS)
    
    # Save compressed version
    screenshot.save(compressed_path, "JPEG", quality=quality)
    
    # Log compression statistics
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    logger.info(f"Screenshot saved to {original_path}")
    logger.info(f"Compressed version saved to {compressed_path}")
    logger.info(f"Original size: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    logger.info(f"Compressed size: {compressed_size:,} bytes ({compressed_size/1024:.1f} KB)")
    logger.info(f"Compression ratio: {ratio:.2f}x")
    logger.info(f"Compression completed in {time.time() - start_time:.2f} seconds")
    
    return compressed_path

def capture_and_save(directory=DEFAULT_SCREENSHOT_DIR, use_timestamp=False, 
                    compress=True, max_width=800, quality=75):
    """
    Capture a screenshot and save it.
    
    Args:
        directory (str): Directory to save the screenshot
        use_timestamp (bool): Whether to use timestamp in filename
        compress (bool): Whether to compress the screenshot
        max_width (int): Maximum width for compression
        quality (int): JPEG quality for compression
        
    Returns:
        str or None: Path to the saved screenshot, or None if capture failed
    """
    logger.info("Capturing screenshot of Cursor window...")
    screenshot = capture_cursor_window()
    
    if screenshot is None:
        logger.error("Failed to capture screenshot")
        return None
    
    logger.info(f"Screenshot captured successfully ({screenshot.width}x{screenshot.height})")
    return save_screenshot(
        screenshot, 
        directory=directory, 
        use_timestamp=use_timestamp,
        compress=compress,
        max_width=max_width,
        quality=quality
    )

def get_latest_screenshot(directory=DEFAULT_SCREENSHOT_DIR, compressed=True):
    """
    Get the path to the latest screenshot.
    
    Args:
        directory (str): Directory where screenshots are saved
        compressed (bool): Whether to get the compressed version
        
    Returns:
        str or None: Path to the latest screenshot, or None if not found
    """
    if compressed:
        path = os.path.join(directory, "latest_screenshot_compressed.jpg")
    else:
        path = os.path.join(directory, LATEST_SCREENSHOT_NAME)
    
    if os.path.exists(path):
        return path
    return None

def get_most_recent_screenshot(directory=DEFAULT_SCREENSHOT_DIR, compressed=True):
    """
    Find the most recent screenshot by timestamp.
    
    Args:
        directory (str): Directory where screenshots are saved
        compressed (bool): Whether to look for compressed versions
        
    Returns:
        str or None: Path to the most recent screenshot, or None if none found
    """
    if not os.path.exists(directory):
        return None
    
    # Filter files by the pattern
    prefix = "screenshot_"
    suffix = "_compressed.jpg" if compressed else ".png"
    
    matching_files = [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    
    if not matching_files:
        return None
    
    # Sort by timestamp in filename
    matching_files.sort(reverse=True)
    return os.path.join(directory, matching_files[0]) 