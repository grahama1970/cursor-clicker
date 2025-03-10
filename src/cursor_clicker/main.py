"""
Cursor Clicker - An automated tool for handling Cursor.sh tool call limits.

This script uses visual recognition with Qwen2.5-VL-3B-Instruct model to:
1. Capture screenshots of the Cursor.sh window
2. Identify "We've hit our tool call limit" messages
3. Automate the clicking of the "Continue" button
4. Pause for a configured time to allow rate limits to reset
"""

import os
import time
import logging
import datetime
import pyautogui
import numpy as np
import torch
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoModelForVision2Seq
from dotenv import load_dotenv
import json
from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, Field, ValidationError

from cursor_clicker.screenshot_manager import (
    capture_and_save, 
    capture_cursor_window,
    get_latest_screenshot
)
from cursor_clicker.models import ButtonLocation, ScreenAnalysisResult

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
SCREENSHOT_INTERVAL_SECONDS = 5
PAUSE_DURATION_MINUTES = 2
AGENT_TOOLS_CALL_THRESHOLD = 5  # Positive detection after this many consecutive tool calls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cursor_clicker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# OCR Functions
# =============================================================================

def perform_ocr(image_path):
    """
    Perform OCR on an image to extract text.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        logger.info(f"OCR completed, extracted {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""

def contains_tool_call_limit_text(text):
    """
    Check if the text contains indicators of hitting the tool call limit.
    
    Args:
        text (str): Text from OCR
        
    Returns:
        bool: True if tool call limit message detected
    """
    markers = [
        "hit our tool call limit",
        "we've hit our tool call limit", 
        "We've hit our tool call limit",
        "daily limit for tool calls",
        "reached our limit"
    ]
    
    for marker in markers:
        if marker in text:
            logger.info(f"Tool call limit text detected: '{marker}'")
            return True
    
    return False

# =============================================================================
# ML Model Functions
# =============================================================================

def setup_model():
    """
    Setup the Qwen2.5-VL model for visual processing.
    
    Returns:
        tuple: (tokenizer, model) or (None, None) if setup fails
    """
    try:
        logger.info(f"Setting up model: {MODEL_NAME}")
        
        # Get Hugging Face token from environment
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        # Log device info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        if device == "cuda":
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        start_time = time.time()
        
        # Load tokenizer with token
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        
        # Load model with token
        logger.info("Loading model...")
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME, 
            token=hf_token,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        
        # Log model loading time
        logger.info(f"Model setup complete in {time.time() - start_time:.2f} seconds")
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Model setup failed: {e}")
        return None, None

def process_image_with_model(image_path, tokenizer, model):
    """
    Process an image with the Qwen model to detect errors in Cursor.sh.
    
    Args:
        image_path (str): Path to the image
        tokenizer: Model tokenizer
        model: Qwen model
        
    Returns:
        ScreenAnalysisResult or None: Analysis result or None if no errors detected
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Define schema as a Python dictionary
        schema = {
            "error_type": {
                "type": "string", 
                "enum": ["none", "tool_call_limit", "anthropic_unavailable"],
                "description": "Type of error detected in the screenshot"
            },
            "message": {
                "type": "string",
                "description": "The exact error message text"
            },
            "buttons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "integer", "description": "Left coordinate of the button"},
                        "y1": {"type": "integer", "description": "Top coordinate of the button"},
                        "x2": {"type": "integer", "description": "Right coordinate of the button"},
                        "y2": {"type": "integer", "description": "Bottom coordinate of the button"},
                        "text": {"type": "string", "description": "Text on the button"}
                    },
                    "required": ["x1", "y1", "x2", "y2", "text"]
                },
                "description": "Locations of action buttons like 'Try Again' or 'Continue'"
            }
        }
        
        # Format schema as JSON
        schema_json = json.dumps(schema, indent=2)
        
        # Define prompt for error detection
        query = (
            "Analyze this screenshot from Cursor.sh. "
            "Look for either a 'tool call limit' message or an 'Unable to reach anthropic' error. "
            f"Respond with Well-formatted JSON that follows this schema:\n{schema_json}\n\n"
            "If you see a 'Try Again' or 'Continue' button, include its precise coordinates. "
            "The coordinates should be relative to the screenshot (x1,y1 is top-left, x2,y2 is bottom-right)."
        )
        
        # Prepare inputs for the model
        start_time = time.time()
        inputs = tokenizer.build_conversation_input_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": image_path}}
                ]}],
                return_dict=True,
                add_generation_prompt=True
            )
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"].unsqueeze(0).to(model.device),
                attention_mask=inputs["attention_mask"].unsqueeze(0).to(model.device),
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[0]:], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        # Log response and detection
        logger.info(f"Model response: {response}")
        logger.info(f"Inference time: {inference_time:.2f} seconds")
        
        try:
            # Parse the response as JSON
            result_json = json.loads(response)
            # Validate against our Pydantic model
            analysis = ScreenAnalysisResult(**result_json)
            
            if analysis.error_type != "none":
                logger.info(f"Detected error: {analysis.error_type}")
                logger.info(f"Error message: {analysis.message}")
                
                if analysis.buttons:
                    for button in analysis.buttons:
                        logger.info(f"Found button '{button.text}' at coordinates: {button.x1},{button.y1},{button.x2},{button.y2}")
                
                return analysis
            
            return None
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse model response as JSON: {e}")
            # Fallback to basic text analysis if JSON parsing fails
            if "tool call limit" in response.lower():
                return ScreenAnalysisResult(error_type="tool_call_limit", message="Tool call limit detected")
            if "unable to reach anthropic" in response.lower():
                return ScreenAnalysisResult(error_type="anthropic_unavailable", message="Anthropic unavailable")
            return None
            
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return None

# =============================================================================
# Automation Functions
# =============================================================================

def find_and_click_continue_button(window_title="Cursor.sh"):
    """
    Find and click the "Continue" button in the Cursor window.
    
    Args:
        window_title (str): Title of the window to look for
        
    Returns:
        bool: True if successful
    """
    try:
        # Capture window
        logger.info("Capturing window for Continue button detection")
        screenshot = capture_cursor_window(window_title)
        
        if screenshot is None:
            logger.error("Failed to capture window for Continue button detection")
            return False
        
        # Define regions to search for the button (middle-bottom of window)
        width, height = screenshot.size
        search_regions = [
            # Bottom center
            (width // 3, height * 2 // 3, width * 2 // 3, height),
            # Center
            (width // 4, height // 3, width * 3 // 4, height * 2 // 3),
            # Entire window as fallback
            (0, 0, width, height)
        ]
        
        # Try to locate "Continue" button in each region
        for region in search_regions:
            logger.info(f"Searching for Continue button in region {region}")
            
            # Extract region
            region_img = screenshot.crop(region)
            
            # Try OCR on region
            region_text = pytesseract.image_to_string(region_img)
            
            if "continue" in region_text.lower():
                logger.info("Found 'Continue' text in region")
                
                # Calculate click position (center of region)
                click_x = region[0] + (region[2] - region[0]) // 2
                click_y = region[1] + (region[3] - region[1]) // 2
                
                # Convert to screen coordinates
                windows = pyautogui.getWindowsWithTitle(window_title)
                if not windows:
                    logger.error("Window not found for clicking")
                    return False
                
                window = windows[0]
                screen_x = window.left + click_x
                screen_y = window.top + click_y
                
                # Perform click
                logger.info(f"Clicking at position ({screen_x}, {screen_y})")
                pyautogui.click(screen_x, screen_y)
                return True
        
        logger.warning("Continue button not found in any region")
        return False
        
    except Exception as e:
        logger.error(f"Error finding Continue button: {e}")
        return False

def pause_for_reset():
    """
    Pause execution to allow tool call limits to reset.
    """
    pause_seconds = PAUSE_DURATION_MINUTES * 60
    logger.info(f"Pausing for {PAUSE_DURATION_MINUTES} minutes to allow rate limits to reset")
    
    start_time = time.time()
    end_time = start_time + pause_seconds
    
    try:
        while time.time() < end_time:
            remaining_seconds = int(end_time - time.time())
            remaining_minutes = remaining_seconds // 60
            remaining_secs = remaining_seconds % 60
            
            logger.info(f"Waiting: {remaining_minutes}m {remaining_secs}s remaining")
            time.sleep(min(30, remaining_seconds))  # Update status every 30 seconds
    except KeyboardInterrupt:
        logger.info("Pause interrupted by user")
    
    logger.info(f"Resuming after {time.time() - start_time:.1f} seconds")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main execution function.
    """
    logger.info("=" * 80)
    logger.info("Starting Cursor Clicker")
    logger.info("=" * 80)
    
    # Initial setup
    tokenizer, model = setup_model()
    if tokenizer is None or model is None:
        logger.error("Failed to set up model, exiting")
        return
    
    # Initialize tracking variables
    consecutive_tool_calls = 0
    
    try:
        while True:
            # Capture screenshot
            logger.info(f"Capturing screenshot (Interval: {SCREENSHOT_INTERVAL_SECONDS}s)")
            screenshot_path = capture_and_save(compress=True, use_timestamp=False)
            
            if screenshot_path is None:
                logger.error("Failed to capture screenshot, retrying...")
                time.sleep(SCREENSHOT_INTERVAL_SECONDS)
                continue
            
            # Process with model for accurate detection
            analysis = process_image_with_model(screenshot_path, tokenizer, model)
            
            if analysis is not None:
                # Handle different error types
                if analysis.error_type == "tool_call_limit":
                    consecutive_tool_calls += 1
                    logger.info(f"Possible tool call limit detected ({consecutive_tool_calls}/{AGENT_TOOLS_CALL_THRESHOLD})")
                    
                    # Require multiple consecutive detections to avoid false positives
                    if consecutive_tool_calls >= AGENT_TOOLS_CALL_THRESHOLD:
                        logger.info("Tool call limit confirmed, attempting to click Continue button")
                        
                        # Look for a continue button in the analysis
                        continue_buttons = [b for b in analysis.buttons if "continue" in b.text.lower()]
                        
                        if continue_buttons:
                            button = continue_buttons[0]
                            # Calculate center of button
                            center_x = (button.x1 + button.x2) // 2
                            center_y = (button.y1 + button.y2) // 2
                            
                            # Get the window to click in
                            windows = pyautogui.getWindowsWithTitle("Cursor.sh")
                            if not windows:
                                logger.error("Cursor.sh window not found")
                                time.sleep(SCREENSHOT_INTERVAL_SECONDS)
                                continue
                            
                            window = windows[0]
                            # Convert to screen coordinates
                            screen_x = window.left + center_x
                            screen_y = window.top + center_y
                            
                            # Click the button
                            logger.info(f"Clicking Continue button at ({screen_x}, {screen_y})")
                            pyautogui.click(screen_x, screen_y)
                            consecutive_tool_calls = 0
                            
                            # Pause to allow rate limits to reset
                            pause_for_reset()
                        else:
                            # Fallback to the old method if button not found in analysis
                            clicked = find_and_click_continue_button()
                            if clicked:
                                logger.info("Successfully clicked Continue button")
                                consecutive_tool_calls = 0
                                pause_for_reset()
                            else:
                                logger.warning("Failed to click Continue button")
                                time.sleep(SCREENSHOT_INTERVAL_SECONDS)
                    else:
                        time.sleep(SCREENSHOT_INTERVAL_SECONDS)
                
                elif analysis.error_type == "anthropic_unavailable":
                    logger.info("Detected 'Unable to reach anthropic' error")
                    
                    # Find the "Try Again" button
                    try_again_buttons = [b for b in analysis.buttons if "try again" in b.text.lower()]
                    
                    if try_again_buttons:
                        # Wait 5 minutes before clicking Try Again
                        wait_minutes = 5
                        logger.info(f"Waiting {wait_minutes} minutes before clicking Try Again button")
                        
                        # Wait period
                        wait_seconds = wait_minutes * 60
                        start_time = time.time()
                        end_time = start_time + wait_seconds
                        
                        while time.time() < end_time:
                            remaining_seconds = int(end_time - time.time())
                            remaining_minutes = remaining_seconds // 60
                            remaining_secs = remaining_seconds % 60
                            
                            logger.info(f"Anthropic error: Waiting {remaining_minutes}m {remaining_secs}s before retry")
                            time.sleep(min(30, remaining_seconds))  # Update status every 30 seconds
                        
                        # After waiting, click the Try Again button
                        button = try_again_buttons[0]
                        center_x = (button.x1 + button.x2) // 2
                        center_y = (button.y1 + button.y2) // 2
                        
                        windows = pyautogui.getWindowsWithTitle("Cursor.sh")
                        if not windows:
                            logger.error("Cursor.sh window not found")
                            time.sleep(SCREENSHOT_INTERVAL_SECONDS)
                            continue
                        
                        window = windows[0]
                        screen_x = window.left + center_x
                        screen_y = window.top + center_y
                        
                        logger.info(f"Clicking Try Again button at ({screen_x}, {screen_y})")
                        pyautogui.click(screen_x, screen_y)
                    else:
                        logger.warning("Try Again button not found in analysis")
                        time.sleep(SCREENSHOT_INTERVAL_SECONDS)
            else:
                # Reset consecutive detection counter if no error detected
                if consecutive_tool_calls > 0:
                    logger.info(f"No tool call limit detected, resetting counter from {consecutive_tool_calls} to 0")
                    consecutive_tool_calls = 0
                else:
                    logger.info("No errors detected")
                
                time.sleep(SCREENSHOT_INTERVAL_SECONDS)
                
    except KeyboardInterrupt:
        logger.info("Cursor Clicker stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        logger.info("Cursor Clicker shutting down")

if __name__ == "__main__":
    main() 