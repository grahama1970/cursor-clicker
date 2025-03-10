"""
Utility functions for cursor_clicker.

This module contains utility functions that help with image processing, model interaction,
and other common tasks needed by the main application.
"""

import os
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def image_to_tensor(image_path):
    """
    Convert an image to a tensor for model input.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        torch.Tensor: Processed image tensor or None if failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        # Open and convert image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Simple preprocessing - actual transformations would depend on the model
        # In a full implementation, we'd apply the model's image processor here
        return image
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        return None

def format_inference_stats(inference_start_time, inference_end_time, output_size):
    """
    Format inference statistics for logging.
    
    Args:
        inference_start_time (float): Start time of inference
        inference_end_time (float): End time of inference
        output_size (int): Size of the output in tokens or characters
        
    Returns:
        str: Formatted statistics string
    """
    inference_time = inference_end_time - inference_start_time
    tokens_per_second = output_size / inference_time if inference_time > 0 else 0
    
    return (
        f"Inference completed in {inference_time:.2f} seconds | "
        f"Output size: {output_size} chars | "
        f"Speed: {tokens_per_second:.2f} tokens/sec"
    )

def get_device_info():
    """
    Get detailed information about the compute device.
    
    Returns:
        dict: Device information including type, name, and memory
    """
    device_info = {
        "type": "cuda" if torch.cuda.is_available() else "cpu",
        "name": "CPU",
        "memory": "N/A"
    }
    
    if device_info["type"] == "cuda":
        try:
            device_info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            device_info["memory"] = f"{props.total_memory / 1e9:.2f} GB"
            device_info["compute_capability"] = f"{props.major}.{props.minor}"
        except Exception as e:
            logger.error(f"Error getting CUDA device info: {e}")
    
    return device_info

def log_device_info():
    """
    Log information about the compute device.
    """
    device_info = get_device_info()
    logger.info(f"Using device: {device_info['type']}")
    logger.info(f"Device name: {device_info['name']}")
    logger.info(f"Device memory: {device_info['memory']}")
    
    if device_info["type"] == "cuda":
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Compute capability: {device_info.get('compute_capability', 'unknown')}")
    
    logger.info(f"PyTorch version: {torch.__version__}")

def safe_click(x, y, message="", delay=0.5):
    """
    Perform a safe mouse click with error handling.
    
    Args:
        x (int): X coordinate
        y (int): Y coordinate
        message (str): Optional message to log
        delay (float): Delay after click in seconds
        
    Returns:
        bool: True if click was successful
    """
    try:
        import pyautogui
        
        # Log what we're clicking on if a message was provided
        if message:
            logger.info(f"Clicking {message} at ({x}, {y})")
        else:
            logger.info(f"Clicking at position ({x}, {y})")
        
        # Validate coordinates
        screen_width, screen_height = pyautogui.size()
        if x < 0 or x >= screen_width or y < 0 or y >= screen_height:
            logger.error(f"Click coordinates ({x}, {y}) out of screen bounds ({screen_width}x{screen_height})")
            return False
        
        # Perform the click
        pyautogui.click(x, y)
        
        # Wait after clicking
        if delay > 0:
            import time
            time.sleep(delay)
            
        return True
    except Exception as e:
        logger.error(f"Click operation failed: {e}")
        return False 