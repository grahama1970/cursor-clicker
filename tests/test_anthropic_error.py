"""Tests for anthropic error detection and handling."""

import os
import sys
import json
import pytest
import time
from unittest.mock import patch, MagicMock, ANY
from PIL import Image

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cursor_clicker.main import process_image_with_model
from cursor_clicker.models import ScreenAnalysisResult, ButtonLocation

@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    # Configure the mock tokenizer and model
    mock_tokenizer.build_conversation_input_ids.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock()
    }
    mock_model.generate.return_value = [MagicMock()]
    mock_model.device = "cpu"
    
    return mock_tokenizer, mock_model

@pytest.fixture
def mock_test_image():
    """Create a test image for testing."""
    img = Image.new('RGB', (800, 600), color='white')
    img_path = "test_anthropic_error.png"
    img.save(img_path)
    yield img_path
    
    # Cleanup
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
    except Exception as e:
        print(f"Warning: Failed to clean up test image: {e}")

def test_process_image_anthropic_error(mock_model_and_tokenizer, mock_test_image):
    """Test processing an image with anthropic error."""
    mock_tokenizer, mock_model = mock_model_and_tokenizer
    
    # Mock JSON response for anthropic error
    json_response = {
        "error_type": "anthropic_unavailable",
        "message": "Unable to reach anthropic",
        "buttons": [
            {
                "x1": 400,
                "y1": 300,
                "x2": 500,
                "y2": 330,
                "text": "Try Again"
            }
        ]
    }
    
    # Set up the mock to return our JSON response
    mock_tokenizer.decode.return_value = json.dumps(json_response)
    
    # Call the function
    result = process_image_with_model(mock_test_image, mock_tokenizer, mock_model)
    
    # Verify the result
    assert result is not None
    assert result.error_type == "anthropic_unavailable"
    assert "Unable to reach anthropic" in result.message
    assert len(result.buttons) == 1
    assert result.buttons[0].text == "Try Again"
    assert result.buttons[0].x1 == 400
    assert result.buttons[0].y1 == 300
    assert result.buttons[0].x2 == 500
    assert result.buttons[0].y2 == 330
    
    # Verify the model was called with the correct arguments
    mock_tokenizer.build_conversation_input_ids.assert_called_once()
    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once()

def test_process_image_no_error(mock_model_and_tokenizer, mock_test_image):
    """Test processing an image with no error."""
    mock_tokenizer, mock_model = mock_model_and_tokenizer
    
    # Mock JSON response for no error
    json_response = {
        "error_type": "none",
        "message": "",
        "buttons": []
    }
    
    # Set up the mock to return our JSON response
    mock_tokenizer.decode.return_value = json.dumps(json_response)
    
    # Call the function
    result = process_image_with_model(mock_test_image, mock_tokenizer, mock_model)
    
    # Verify the result
    assert result is None
    
    # Verify the model was called with the correct arguments
    mock_tokenizer.build_conversation_input_ids.assert_called_once()
    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once()

def test_process_image_invalid_json(mock_model_and_tokenizer, mock_test_image):
    """Test handling of invalid JSON response."""
    mock_tokenizer, mock_model = mock_model_and_tokenizer
    
    # Set up the mock to return an invalid JSON response
    mock_tokenizer.decode.return_value = "This is not valid JSON"
    
    # Call the function
    result = process_image_with_model(mock_test_image, mock_tokenizer, mock_model)
    
    # Verify the result
    assert result is None
    
    # Verify the model was called with the correct arguments
    mock_tokenizer.build_conversation_input_ids.assert_called_once()
    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once()

def test_process_image_fallback_detection(mock_model_and_tokenizer, mock_test_image):
    """Test fallback text-based detection when JSON parsing fails."""
    mock_tokenizer, mock_model = mock_model_and_tokenizer
    
    # Set up the mock to return a non-JSON response with error text
    mock_tokenizer.decode.return_value = "I can see a message saying 'Unable to reach anthropic'"
    
    # Call the function
    result = process_image_with_model(mock_test_image, mock_tokenizer, mock_model)
    
    # Verify the result (should detect based on text content)
    assert result is not None
    assert result.error_type == "anthropic_unavailable"
    
    # Verify the model was called with the correct arguments
    mock_tokenizer.build_conversation_input_ids.assert_called_once()
    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once()

@patch('cursor_clicker.main.pyautogui.click')
@patch('cursor_clicker.main.pyautogui.getWindowsWithTitle')
@patch('cursor_clicker.main.time.sleep')
def test_anthropic_error_handling_flow(mock_sleep, mock_get_windows, mock_click):
    """Test the entire flow of handling an anthropic error (integration test)."""
    # This test would need to integrate with the main function
    # It's a more complex integration test that would need to mock multiple components
    # Keeping as a placeholder for future implementation
    pass 