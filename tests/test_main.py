"""Tests for the main application module."""

import os
import sys
import pytest
import torch
from PIL import Image
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cursor_clicker.main import (
    setup_model,
    process_image_with_model,
    perform_ocr,
    contains_tool_call_limit_text
)

@pytest.fixture(scope="function")
def test_image():
    """Create a test image with some text."""
    img = Image.new('RGB', (400, 100), color='white')
    # Save the image
    img_path = "test_capture.png"
    img.save(img_path)
    # Close the image to release file handles
    img.close()
    
    yield img_path
    
    # Cleanup after tests
    try:
        # Make sure any open file handles are closed by the garbage collector
        import gc
        gc.collect()
        if os.path.exists(img_path):
            os.remove(img_path)
    except (PermissionError, OSError) as e:
        # Log the error but don't fail the test over cleanup issues
        print(f"WARNING: Could not remove test file {img_path}: {e}")

@pytest.mark.skip("Requires HuggingFace authentication")
def test_setup_model():
    """Test that the model can be loaded successfully."""
    with patch.dict(os.environ, {"HF_TOKEN": "dummy_token"}):
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            with patch("transformers.AutoModelForVision2Seq.from_pretrained") as mock_model:
                mock_tokenizer.return_value = MagicMock()
                mock_model.return_value = MagicMock()
                
                tokenizer, model = setup_model()
                
                assert tokenizer is not None, "Tokenizer should not be None"
                assert model is not None, "Model should not be None"
                mock_tokenizer.assert_called_once()
                mock_model.assert_called_once()

def test_perform_ocr(test_image):
    """Test OCR functionality."""
    with patch("pytesseract.image_to_string", return_value="test text"):
        result = perform_ocr(test_image)
        assert result == "test text"

def test_contains_tool_call_limit_text():
    """Test detection of tool call limit text."""
    # Test positive cases
    assert contains_tool_call_limit_text("We've hit our tool call limit")
    assert contains_tool_call_limit_text("Sorry, but we've hit our tool call limit today")
    assert contains_tool_call_limit_text("reached our limit for API calls")
    
    # Test negative cases
    assert not contains_tool_call_limit_text("Everything is working fine")
    assert not contains_tool_call_limit_text("No issues detected")

@pytest.mark.skip("Requires actual model for inference")
def test_process_image_with_model(test_image):
    """Test image processing with the model."""
    # Mock tokenizer and model
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    
    # Configure mock responses
    mock_tokenizer.build_conversation_input_ids.return_value = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1])
    }
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_tokenizer.decode.return_value = "Yes, I see a tool call limit message."
    mock_model.device = "cpu"
    
    result = process_image_with_model(test_image, mock_tokenizer, mock_model)
    assert result is True

    # Test negative response
    mock_tokenizer.decode.return_value = "No, I don't see any tool call limit message."
    result = process_image_with_model(test_image, mock_tokenizer, mock_model)
    assert result is False 