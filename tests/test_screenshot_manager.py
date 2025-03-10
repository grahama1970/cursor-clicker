"""Tests for the screenshot_manager module."""

import os
import time
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cursor_clicker.screenshot_manager import (
    ensure_dir_exists,
    get_timestamp,
    capture_cursor_window,
    save_screenshot,
    capture_and_save,
    get_latest_screenshot,
    get_most_recent_screenshot
)

# Test directory for test screenshots
TEST_DIR = "test_screenshots"

@pytest.fixture(scope="function")
def cleanup_test_dir():
    """Delete test directory after tests."""
    yield
    if os.path.exists(TEST_DIR):
        for file in os.listdir(TEST_DIR):
            try:
                os.remove(os.path.join(TEST_DIR, file))
            except Exception:
                pass
        try:
            os.rmdir(TEST_DIR)
        except Exception:
            pass

def test_ensure_dir_exists(cleanup_test_dir):
    """Test directory creation."""
    ensure_dir_exists(TEST_DIR)
    assert os.path.exists(TEST_DIR)
    assert os.path.isdir(TEST_DIR)

def test_get_timestamp():
    """Test timestamp generation."""
    timestamp = get_timestamp()
    assert isinstance(timestamp, str)
    assert len(timestamp) == 15
    assert "_" in timestamp

@patch('pygetwindow.getWindowsWithTitle')
def test_capture_cursor_window_no_window(mock_get_windows):
    """Test behavior when no Cursor window is found."""
    mock_get_windows.return_value = []
    result = capture_cursor_window()
    assert result is None
    mock_get_windows.assert_called_once_with("Cursor.sh")

@patch('pygetwindow.getWindowsWithTitle')
def test_capture_cursor_window_activation_fail(mock_get_windows):
    """Test behavior when window activation fails."""
    mock_window = MagicMock()
    mock_window.activate.side_effect = Exception("Test exception")
    mock_window.isActive = False
    mock_get_windows.return_value = [mock_window]
    
    result = capture_cursor_window(activation_timeout=0.1)
    assert result is None
    mock_get_windows.assert_called_once_with("Cursor.sh")

@patch('pygetwindow.getWindowsWithTitle')
@patch('pyautogui.screenshot')
@patch('time.sleep')
def test_capture_cursor_window_success(mock_sleep, mock_screenshot, mock_get_windows):
    """Test successful window capture."""
    mock_window = MagicMock()
    mock_window.left = 0
    mock_window.top = 0
    mock_window.width = 800
    mock_window.height = 600
    mock_window.isActive = True
    mock_get_windows.return_value = [mock_window]
    
    mock_image = MagicMock(spec=Image.Image)
    mock_screenshot.return_value = mock_image
    
    result = capture_cursor_window()
    assert result == mock_image
    mock_screenshot.assert_called_once_with(region=(0, 0, 800, 600))

def test_save_screenshot_no_compression(cleanup_test_dir):
    """Test saving screenshot without compression."""
    mock_image = MagicMock(spec=Image.Image)
    result = save_screenshot(
        mock_image, 
        directory=TEST_DIR,
        use_timestamp=False,
        compress=False
    )
    
    assert result == os.path.join(TEST_DIR, "latest_screenshot.png")
    mock_image.save.assert_called_once()
    assert os.path.exists(TEST_DIR)

@patch('time.time')
def test_save_screenshot_with_compression(mock_time, cleanup_test_dir):
    """Test saving screenshot with compression."""
    mock_time.return_value = 123456789.0
    
    # Create a test image
    test_image = Image.new('RGB', (400, 300), color='red')
    
    result = save_screenshot(
        test_image, 
        directory=TEST_DIR,
        use_timestamp=False,
        compress=True
    )
    
    assert result == os.path.join(TEST_DIR, "latest_screenshot_compressed.jpg")
    assert os.path.exists(os.path.join(TEST_DIR, "latest_screenshot.png"))
    assert os.path.exists(os.path.join(TEST_DIR, "latest_screenshot_compressed.jpg"))

@patch('time.time')
def test_save_screenshot_with_timestamp(mock_time, cleanup_test_dir):
    """Test saving screenshot with timestamp."""
    mock_time.return_value = 123456789.0
    
    # Mock timestamp
    timestamp = "20230101_120000"
    with patch('cursor_clicker.screenshot_manager.get_timestamp', return_value=timestamp):
        # Create a test image
        test_image = Image.new('RGB', (400, 300), color='blue')
        
        result = save_screenshot(
            test_image, 
            directory=TEST_DIR,
            use_timestamp=True,
            compress=True
        )
        
        expected_path = os.path.join(TEST_DIR, f"screenshot_{timestamp}_compressed.jpg")
        assert result == expected_path
        assert os.path.exists(os.path.join(TEST_DIR, f"screenshot_{timestamp}.png"))
        assert os.path.exists(expected_path)

@patch('cursor_clicker.screenshot_manager.capture_cursor_window')
def test_capture_and_save_failure(mock_capture):
    """Test behavior when capture fails."""
    mock_capture.return_value = None
    result = capture_and_save(directory=TEST_DIR)
    assert result is None

@patch('cursor_clicker.screenshot_manager.capture_cursor_window')
@patch('cursor_clicker.screenshot_manager.save_screenshot')
def test_capture_and_save_success(mock_save, mock_capture, cleanup_test_dir):
    """Test successful capture and save."""
    mock_image = MagicMock(spec=Image.Image)
    mock_image.width = 800
    mock_image.height = 600
    mock_capture.return_value = mock_image
    
    expected_path = os.path.join(TEST_DIR, "screenshot.jpg")
    mock_save.return_value = expected_path
    
    result = capture_and_save(directory=TEST_DIR)
    assert result == expected_path
    mock_save.assert_called_once()

def test_get_latest_screenshot_not_found():
    """Test behavior when no screenshot exists."""
    non_existent_dir = "non_existent_directory"
    result = get_latest_screenshot(directory=non_existent_dir)
    assert result is None

def test_get_latest_screenshot_found(cleanup_test_dir):
    """Test finding the latest screenshot."""
    ensure_dir_exists(TEST_DIR)
    
    # Create a dummy file
    test_file = os.path.join(TEST_DIR, "latest_screenshot_compressed.jpg")
    with open(test_file, 'w') as f:
        f.write("test")
    
    result = get_latest_screenshot(directory=TEST_DIR)
    assert result == test_file
    
    # Test non-compressed variant
    test_file2 = os.path.join(TEST_DIR, "latest_screenshot.png")
    with open(test_file2, 'w') as f:
        f.write("test")
    
    result2 = get_latest_screenshot(directory=TEST_DIR, compressed=False)
    assert result2 == test_file2 