# 🖱️ Cursor Clicker

Automated tool for handling Cursor.sh's limitations when running unattended. This application uses the Qwen2.5-VL visual language model to analyze screenshots of the Cursor window, detect error states, and automatically resolve them so your AI coding sessions can continue without manual intervention.

## 🔥 Why Cursor Clicker is Needed

If you've used Cursor.sh for an extended period, you've likely encountered these frustrating limitations:

- **Tool call limits**: Cursor imposes daily limits on tool calls (such as file reads, terminal commands, etc.). When you hit this limit, you need to manually click "Continue" to proceed with limited functionality.
- **Anthropic availability issues**: Claude (the only decent model in Cursor) occasionally becomes unavailable with the dreaded "Unable to reach anthropic" error, requiring a manual "Try Again" click.

These limitations make it impossible to run Cursor unattended for long periods, which is **highly annoying** and something Cursor should have implemented long ago. Cursor Clicker solves this by:

1. Automatically detecting when these errors occur
2. Waiting an appropriate time (for rate limits to reset or services to recover)
3. Clicking the correct button to resume operation

**No more babysitting your Cursor instance!**

## 🧠 Features

- Automatically captures screenshots of the Cursor.sh window
- Uses advanced vision AI (Qwen2.5-VL) to analyze screenshots
- Detects multiple error conditions:
  - Tool call limits reached
  - Anthropic service unavailability
- Precisely locates buttons on the screen using AI vision
- Automatically clicks the appropriate buttons after waiting periods
- Manages screenshots with either timestamp-based naming or overwriting
- Detailed logging of all operations

## ⚠️ Edge Cases Handled

Cursor Clicker handles several challenging edge cases:

1. **Tool call limit detection**: Recognizes various phrasings of the limit message
2. **Anthropic unavailability**: Detects "Unable to reach anthropic" errors
3. **Button location**: Uses AI vision to precisely locate buttons regardless of window size or position
4. **Service recovery**: Waits 5 minutes before retrying Anthropic connections
5. **False positives**: Uses threshold detection to avoid false positives
6. **Window activation**: Properly handles window focus and activation
7. **Accessibility**: Works in various screen resolutions and window configurations

## ⚙️ Installation

```bash
# Install from PyPI (recommended)
pip install cursor-clicker

# Or install from source
git clone https://github.com/yourusername/cursor_clicker.git
cd cursor_clicker
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Prerequisites

- Python 3.8 or higher
- A Hugging Face API token (free, but required to use Qwen models)
- PyTorch 2.0 or higher
- CUDA-compatible GPU recommended but not required

### Configuration

Create a `.env` file in your working directory with your Hugging Face token:

```
HF_TOKEN=your_hugging_face_token_here
```

## 🚀 Usage

### Running the main application

```bash
# Start the main application that monitors Cursor.sh
cursor-clicker
```

The application will:
1. Start monitoring your Cursor.sh window
2. Detect when tool call limits or Anthropic errors occur
3. Wait appropriate times (for rate limits or service recovery)
4. Automatically click buttons to resume operation

### Testing without loading the ML model

If you just want to test the screenshot functionality without loading the model:

```bash
# Run the continuous monitor only (for testing)
continuous-monitor

# Or with options
continuous-monitor --interval 10 --timestamp
```

## 🛠️ Development

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (optional, but recommended)

### Testing

```bash
# Run tests
pytest
```

## 📝 License

MIT License. See the `LICENSE` file for details. 