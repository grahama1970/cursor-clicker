"""
Tests for Qwen2.5-VL model functionality.
Based on official documentation: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
"""

import os
import pytest
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

@pytest.fixture
def image_path():
    """Fixture for the test image path."""
    path = "screenshots/test_capture.PNG"
    if not os.path.exists(path):
        pytest.skip(f"Test image not found at {path}")
    return path

@pytest.fixture
def compressed_image_path(image_path):
    """Create a compressed version of the test image."""
    compressed_path = "screenshots/test_capture_compressed.jpg"
    
    # Skip if already exists
    if os.path.exists(compressed_path):
        return compressed_path
        
    # Compress the image
    original = Image.open(image_path)
    # Convert RGBA to RGB for JPEG compatibility
    if original.mode == 'RGBA':
        original = original.convert('RGB')
        
    width, height = original.size
    new_width = min(width, 800)  # Max width of 800 pixels
    new_height = int(height * (new_width / width))
    
    resized = original.resize((new_width, new_height), Image.LANCZOS)
    resized.save(compressed_path, "JPEG", quality=75)
    
    return compressed_path

@pytest.fixture
def hf_token():
    """Fixture for the Hugging Face token."""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN environment variable not found")
    return token

@pytest.fixture
def model(hf_token):
    """Fixture for the Qwen model."""
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        token=hf_token
    )
    return model

@pytest.fixture
def processor(hf_token):
    """Fixture for the model processor."""
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    return AutoProcessor.from_pretrained(model_name, token=hf_token)

def test_model_loading(model, processor):
    """Test that the model and processor can be loaded."""
    assert model is not None, "Model should be loaded"
    assert processor is not None, "Processor should be loaded"

def test_image_compression(image_path, compressed_image_path):
    """Test that the image can be compressed."""
    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize(compressed_image_path)
    
    assert compressed_size < original_size, "Compressed image should be smaller"
    print(f"Original size: {original_size} bytes, Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {original_size / compressed_size:.2f}x")

def test_model_inference(model, processor, compressed_image_path):
    """Test that the model can perform inference on an image."""
    # Create the message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{compressed_image_path}"
                },
                {
                    "type": "text", 
                    "text": "What text is visible at the bottom of this screenshot?"
                }
            ]
        }
    ]
    
    # Process the input
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Generate a short response for testing
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=50,  # Short response for testing
            do_sample=False     # Deterministic output for testing
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    
    # We just need to verify that we get some text output
    assert isinstance(output_text, list), "Output should be a list"
    assert len(output_text) > 0, "Output list should not be empty"
    assert isinstance(output_text[0], str), "Output item should be a string"
    assert len(output_text[0]) > 0, "Output string should not be empty"
    
    print(f"Model output: {output_text[0]}") 