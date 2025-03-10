"""
Script to use Qwen2.5-VL model for analyzing an image with proper logging.
Based on official documentation: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
"""

import os
import time
import torch
import logging
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def compress_image(input_path, max_width=800, quality=75):
    """Compress an image to reduce size for faster processing."""
    # Create output path based on input path
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, _ = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_compressed.jpg")
    
    # Skip if already exists but log the sizes
    if os.path.exists(output_path):
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        logger.info(f"Using existing compressed image: {output_path}")
        logger.info(f"Original image: {original_size:,} bytes ({original_size/1024:.1f} KB)")
        logger.info(f"Compressed image: {compressed_size:,} bytes ({compressed_size/1024:.1f} KB)")
        logger.info(f"Compression ratio: {ratio:.2f}x")
        
        return output_path
        
    # Compress the image
    logger.info(f"Compressing image {input_path}...")
    start_time = time.time()
    
    original = Image.open(input_path)
    logger.info(f"Original dimensions: {original.width}x{original.height}, Mode: {original.mode}")
    
    # Convert RGBA to RGB for JPEG compatibility
    if original.mode == 'RGBA':
        original = original.convert('RGB')
        logger.info("Converted image from RGBA to RGB for JPEG compatibility")
    
    width, height = original.size
    new_width = min(width, max_width)
    new_height = int(height * (new_width / width))
    
    resized = original.resize((new_width, new_height), Image.LANCZOS)
    logger.info(f"Resized dimensions: {resized.width}x{resized.height}")
    
    resized.save(output_path, "JPEG", quality=quality)
    
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    logger.info(f"Original image: {original_size:,} bytes ({original_size/1024:.1f} KB)")
    logger.info(f"Compressed image: {compressed_size:,} bytes ({compressed_size/1024:.1f} KB)")
    logger.info(f"Compression ratio: {ratio:.2f}x")
    logger.info(f"Compression completed in {time.time() - start_time:.2f} seconds")
    
    return output_path

def get_device_info():
    """Get detailed information about available devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_info = []
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_props = torch.cuda.get_device_properties(i)
            total_memory = device_props.total_memory / 1024 / 1024  # Convert to MB
            device_info.append(f"GPU {i}: {device_name}, {total_memory:.0f} MB total memory")
        
        return "cuda", device_info
    else:
        return "cpu", ["CPU only, no GPU detected"]

def main():
    logger.info("===== Starting Qwen2.5-VL Image Analysis =====")
    
    # Check device availability
    device_type, device_info = get_device_info()
    for info in device_info:
        logger.info(info)
    
    # Path to the test image
    image_path = "screenshots/test_capture.PNG"
    if not os.path.exists(image_path):
        logger.error(f"Error: Image file {image_path} not found!")
        return
        
    # Get the Hugging Face token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("Error: HF_TOKEN environment variable not found")
        return
    
    # Compress the image for faster processing
    compressed_path = compress_image(image_path)
    
    logger.info("Loading model and processor...")
    model_load_start = time.time()
    
    # Determine device and appropriate dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    logger.info(f"Using device: {device} with dtype: {dtype}")
    
    # Load model exactly as documented
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token
        )
        
        # Use AutoProcessor as specified in the documentation
        processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        
        model_load_time = time.time() - model_load_start
        logger.info(f"Model and processor loaded successfully in {model_load_time:.2f} seconds")
        
        # Check where the model is actually loaded
        if hasattr(model, 'hf_device_map'):
            logger.info("Model device map:")
            for module, device in model.hf_device_map.items():
                logger.info(f"  {module}: {device}")
        else:
            logger.info(f"Model device: {next(model.parameters()).device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
    # Construct the messages in the exact format shown in documentation
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{compressed_path}"
                },
                {
                    "type": "text", 
                    "text": "Please look at this screenshot and tell me what message is displayed at the bottom of the chat window. Read the text carefully and tell me exactly what it says."
                }
            ]
        }
    ]
    
    logger.info("Processing input...")
    process_start = time.time()
    
    try:
        # Preparation for inference - exactly as in documentation
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        input_process_time = time.time() - process_start
        logger.info(f"Input processing completed in {input_process_time:.2f} seconds")
        
        # Log input shape information
        logger.info("Preparing model inputs...")
        tokenize_start = time.time()
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the appropriate device
        inputs = inputs.to(model.device)
        
        tokenize_time = time.time() - tokenize_start
        logger.info(f"Tokenization completed in {tokenize_time:.2f} seconds")
        logger.info(f"Input shape: {inputs.input_ids.shape}")
        
        # Inference: Generation of the output
        logger.info("Generating response...")
        inference_start = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128,  # Reduced for faster inference
                do_sample=True,
                temperature=0.2
            )
            
            # Process the output correctly
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        inference_time = time.time() - inference_start
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        total_time = time.time() - process_start
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        logger.info("\nQwen's response:")
        logger.info("=" * 50)
        logger.info(output_text[0])
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 