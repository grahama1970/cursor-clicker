"""
Script to use Qwen2.5-VL model for analyzing an image.
Based on official documentation: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
"""

import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def compress_image(input_path, max_width=800, quality=75):
    """Compress an image to reduce size for faster processing."""
    # Create output path based on input path
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, _ = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_compressed.jpg")
    
    # Skip if already exists
    if os.path.exists(output_path):
        print(f"Using existing compressed image: {output_path}")
        return output_path
        
    # Compress the image
    print(f"Compressing image {input_path}...")
    original = Image.open(input_path)
    
    # Convert RGBA to RGB for JPEG compatibility
    if original.mode == 'RGBA':
        original = original.convert('RGB')
    
    width, height = original.size
    new_width = min(width, max_width)
    new_height = int(height * (new_width / width))
    
    resized = original.resize((new_width, new_height), Image.LANCZOS)
    resized.save(output_path, "JPEG", quality=quality)
    
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:,} bytes")
    print(f"Compression ratio: {original_size / compressed_size:.2f}x")
    
    return output_path

def main():
    # Path to the test image
    image_path = "screenshots/test_capture.PNG"
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
        
    # Get the Hugging Face token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not found")
        return
    
    # Compress the image for faster processing
    compressed_path = compress_image(image_path)
    
    print("Loading model and processor...")
    
    # Determine device and appropriate dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Using device: {device} with dtype: {dtype}")
    
    # Load model exactly as documented
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        token=hf_token
    )
    
    # Use AutoProcessor as specified in the documentation
    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    
    print(f"Model and processor loaded successfully. Using image: {compressed_path}")
    
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
    
    print("Processing input...")
    
    # Preparation for inference - exactly as in documentation
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
    )
    inputs = inputs.to(model.device)
    
    print("Generating response...")
    
    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256,
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
        
    print("\nQwen's response:")
    print("-" * 50)
    print(output_text[0])
    print("-" * 50)

if __name__ == "__main__":
    main() 