import os
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info, fetch_image

def main():
    print("Loading model and tokenizer...")
    
    # Get Hugging Face token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not found")
        return
    
    # Path to the test image
    image_path = "screenshots/test_capture.PNG"
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    
    # Use float32 since we're on CPU
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True,
        token=hf_token
    )
    
    # Create messages with the image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}"
                },
                {
                    "type": "text",
                    "text": "Please look at this image and tell me what message is displayed at the bottom of the chat window."
                }
            ]
        }
    ]
    
    # Process the vision inputs
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Create prompt with the chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Prepare inputs for the model
    inputs = tokenizer(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.001,
            top_k=1
        )
    
    # Decode the output
    output_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    response = tokenizer.batch_decode(output_ids_trimmed, skip_special_tokens=True)[0]
    
    print("\nResponse:")
    print("-" * 50)
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    main() 