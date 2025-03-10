from cursor_clicker import setup_model, chat_with_qwen
import os

def main():
    # Setup the model and tokenizer
    print("Loading model...")
    tokenizer, model = setup_model()
    
    # Path to the test image
    image_path = "screenshots/test_capture.PNG"
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    # Create the message with the user's latest query and the image
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'image': f"file://{image_path}"
                },
                {
                    'type': 'text',
                    'text': 'Please read and identify the text in this screenshot. Specifically, tell me what message the user is seeing at the bottom of the chat window.'
                }
            ]
        }
    ]
    
    print(f"Sending prompt to Qwen with image: {image_path}...")
    
    # Use the updated chat_with_qwen function
    response = chat_with_qwen(messages, model, tokenizer)
    
    print('\nQwen\'s response:')
    print('-' * 50)
    print(response)
    print('-' * 50)

if __name__ == "__main__":
    main() 