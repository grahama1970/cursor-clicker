import os
from PIL import Image
import pytesseract

def main():
    # Path to the test image
    image_path = "screenshots/test_capture.PNG"
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    # Open the image
    print(f"Opening image: {image_path}")
    image = Image.open(image_path)
    
    # Extract text from the image
    print("Extracting text from image...")
    text = pytesseract.image_to_string(image)
    
    # Display the extracted text
    print("\nExtracted text:")
    print("-" * 50)
    print(text)
    print("-" * 50)
    
    # Try to find the specific message about providing a sample image
    lines = text.split('\n')
    for line in lines:
        if "sample image" in line.lower():
            print("\nFound message about sample image:")
            print(line)

if __name__ == "__main__":
    main() 