import os
from PIL import Image

def main():
    # Path to the test image
    image_path = "screenshots/test_capture.PNG"
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    # Open the image
    print(f"Opening image: {image_path}")
    image = Image.open(image_path)
    
    # Display basic information about the image
    print("\nImage information:")
    print("-" * 50)
    print(f"Format: {image.format}")
    print(f"Mode: {image.mode}")
    print(f"Size: {image.width} x {image.height} pixels")
    
    # Analyze the image (simple analysis using colors)
    print("\nAnalyzing image...")
    
    # Get a small sample of pixels from the bottom area (where chat messages typically are)
    bottom_region = image.crop((0, image.height - 100, image.width, image.height))
    
    # Count the most common colors in the bottom region
    colors = bottom_region.getcolors(maxcolors=10000)
    if colors:
        colors.sort(reverse=True)
        print("\nMost common colors in the bottom region:")
        for count, color in colors[:5]:
            print(f"Count: {count}, Color: {color}")
    
    print("\nImage loaded and analyzed successfully.")
    
if __name__ == "__main__":
    main() 