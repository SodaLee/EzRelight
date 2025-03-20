from PIL import Image
import base64
import io
import os
import ollama

def image_to_base64(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Create a BytesIO object to hold the image data
        buffered = io.BytesIO()
        # Save the image to the BytesIO object in a specific format (e.g., JPEG)
        img.save(buffered, format="PNG")
        # Get the byte data from the BytesIO object
        img_bytes = buffered.getvalue()
        # Encode the byte data to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

dataset_root = '/data/lihaochen/datasets/TikTok_dataset/'
for j in range(1, 341):
    video_dir = os.path.join(dataset_root, '%05d' % j)
    image_dir = os.path.join(video_dir, 'images')
    video_len = len(os.listdir(image_dir))

    for i in range(1, video_len+1):
        image_path = os.path.join(image_dir, '%04d.png' % i)
        base64_image = image_to_base64(image_path)

        # Use Ollama to clean and structure the OCR output
        response = ollama.chat(
            model="gemma3:27b",
            messages=[{
            "role": "system",
            "content": "You are a creative assistant that generates detailed and visually descriptive prompts for images. Your task is to create a prompt based on the provided image. Only output the prompt itself, without any additional explanations, formatting, labels, or line breaks."
            },
            {
            "role": "user",
            "content": "Please generate a detailed and creative prompt for an image. No more than 100 words. Only output the prompt itself, without any additional explanations, formatting, or labels. The prompt should be visually descriptive, inspiring, and suitable for generating high-quality artwork or photography. Include details about the setting, mood, colors, lighting, and any specific elements or emotions you want to convey. Make the prompt engaging and imaginative, while ensuring it is clear and easy to understand.",
            "images": [base64_image]
            }],
        )
        # Extract cleaned text
        cleaned_text = response['message']['content'].replace("\n", " ").strip()
        print(j, i, cleaned_text)
        with open(os.path.join(video_dir, 'captions.txt'), 'a') as f:
            f.write(cleaned_text + '\n')