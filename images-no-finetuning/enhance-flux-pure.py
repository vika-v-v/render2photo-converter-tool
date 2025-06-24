import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from dotenv import load_dotenv

def process_images(input_folder, output_folder, model_id="black-forest-labs/FLUX.1-schnell", strength=0.2, seed=42):
    """Process all images in the input folder using FLUX model"""
    
    # Load environment variables
    load_dotenv()
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    
    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN not found in .env file")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Configure to use 2 CPUs
    torch.set_num_threads(2)
    print(f"Using {torch.get_num_threads()} CPU threads")
    
    # Load FLUX model specifically for image-to-image
    print(f"Loading FLUX model for image-to-image: {model_id}...")
    
    # Use AutoPipelineForImage2Image specifically for img2img
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float32,  # Use float32 for CPU
        use_auth_token=hf_token,    # Use the token from .env
        local_files_only=False      # Allow downloading if needed
    )
    
    # Move to CPU
    pipe = pipe.to("cpu")
    print("FLUX model loaded successfully")
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    generator = torch.Generator("cpu").manual_seed(seed)
    
    for i, filename in enumerate(image_files, 1):
        print(f"Processing image {i}/{len(image_files)}: {filename}")
        
        # Load image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")
        
        # Get original dimensions
        width, height = img.size
        aspect_ratio = width / height
        
        # Resize if necessary
        max_dimension = 1024
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(max_dimension / aspect_ratio)
            else:
                new_height = max_dimension
                new_width = int(max_dimension * aspect_ratio)
            
            print(f"Resizing from {width}x{height} to {new_width}x{new_height}")
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Make sure dimensions are divisible by 8
        width, height = img.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        if new_width != width or new_height != height:
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"Adjusted dimensions to be divisible by 8: {new_width}x{new_height}")
        
        # For FLUX model, use a prompt that describes your office interior
        prompt = "high quality photograph of office interrior, photorealistic, realistic people, proportional faces, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighting, consistent shadows, preserve as many details from the original image as possible, 8k, 4k, sharp focus"
        negative_prompt = "low quality, bad anatomy, bad hands, text, error, blurry, out of focus, low resolution, cropped, worst quality, jpeg artifacts, signature, watermark, distorted"

        # Process with FLUX using the image2image pipeline
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img,
            strength=strength,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=img.width,  # Explicitly set width to match input image
            height=img.height,  # Explicitly set height to match input image
            generator=generator
        ).images[0]
        
        # Save the result
        output_path = os.path.join(output_folder, f"realistic_{filename}")
        result.save(output_path)
        print(f"Saved enhanced image to {output_path}")
    
    print("All images processed successfully!")

if __name__ == "__main__":
    input_folder = "task-images"
    
    # You can easily switch between FLUX models
    # model_id = "black-forest-labs/FLUX.1-schnell"  # Fast version
    model_id = "black-forest-labs/FLUX.1-schnell"      # Higher quality version
    
    # Update output folder name based on model
    model_name = model_id.split("/")[-1].lower()
    output_folder = f"output-images-{model_name}-30-steps-strength-04-seed-42"
    
    # Use a strength in the middle of the requested range
    strength = 0.4
    
    process_images(input_folder, output_folder, model_id, strength)