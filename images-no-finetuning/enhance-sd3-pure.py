import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from PIL import Image
import os
import gc
from dotenv import load_dotenv

# Set GPU device before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 to avoid memory conflicts

# Load environment variables
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# Configuration
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
INPUT_IMAGE_PATH = "task-images/KI_03.jpg"
OUTPUT_IMAGE_PATH = "output/realistic_photo.png"

PROMPT = "high quality photograph, photorealistic, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighting, consistent shadows, preserve as many details from the original image as possible, 8k, 4k, sharp focus, distant objects have consistent detailes, distant objects have consistent shapes"
NEGATIVE_PROMPT = "low quality, bad anatomy, bad hands, error, blurry, out of focus, low resolution, cropped, worst quality, jpeg artifacts, signature, watermark, distorted, inconsistent shapes, inconsistent lighting, inconsistent shadows, descaled, descaling, descaled image, descaled photo"

# Parameters
STRENGTH = 0.3
GUIDANCE_SCALE = 7
NUM_INFERENCE_STEPS = 400
SEED = 42  # Set to None for random results
MAX_IMAGE_SIZE = 2048  # Maximum size for the larger dimension

def resize_image_to_max_dimension(image, max_size=1024):
    """
    Resize image so that its larger dimension is at most max_size pixels,
    while maintaining the aspect ratio. Ensures dimensions are multiples of 8.
    """
    width, height = image.size
    
    # Calculate the scaling factor based on the larger dimension
    max_dimension = max(width, height)
    
    if max_dimension > max_size:
        scale_factor = max_size / max_dimension
    else:
        scale_factor = 1.0
    
    # Calculate new dimensions - ensure we maintain aspect ratio
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Ensure dimensions are multiples of 16 (required by SD models)
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16
    
    # Ensure at least 16 pixels
    new_width = max(new_width, 16)
    new_height = max(new_height, 16)
    
    # Verify aspect ratio is maintained (for debugging)
    original_ratio = width / height
    new_ratio = new_width / new_height
    
    # Only resize if needed
    if scale_factor < 1.0 or width != new_width or height != new_height:
        print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        print(f"Original aspect ratio: {original_ratio:.3f}, New aspect ratio: {new_ratio:.3f}")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        print(f"Image size {width}x{height} is already within limits")
    
    return image

def load_and_prepare_image(image_path):
    """Load and prepare the input image with proper resizing."""
    image = Image.open(image_path).convert("RGB")
    
    # Resize if needed
    image = resize_image_to_max_dimension(image, MAX_IMAGE_SIZE)
    
    return image

def setup_pipeline():
    """Initialize the Stable Diffusion 3.5 Large pipeline for img2img with memory optimizations."""
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the pipeline with memory optimizations
    print("Loading Stable Diffusion 3.5 Large model...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,  # Use fp16 variant if available
        use_auth_token=HUGGING_FACE_TOKEN if HUGGING_FACE_TOKEN else None,
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
    )
    
    # Enable memory efficient features before moving to device
    pipe.enable_model_cpu_offload()  # Offload model parts to CPU when not in use
    
    # Enable additional memory optimizations
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)  # Maximum memory savings
    
    # Enable xformers memory efficient attention if available
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("Xformers memory efficient attention enabled")
    except:
        print("Xformers not available, using default attention")
    
    # VAE slicing for large images
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        print("VAE slicing enabled for memory efficiency")
    
    # Set memory efficient attention processor
    if hasattr(pipe, "set_attn_processor"):
        from diffusers.models.attention_processor import AttnProcessor2_0
        pipe.unet.set_attn_processor(AttnProcessor2_0())
    
    return pipe

def generate_image(pipe, input_image):
    """Generate the transformed image with memory management."""
    # Clear cache before generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get input image dimensions
    width, height = input_image.size
    
    # Set random seed for reproducibility
    generator = None
    if SEED is not None:
        generator = torch.Generator(device="cpu").manual_seed(SEED)  # Use CPU generator
    
    # Generate the image with memory efficient settings
    with torch.inference_mode():  # Disable gradient computation
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=input_image,
            strength=STRENGTH,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            width=width,  # Explicitly set output width
            height=height,  # Explicitly set output height
        )
    
    # Clear cache after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result.images[0]

def main():
    """Main execution function."""
    print("Starting Stable Diffusion 3.5 Large Image-to-Image generation...")
    
    # Print GPU memory status
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    
    # Load input image
    print(f"Loading input image from: {INPUT_IMAGE_PATH}")
    input_image = load_and_prepare_image(INPUT_IMAGE_PATH)
    print(f"Final input image size: {input_image.size}")
    
    # Setup pipeline
    print("Setting up Stable Diffusion 3.5 Large pipeline...")
    pipe = setup_pipeline()
    
    # Generate image
    print("Generating realistic photo...")
    print(f"Parameters: strength={STRENGTH}, guidance_scale={GUIDANCE_SCALE}, steps={NUM_INFERENCE_STEPS}")
    output_image = generate_image(pipe, input_image)
    
    # Save the result
    output_image.save(OUTPUT_IMAGE_PATH)
    print(f"Image saved to: {OUTPUT_IMAGE_PATH}")
    print(f"Output image size: {output_image.size}")
    
    # Final cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    main()

# Installation requirements:
"""
pip install diffusers transformers torch torchvision accelerate pillow python-dotenv
# Required for memory efficiency:
pip install xformers
"""

# Memory optimization tips:
"""
If you still encounter OOM errors, try these additional optimizations:

1. Reduce MAX_IMAGE_SIZE further (e.g., to 768 or 512)
2. Use sequential CPU offloading instead of model CPU offloading:
   pipe.enable_sequential_cpu_offload()
3. Reduce batch size if processing multiple images
4. Use torch.cuda.amp.autocast for mixed precision:
   with torch.cuda.amp.autocast():
       result = pipe(...)
5. Set environment variable: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""