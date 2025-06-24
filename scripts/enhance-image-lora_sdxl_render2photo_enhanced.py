import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import json
import torch
import gc
import numpy as np
import cv2
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    DDPMScheduler
)
from safetensors.torch import load_file
from PIL import Image, ImageEnhance, ImageDraw, ImageFile
from tqdm import tqdm
from torchvision import transforms
import math
from peft import LoraConfig, PeftModel

# Configure PIL to handle truncated images more gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Configuration parameters
BASE_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH = "stabilityai/sdxl-vae"  # Improved VAE for better colors
LORA_DIR = "loras/final/lora-weights-epoch-10"
INPUT_DIR = "task-images"
OUTPUT_DIR = "processed-images"

# Enhanced prompts (from the training script)
PROMPT = "high quality photograph, photorealistic, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighting, consistent shadows, preserve as many details from the original image as possible, 8k, 4k, sharp focus"
FACE_PROMPT = "high quality photograph, photorealistic, masterpiece, perfect face details, realistic face features, high quality, detailed face, ultra realistic human face, perfect eyes, perfect skin texture, perfect facial proportions, clean render"
NEGATIVE_PROMPT = "low quality, bad anatomy, bad hands, text, error, blurry, out of focus, low resolution, cropped, worst quality, jpeg artifacts, signature, watermark, distorted"
FACE_NEGATIVE_PROMPT = "low quality, bad anatomy, distorted face, deformed face, disfigured face, unrealistic face, bad eyes, crossed eyes, misaligned eyes, bad nose, bad mouth, bad teeth, bad skin"

# Processing strengths
STRENGTH = 0.3 
FACE_STRENGTH = 0.35 
GUIDANCE_SCALE = 10.5
FACE_GUIDANCE_SCALE = 8.0 
RESIZE_LIMIT = 2048
SEED = 42

# LoRA configuration (aligned with training script)
UNET_RANK = 32  # From training script
TEXT_ENCODER_RANK = 8  # From training script
LORA_SCALE = 0.8  # For the main image processing
FACE_LORA_SCALE = 0.3  # Stronger LoRA influence for faces

# Inference parameters
NUM_STEPS = 100
FACE_NUM_STEPS = 200  # Fewer steps needed for smaller face regions
USE_CUSTOM_NOISE = True  # Enable custom noise initialization
MIXED_PRECISION = "fp16"  # Match training script setting
GRADIENT_CHECKPOINTING = True  # Memory optimization from training script

# Post-processing parameters
POST_PROCESS = True  # Enable post-processing
CONTRAST_FACTOR = 1.2  # Post-processing contrast enhancement factor
SHARPNESS_FACTOR = 1.7  # Post-processing sharpness enhancement factor
SATURATION_FACTOR = 1.1  # Post-processing saturation enhancement factor

# Face detection parameters
FACE_DETECTION_CONFIDENCE = 0.7  # Confidence threshold for face detection
FACE_PADDING_PERCENT = 30  # Percentage to expand face crop area
ENABLE_FACE_ENHANCEMENT = False  # Set to False to skip face processing
DEBUG_MODE = False  # Set to True to visualize face detection
USE_DNN_FACE_DETECTOR = True  # Use more robust DNN face detector
FACE_DETECTOR_MODEL_PATH = "models/opencv_face_detector_uint8.pb"  # Path to face detector model
FACE_DETECTOR_CONFIG_PATH = "models/opencv_face_detector.pbtxt"  # Path to face detector config

# New configuration parameters from training script
MAX_IMG_SIZE = 2048  # Maximum image size for resizing
USE_EMA = True  # Enable EMA for more stable results
EMA_DECAY = 0.9995  # EMA decay rate from training script

def resize_if_needed(img):
    """Resize image if either dimension exceeds max_size while preserving aspect ratio"""
    width, height = img.size
    
    # Check if resizing is needed
    if width <= MAX_IMG_SIZE and height <= MAX_IMG_SIZE:
        return img
    
    # Calculate new dimensions preserving aspect ratio
    if width > height:
        new_width = MAX_IMG_SIZE
        new_height = int(height * (MAX_IMG_SIZE / width))
    else:
        new_height = MAX_IMG_SIZE
        new_width = int(width * (MAX_IMG_SIZE / height))
    
    # Resize and return
    return img.resize((new_width, new_height), Image.LANCZOS)

def enhance_image(img):
    """
    Post-process image with color correction, contrast and sharpness enhancement.
    """
    if not POST_PROCESS:
        return img
        
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(CONTRAST_FACTOR)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(SHARPNESS_FACTOR)
    
    # Enhance color saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(SATURATION_FACTOR)
    
    return img

def detect_faces(image):
    """
    Detect faces in the image using either DNN-based detector (more robust) or
    OpenCV's traditional Haar Cascade classifier.
    Returns list of face rectangles (x, y, w, h)
    """
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    
    if USE_DNN_FACE_DETECTOR:
        # Check if model files exist
        if not (os.path.exists(FACE_DETECTOR_MODEL_PATH) and os.path.exists(FACE_DETECTOR_CONFIG_PATH)):
            print(f"Face detector model files not found. Falling back to Haar Cascade.")
            return detect_faces_haar(open_cv_image)
        
        try:
            # Load DNN face detector
            net = cv2.dnn.readNetFromTensorflow(FACE_DETECTOR_MODEL_PATH, FACE_DETECTOR_CONFIG_PATH)
            
            # Prepare image
            height, width = open_cv_image.shape[:2]
            blob = cv2.dnn.blobFromImage(open_cv_image, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            
            # Detect faces
            detections = net.forward()
            faces = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > FACE_DETECTION_CONFIDENCE:
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    # Ensure coordinates are within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:  # Valid face dimensions
                        faces.append((x1, y1, w, h))
            
            return faces
        except Exception as e:
            print(f"Error using DNN face detector: {str(e)}. Falling back to Haar Cascade.")
            return detect_faces_haar(open_cv_image)
    else:
        return detect_faces_haar(open_cv_image)

def detect_faces_haar(open_cv_image):
    """
    Detect faces using OpenCV's Haar Cascade classifier as a fallback method.
    """
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's pre-trained face detector
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Detect faces with adjusted parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # If no faces detected, try with a more aggressive approach
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
    return faces

def validate_face(face, image_width, image_height):
    """
    Validate detected face to reduce false positives
    """
    x, y, w, h = face
    
    # Check if coordinates are valid
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False
    
    # Check if the face detection makes sense geometrically
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False  # Face too wide or too tall
    
    # Check minimum face size (relative to image)
    min_dim_percent = 5  # Face should be at least 5% of image dimension
    if (w < image_width * min_dim_percent / 100) or (h < image_height * min_dim_percent / 100):
        return False  # Face too small
    
    # Max face size check
    max_dim_percent = 60  # Face shouldn't be more than 60% of image dimension
    if (w > image_width * max_dim_percent / 100) or (h > image_height * max_dim_percent / 100):
        return False  # Detected area too large to be a face
    
    return True

def expand_face_area(face, image_width, image_height, padding_percent=40):
    """
    Expand the face rectangle with padding, ensuring it stays within image bounds.
    """
    x, y, w, h = face
    
    # Calculate padding
    padding_x = int(w * padding_percent / 100)
    padding_y = int(h * padding_percent / 100)
    
    # Apply padding
    new_x = max(0, x - padding_x)
    new_y = max(0, y - padding_y)
    new_w = min(image_width - new_x, w + 2 * padding_x)
    new_h = min(image_height - new_y, h + 2 * padding_y)
    
    return (new_x, new_y, new_w, new_h)

class EMA:
    """
    Exponential Moving Average for model parameters (from training script)
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = model.copy()  # Simple copy for inference models
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
                
    def state_dict(self):
        return self.model.state_dict()

def process_face(pipeline, face_img, device):
    """
    Process a face image with optimized parameters for faces.
    """
    # Generate custom noise
    latent_height = face_img.height // 8
    latent_width = face_img.width // 8
    
    # Create a generator that matches the device
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(SEED)  # Use the global seed parameter for reproducibility
    
    noise = torch.randn(
        (1, pipeline.unet.config.in_channels, latent_height, latent_width),
        device=pipeline.device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        generator=generator
    )
    
    # Process the face with face-specific parameters
    face_result = pipeline(
        prompt=FACE_PROMPT,
        negative_prompt=FACE_NEGATIVE_PROMPT,
        image=face_img,
        strength=FACE_STRENGTH,
        guidance_scale=FACE_GUIDANCE_SCALE,
        num_inference_steps=FACE_NUM_STEPS,
        cross_attention_kwargs={"scale": FACE_LORA_SCALE},
        noise=noise
    ).images[0]
    
    # Apply post-processing
    face_result = enhance_image(face_result)
    
    return face_result

def paste_face(original_img, face_img, face_coords):
    """
    Paste the processed face back into the original image with improved blending.
    """
    x, y, w, h = face_coords
    
    # Resize face to match the crop area
    face_img_resized = face_img.resize((w, h), Image.LANCZOS)
    
    # Convert images to numpy arrays for processing
    original_array = np.array(original_img)
    face_array = np.array(face_img_resized)
    
    # Create a mask for smooth blending (feathered edges)
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Create an oval mask instead of elliptical for more natural face blending
    cv2.ellipse(mask, 
                center=(w // 2, h // 2),
                axes=(int(w * 0.4), int(h * 0.5)),  # Slightly smaller for more natural blend
                angle=0,
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=-1)
    
    # More aggressive feathering for smoother transitions
    blur_size = max(int(min(w, h) * 0.2) | 1, 11)  # Ensure odd number and reasonable size
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    # Convert mask to 3 channels
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    
    # Create a copy of the original image
    result_array = original_array.copy()
    
    # Select the region to replace
    roi = result_array[y:y+h, x:x+w]
    
    # Blend the processed face with the original using the mask
    if roi.shape == face_array.shape and roi.shape == mask_3ch.shape:
        result_array[y:y+h, x:x+w] = face_array * mask_3ch + roi * (1 - mask_3ch)
    else:
        print(f"Shape mismatch during blending: roi={roi.shape}, face={face_array.shape}, mask={mask_3ch.shape}")
    
    # Convert back to PIL
    return Image.fromarray(result_array)

def save_debug_image(image, faces, valid_faces, filename):
    """
    Save debug image with face bounding boxes. Valid faces in green, invalid in red.
    """
    if not DEBUG_MODE:
        return
        
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)
    
    # Draw all detected faces in red
    for face in faces:
        x, y, w, h = face
        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
    
    # Draw valid faces in green
    for face in valid_faces:
        x, y, w, h = face
        draw.rectangle([x, y, x+w, y+h], outline="green", width=3)
    
    debug_path = os.path.join(OUTPUT_DIR, f"debug_{filename}")
    debug_img.save(debug_path)
    print(f"Saved debug image to {debug_path}")

def load_lora_weights(pipeline, lora_dir, device):
    """
    Load LoRA weights into model components - supporting the new structure
    from the training script that uses separate LoRA adapters for UNet and
    text encoders.
    """
    print(f"Loading LoRA weights from {lora_dir}...")
    
    # First check if this is a combined adapter_model.safetensors or separate adapters
    adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")
    
    if os.path.exists(adapter_path):
        # Old style - single adapter file
        pipeline.load_lora_weights(lora_dir)
        print(f"Successfully loaded legacy LoRA weights from {adapter_path}")
        return pipeline
    
    # New style - separate adapters for UNet and text encoders
    unet_adapter_path = os.path.join(lora_dir, "unet")
    text_encoder_adapter_path = os.path.join(lora_dir, "text_encoder")
    text_encoder_2_adapter_path = os.path.join(lora_dir, "text_encoder_2")
    
    if not os.path.exists(unet_adapter_path):
        raise FileNotFoundError(f"Could not find UNet adapter at {unet_adapter_path}")
    
    # Define LoRA configs matching the training script
    unet_lora_config = LoraConfig(
        r=UNET_RANK,
        lora_alpha=2 * UNET_RANK,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # Main attention blocks
            "ff.net.0.proj", "ff.net.2",         # Feed-forward blocks
            "conv_in", "conv_out",               # Add convolution layers
            "time_emb.linear_1", "time_emb.linear_2"  # Add time embedding layers
        ],
        lora_dropout=0.05,
        bias="none"
    )
    
    text_encoder_lora_config = LoraConfig(
        r=TEXT_ENCODER_RANK,
        lora_alpha=2 * TEXT_ENCODER_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Load UNet adapter
    try:
        print(f"Loading UNet LoRA adapter from {unet_adapter_path}")
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet, 
            unet_adapter_path, 
            adapter_name="default"
        )
    except Exception as e:
        print(f"Error loading UNet adapter: {str(e)}")
        raise
    
    # Load text encoder adapter if it exists
    if os.path.exists(text_encoder_adapter_path):
        try:
            print(f"Loading Text Encoder LoRA adapter from {text_encoder_adapter_path}")
            pipeline.text_encoder = PeftModel.from_pretrained(
                pipeline.text_encoder, 
                text_encoder_adapter_path,
                adapter_name="default"
            )
        except Exception as e:
            print(f"Warning: Could not load Text Encoder adapter: {str(e)}")
    
    # Load text encoder 2 adapter if it exists
    if os.path.exists(text_encoder_2_adapter_path):
        try:
            print(f"Loading Text Encoder 2 LoRA adapter from {text_encoder_2_adapter_path}")
            pipeline.text_encoder_2 = PeftModel.from_pretrained(
                pipeline.text_encoder_2, 
                text_encoder_2_adapter_path,
                adapter_name="default"
            )
        except Exception as e:
            print(f"Warning: Could not load Text Encoder 2 adapter: {str(e)}")
    
    print("Successfully loaded all available LoRA adapters")
    return pipeline


def get_random_sized_transform(size):
    """Get transform with a specific image size (from training script)"""
    # Use a fixed size for all images in a batch to ensure compatibility
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        # First make the image square via center crop
        transforms.CenterCrop(min(size, size)),
        # Then resize to the target size
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def process_renders():
    """
    Process render images using Stable Diffusion XL with LoRA weights,
    custom scheduler, improved VAE and post-processing, with special
    handling for faces.
    """
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Configure PyTorch for memory optimization
    if device == "cuda":
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load the SDXL base model
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable safety checker to save memory
    )
    
    # Apply memory optimizations
    if device == "cuda":
        pipeline.enable_attention_slicing(slice_size=1)
        if GRADIENT_CHECKPOINTING:
            pipeline.unet.enable_gradient_checkpointing()
            
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Using xformers memory efficient attention")
        except:
            print("Xformers not available, using default attention mechanism")
    
    # Load custom VAE for better color handling
    print(f"Loading improved VAE from {VAE_PATH}...")
    try:
        vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipeline.vae = vae
        print("Successfully loaded improved VAE")
    except Exception as e:
        print(f"Error loading VAE: {str(e)}. Using default VAE.")
    
    # Configure scheduler for better sampling quality
    print("Configuring DPM++ 2M Karras scheduler...")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )
    
    # Load pipeline to device
    pipeline = pipeline.to(device)
    
    # Load LoRA weights using the new loading method
    try:
        pipeline = load_lora_weights(pipeline, LORA_DIR, device)
    except Exception as e:
        print(f"Error loading LoRA weights: {str(e)}")
        raise
    
    # Get all image files from the input directory
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not image_files:
        print(f"No image files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images one by one
    for image_file in tqdm(image_files):
        input_path = os.path.join(INPUT_DIR, image_file)
        
        # Construct output filename with -enhanced suffix
        file_name, file_ext = os.path.splitext(image_file)
        output_file = f"{file_name}-enhanced{file_ext}"
        output_path = os.path.join(OUTPUT_DIR, output_file)
        
        # Skip if already processed
        if os.path.exists(output_path):
            print(f"Skipping {image_file} (already processed)")
            continue
        
        try:
            # Clear GPU memory before processing each image
            if device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load and resize image
            original_image = Image.open(input_path).convert("RGB")
            
            # Save original dimensions for later
            original_width, original_height = original_image.size
            
            # Resize for processing if needed - use resize_if_needed from training script
            image = resize_if_needed(original_image)
            print(f"Processing {image_file} (resized from {original_width}x{original_height} to {image.width}x{image.height})")
            
            # Generate custom noise for main image
            latent_height = image.height // 8
            latent_width = image.width // 8
            
            # Create a generator that matches the device
            generator = torch.Generator(device=pipeline.device)
            generator.manual_seed(SEED)  # Fixed seed for reproducibility
            
            noise = torch.randn(
                (1, pipeline.unet.config.in_channels, latent_height, latent_width),
                device=pipeline.device,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                generator=generator
            )
            
            # Process whole image with the model
            result = pipeline(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                strength=STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_STEPS,
                cross_attention_kwargs={"scale": LORA_SCALE},
                noise=noise if USE_CUSTOM_NOISE else None,
            ).images[0]
            
            # Apply post-processing to the whole image
            result = enhance_image(result)
            
            # Process faces if enabled
            if ENABLE_FACE_ENHANCEMENT:
                # Detect faces in the result image
                faces = detect_faces(result)
                
                # Validate detected faces to filter out false positives
                valid_faces = [face for face in faces if validate_face(face, result.width, result.height)]
                
                # Save debug image if enabled
                if DEBUG_MODE:
                    save_debug_image(result, faces, valid_faces, output_file)
                
                if len(valid_faces) > 0:
                    print(f"Found {len(valid_faces)} valid faces in {image_file}")
                    
                    # Process each face individually
                    for i, face in enumerate(valid_faces):
                        # Expand face area with padding
                        expanded_face = expand_face_area(face, result.width, result.height, FACE_PADDING_PERCENT)
                        x, y, w, h = expanded_face
                        
                        # Extract face region
                        face_img = result.crop((x, y, x+w, y+h))
                        
                        # Skip very small faces
                        if face_img.width < 64 or face_img.height < 64:
                            print(f"Skipping face {i+1} - too small ({face_img.width}x{face_img.height})")
                            continue
                        
                        print(f"Processing face {i+1} ({face_img.width}x{face_img.height})...")
                        
                        # Process face region
                        enhanced_face = process_face(pipeline, face_img, device)
                        
                        # Paste enhanced face back to the result image
                        result = paste_face(result, enhanced_face, expanded_face)
                else:
                    print(f"No valid faces detected in {image_file}")
            else:
                print(f"Face enhancement disabled, skipping face detection for {image_file}")
            
            # Resize back to original dimensions if needed
            if original_width != result.width or original_height != result.height:
                result = result.resize((original_width, original_height), Image.LANCZOS)
            
            # Save the result
            result.save(output_path)
            print(f"Saved {output_file}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Clean up memory
    try:
        # Unload LoRA weights using either the original or adapted method
        if hasattr(pipeline, "unload_lora_weights"):
            pipeline.unload_lora_weights()
        else:
            # For PeftModel-style adapters, we just delete the pipeline
            pass
    except:
        print("Note: Could not explicitly unload LoRA weights, but continuing cleanup")
    
    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Processing complete. Results saved to {OUTPUT_DIR}")

def download_face_detection_models():
    """
    Download face detection models if they don't exist.
    """
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    
    # Define model URLs
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"
    config_url = "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    
    # Download model file if it doesn't exist
    if not os.path.exists(FACE_DETECTOR_MODEL_PATH):
        try:
            import urllib.request
            print(f"Downloading face detector model from {model_url}...")
            urllib.request.urlretrieve(model_url, FACE_DETECTOR_MODEL_PATH)
            print(f"Model downloaded to {FACE_DETECTOR_MODEL_PATH}")
        except Exception as e:
            print(f"Error downloading model: {str(e)}. Will use Haar Cascade instead.")
    
    # Download config file if it doesn't exist
    if not os.path.exists(FACE_DETECTOR_CONFIG_PATH):
        try:
            import urllib.request
            print(f"Downloading face detector config from {config_url}...")
            urllib.request.urlretrieve(config_url, FACE_DETECTOR_CONFIG_PATH)
            print(f"Config downloaded to {FACE_DETECTOR_CONFIG_PATH}")
        except Exception as e:
            print(f"Error downloading config: {str(e)}. Will use Haar Cascade instead.")

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Try to download face detection models if DNN detection is enabled
    if USE_DNN_FACE_DETECTOR:
        download_face_detection_models()
    
    # Save current settings to a JSON file in the same folder as OUTPUT_DIR
    settings = {
        # Model paths
        "BASE_MODEL_PATH": BASE_MODEL_PATH,
        "VAE_PATH": VAE_PATH,
        "LORA_DIR": LORA_DIR,
        
        # I/O directories
        "INPUT_DIR": INPUT_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        
        # Prompt settings
        "PROMPT": PROMPT,
        "FACE_PROMPT": FACE_PROMPT,
        "NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "FACE_NEGATIVE_PROMPT": FACE_NEGATIVE_PROMPT,
        
        # Processing parameters
        "STRENGTH": STRENGTH,
        "FACE_STRENGTH": FACE_STRENGTH,
        "GUIDANCE_SCALE": GUIDANCE_SCALE,
        "FACE_GUIDANCE_SCALE": FACE_GUIDANCE_SCALE,
        "RESIZE_LIMIT": RESIZE_LIMIT,
        "MAX_IMG_SIZE": MAX_IMG_SIZE,
        "SEED": SEED,
        
        # LoRA parameters
        "LORA_SCALE": LORA_SCALE,
        "FACE_LORA_SCALE": FACE_LORA_SCALE,
        "UNET_RANK": UNET_RANK,
        "TEXT_ENCODER_RANK": TEXT_ENCODER_RANK,
        
        # Inference settings
        "NUM_STEPS": NUM_STEPS,
        "FACE_NUM_STEPS": FACE_NUM_STEPS,
        "USE_CUSTOM_NOISE": USE_CUSTOM_NOISE,
        "MIXED_PRECISION": MIXED_PRECISION,
        "GRADIENT_CHECKPOINTING": GRADIENT_CHECKPOINTING,
        
        # EMA settings
        "USE_EMA": USE_EMA,
        "EMA_DECAY": EMA_DECAY,
        
        # Post-processing
        "POST_PROCESS": POST_PROCESS,
        "CONTRAST_FACTOR": CONTRAST_FACTOR,
        "SHARPNESS_FACTOR": SHARPNESS_FACTOR,
        "SATURATION_FACTOR": SATURATION_FACTOR,
        
        # Face detection settings
        "FACE_DETECTION_CONFIDENCE": FACE_DETECTION_CONFIDENCE,
        "FACE_PADDING_PERCENT": FACE_PADDING_PERCENT,
        "ENABLE_FACE_ENHANCEMENT": ENABLE_FACE_ENHANCEMENT,
        "DEBUG_MODE": DEBUG_MODE,
        "USE_DNN_FACE_DETECTOR": USE_DNN_FACE_DETECTOR
    }
    
    # Define the settings file path in the same folder as OUTPUT_DIR
    settings_path = os.path.join(OUTPUT_DIR, "settings.json")
    
    # Save settings to JSON file
    try:
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Saved settings to {settings_path}")
    except Exception as e:
        print(f"Error saving settings to {settings_path}: {str(e)}")
    
    # Process images
    process_renders()

if __name__ == "__main__":
    main()