import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import json
import torch
import gc
import numpy as np
import cv2
import io
import base64
import datetime
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
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import shutil
from flask_cors import CORS

# Configure PIL to handle truncated images more gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Default configuration parameters
DEFAULT_CONFIG = {
    # Model paths
    "BASE_MODEL_PATH": "stabilityai/stable-diffusion-xl-base-1.0",
    "VAE_PATH": "stabilityai/sdxl-vae",
    "LORA_DIR": "flora_sdxl_render2photo_enhanced/lora-weights-epoch-15",
    
    # Prompts
    "PROMPT": "high quality photograph, photorealistic, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighting, consistent shadows, preserve as many details from the original image as possible, 8k, 4k, sharp focus",
    "FACE_PROMPT": "high quality photograph, photorealistic, masterpiece, perfect face details, realistic face features, high quality, detailed face, ultra realistic human face, perfect eyes, perfect skin texture, perfect facial proportions, clean render",
    "NEGATIVE_PROMPT": "low quality, bad anatomy, bad hands, text, error, blurry, out of focus, low resolution, cropped, worst quality, jpeg artifacts, signature, watermark, distorted",
    "FACE_NEGATIVE_PROMPT": "low quality, bad anatomy, distorted face, deformed face, disfigured face, unrealistic face, bad eyes, crossed eyes, misaligned eyes, bad nose, bad mouth, bad teeth, bad skin",
    
    # Processing settings
    "STRENGTH": 0.4,
    "FACE_STRENGTH": 0.35,
    "GUIDANCE_SCALE": 6.0,
    "FACE_GUIDANCE_SCALE": 8.0,
    "SEED": 42,
    
    # LoRA configuration
    "UNET_RANK": 32,
    "TEXT_ENCODER_RANK": 8,
    "LORA_SCALE": 0.8,
    "FACE_LORA_SCALE": 0.3,
    
    # Inference parameters
    "NUM_STEPS": 400,
    "FACE_NUM_STEPS": 200,
    "USE_CUSTOM_NOISE": True,
    "MIXED_PRECISION": "fp16",
    "GRADIENT_CHECKPOINTING": True,
    
    # Post-processing parameters
    "POST_PROCESS": True,
    "CONTRAST_FACTOR": 1.2,
    "SHARPNESS_FACTOR": 1.7,
    "SATURATION_FACTOR": 1.1,
    
    # Face detection parameters
    "FACE_DETECTION_CONFIDENCE": 0.7,
    "FACE_PADDING_PERCENT": 30,
    "ENABLE_FACE_ENHANCEMENT": False,
    "DEBUG_MODE": True,
    "USE_DNN_FACE_DETECTOR": True,
    "FACE_DETECTOR_MODEL_PATH": "models/opencv_face_detector_uint8.pb",
    "FACE_DETECTOR_CONFIG_PATH": "models/opencv_face_detector.pbtxt",
    
    # New configuration parameters
    "MAX_IMG_SIZE": 2048,
    "USE_EMA": True,
    "EMA_DECAY": 0.9995,
    
    # GPU settings
    "CUDA_VISIBLE_DEVICES": "0"
}

# Global pipeline variable to avoid reloading model for each request
pipeline = None
device = None
initialized = False

def get_bool_param(data, param_name, default_value):
    """Helper function to convert string boolean parameters to actual booleans"""
    if param_name in data:
        value = data[param_name]
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ["true", "yes", "1", "y"]
    return default_value

def get_float_param(data, param_name, default_value):
    """Helper function to convert string float parameters to actual floats"""
    if param_name in data:
        try:
            return float(data[param_name])
        except (ValueError, TypeError):
            pass
    return default_value

def get_int_param(data, param_name, default_value):
    """Helper function to convert string int parameters to actual ints"""
    if param_name in data:
        try:
            return int(data[param_name])
        except (ValueError, TypeError):
            pass
    return default_value

def resize_if_needed(img, max_size):
    """Resize image if either dimension exceeds max_size while preserving aspect ratio"""
    width, height = img.size
    
    # Check if resizing is needed
    if width <= max_size and height <= max_size:
        return img
    
    # Calculate new dimensions preserving aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize and return
    return img.resize((new_width, new_height), Image.LANCZOS)

def enhance_image(img, config):
    """
    Post-process image with color correction, contrast and sharpness enhancement.
    """
    if not config["POST_PROCESS"]:
        return img
        
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(config["CONTRAST_FACTOR"])
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(config["SHARPNESS_FACTOR"])
    
    # Enhance color saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(config["SATURATION_FACTOR"])
    
    return img

def detect_faces(image, config):
    """
    Detect faces in the image using either DNN-based detector (more robust) or
    OpenCV's traditional Haar Cascade classifier.
    Returns list of face rectangles (x, y, w, h)
    """
    # Convert PIL image to OpenCV format
    open_cv_image = np.array(image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
    
    if config["USE_DNN_FACE_DETECTOR"]:
        # Check if model files exist
        if not (os.path.exists(config["FACE_DETECTOR_MODEL_PATH"]) and os.path.exists(config["FACE_DETECTOR_CONFIG_PATH"])):
            print(f"Face detector model files not found. Falling back to Haar Cascade.")
            return detect_faces_haar(open_cv_image)
        
        try:
            # Load DNN face detector
            net = cv2.dnn.readNetFromTensorflow(config["FACE_DETECTOR_MODEL_PATH"], config["FACE_DETECTOR_CONFIG_PATH"])
            
            # Prepare image
            height, width = open_cv_image.shape[:2]
            blob = cv2.dnn.blobFromImage(open_cv_image, 1.0, (300, 300), [104, 117, 123], False, False)
            net.setInput(blob)
            
            # Detect faces
            detections = net.forward()
            faces = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > config["FACE_DETECTION_CONFIDENCE"]:
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

def process_face(pipeline, face_img, device, config):
    """
    Process a face image with optimized parameters for faces.
    """
    # Generate custom noise
    latent_height = face_img.height // 8
    latent_width = face_img.width // 8
    
    # Create a generator that matches the device
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(config["SEED"])  # Use the global seed parameter for reproducibility
    
    noise = torch.randn(
        (1, pipeline.unet.config.in_channels, latent_height, latent_width),
        device=pipeline.device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        generator=generator
    )
    
    # Process the face with face-specific parameters
    face_result = pipeline(
        prompt=config["FACE_PROMPT"],
        negative_prompt=config["FACE_NEGATIVE_PROMPT"],
        image=face_img,
        strength=config["FACE_STRENGTH"],
        guidance_scale=config["FACE_GUIDANCE_SCALE"],
        num_inference_steps=config["FACE_NUM_STEPS"],
        cross_attention_kwargs={"scale": config["FACE_LORA_SCALE"]},
        noise=noise
    ).images[0]
    
    # Apply post-processing
    face_result = enhance_image(face_result, config)
    
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

def save_debug_image(image, faces, valid_faces, temp_dir, filename, debug_mode):
    """
    Save debug image with face bounding boxes. Valid faces in green, invalid in red.
    Return the path to the debug image.
    """
    if not debug_mode:
        return None
        
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
    
    debug_path = os.path.join(temp_dir, f"debug_{filename}")
    debug_img.save(debug_path)
    print(f"Saved debug image to {debug_path}")
    return debug_path

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
        r=DEFAULT_CONFIG["UNET_RANK"],
        lora_alpha=2 * DEFAULT_CONFIG["UNET_RANK"],
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
        r=DEFAULT_CONFIG["TEXT_ENCODER_RANK"],
        lora_alpha=2 * DEFAULT_CONFIG["TEXT_ENCODER_RANK"],
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

def initialize_model(config=None):
    """Initialize the model with the given configuration."""
    global pipeline, device, initialized
    
    if initialized:
        return
    
    if config is None:
        config = DEFAULT_CONFIG
    
    # Set CUDA device if specified
    if "CUDA_VISIBLE_DEVICES" in config:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Configure PyTorch for memory optimization
    if device == "cuda":
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load the SDXL base model
    print(f"Loading base model from {config['BASE_MODEL_PATH']}...")
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        config["BASE_MODEL_PATH"],
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None  # Disable safety checker to save memory
    )
    
    # Apply memory optimizations
    if device == "cuda":
        pipeline.enable_attention_slicing(slice_size=1)
        if config["GRADIENT_CHECKPOINTING"]:
            pipeline.unet.enable_gradient_checkpointing()
            
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Using xformers memory efficient attention")
        except:
            print("Xformers not available, using default attention mechanism")
    
    # Load custom VAE for better color handling
    print(f"Loading improved VAE from {config['VAE_PATH']}...")
    try:
        vae = AutoencoderKL.from_pretrained(
            config["VAE_PATH"],
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
    print("Scheduler configured successfully")
    
    # Load pipeline to device
    pipeline = pipeline.to(device)
    
    # Load LoRA weights using the new loading method
    try:
        pipeline = load_lora_weights(pipeline, config["LORA_DIR"], device)
        print("Successfully loaded LoRA weights")
    except Exception as e:
        print(f"Error loading LoRA weights: {str(e)}")
        raise
    
    initialized = True
    return pipeline

def download_face_detection_models(config=None):
    """
    Download face detection models if they don't exist.
    """
    if config is None:
        config = DEFAULT_CONFIG
        
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
    
    # Define model URLs
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"
    config_url = "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    
    # Download model file if it doesn't exist
    if not os.path.exists(config["FACE_DETECTOR_MODEL_PATH"]):
        try:
            import urllib.request
            print(f"Downloading face detector model from {model_url}...")
            urllib.request.urlretrieve(model_url, config["FACE_DETECTOR_MODEL_PATH"])
            print(f"Model downloaded to {config['FACE_DETECTOR_MODEL_PATH']}")
        except Exception as e:
            print(f"Error downloading model: {str(e)}. Will use Haar Cascade instead.")
    
    # Download config file if it doesn't exist
    if not os.path.exists(config["FACE_DETECTOR_CONFIG_PATH"]):
        try:
            import urllib.request
            print(f"Downloading face detector config from {config_url}...")
            urllib.request.urlretrieve(config_url, config["FACE_DETECTOR_CONFIG_PATH"])
            print(f"Config downloaded to {config['FACE_DETECTOR_CONFIG_PATH']}")
        except Exception as e:
            print(f"Error downloading config: {str(e)}. Will use Haar Cascade instead.")

def process_image(image, config):
    """Process an image with the given configuration."""
    global pipeline, device
    
    # Clear GPU memory before processing
    if device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save original dimensions for later
    original_width, original_height = image.size
    
    # Resize for processing if needed
    image = resize_if_needed(image, config["MAX_IMG_SIZE"])
    print(f"Processing image (resized from {original_width}x{original_height} to {image.width}x{image.height})")
    
    # Generate custom noise for main image
    latent_height = image.height // 8
    latent_width = image.width // 8
    
    # Create a generator that matches the device
    generator = torch.Generator(device=pipeline.device)
    generator.manual_seed(config["SEED"])  # Fixed seed for reproducibility
    
    noise = torch.randn(
        (1, pipeline.unet.config.in_channels, latent_height, latent_width),
        device=pipeline.device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        generator=generator
    )
    
    # Process whole image with the model
    result = pipeline(
        prompt=config["PROMPT"],
        negative_prompt=config["NEGATIVE_PROMPT"],
        image=image,
        strength=config["STRENGTH"],
        guidance_scale=config["GUIDANCE_SCALE"],
        num_inference_steps=config["NUM_STEPS"],
        cross_attention_kwargs={"scale": config["LORA_SCALE"]},
        noise=noise if config["USE_CUSTOM_NOISE"] else None,
    ).images[0]
    
    # Apply post-processing to the whole image
    result = enhance_image(result, config)
    
    # Create a temporary directory for debug images
    temp_dir = tempfile.mkdtemp()
    debug_image_path = None
    
    # Process faces if enabled
    if config["ENABLE_FACE_ENHANCEMENT"]:
        # Detect faces in the result image
        faces = detect_faces(result, config)
        
        # Validate detected faces to filter out false positives
        valid_faces = [face for face in faces if validate_face(face, result.width, result.height)]
        
        # Save debug image if enabled
        if config["DEBUG_MODE"]:
            debug_image_path = save_debug_image(result, faces, valid_faces, temp_dir, "debug.jpg", config["DEBUG_MODE"])
        
        if len(valid_faces) > 0:
            print(f"Found {len(valid_faces)} valid faces")
            
            # Process each face individually
            for i, face in enumerate(valid_faces):
                # Expand face area with padding
                expanded_face = expand_face_area(face, result.width, result.height, config["FACE_PADDING_PERCENT"])
                x, y, w, h = expanded_face
                
                # Extract face region
                face_img = result.crop((x, y, x+w, y+h))
                
                # Skip very small faces
                if face_img.width < 64 or face_img.height < 64:
                    print(f"Skipping face {i+1} - too small ({face_img.width}x{face_img.height})")
                    continue
                
                print(f"Processing face {i+1} ({face_img.width}x{face_img.height})...")
                
                # Process face region
                enhanced_face = process_face(pipeline, face_img, device, config)
                
                # Paste enhanced face back to the result image
                result = paste_face(result, enhanced_face, expanded_face)
        else:
            print("No valid faces detected")
    else:
        print(f"Face enhancement disabled, skipping face detection")
    
    # Resize back to original dimensions if needed
    if original_width != result.width or original_height != result.height:
        result = result.resize((original_width, original_height), Image.LANCZOS)
    
    return result, debug_image_path, temp_dir

def cleanup_resources():
    """Cleanup GPU resources."""
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

# API Routes
@app.route('/process', methods=['POST'])
def process_image_endpoint():
    """
    API endpoint to process an image using Stable Diffusion.
    
    Request should contain:
    - 'image': File upload
    - Configuration parameters (optional) as form data
    
    Returns:
    - JSON response with processed image (Base64 encoded) or error message
    """
    try:
        # Check if an image file was uploaded
        if 'Image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['Image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        
        # Save the uploaded file
        input_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(input_path)
        
        # Build configuration from request parameters
        config = DEFAULT_CONFIG.copy()
        
        # Update configuration with request parameters
        for key in DEFAULT_CONFIG.keys():
            if key in request.form:
                # Handle different parameter types
                if isinstance(DEFAULT_CONFIG[key], bool):
                    config[key] = get_bool_param(request.form, key, DEFAULT_CONFIG[key])
                elif isinstance(DEFAULT_CONFIG[key], float):
                    config[key] = get_float_param(request.form, key, DEFAULT_CONFIG[key])
                elif isinstance(DEFAULT_CONFIG[key], int):
                    config[key] = get_int_param(request.form, key, DEFAULT_CONFIG[key])
                else:
                    config[key] = request.form[key]
        
        # Print received parameters for debugging
        print(f"Received parameters:")
        for key in request.form:
            print(f"  {key}: {request.form[key]}")
        
        # Format selection - default is JPEG
        output_format = request.form.get('output_format', 'JPEG').upper()
        if output_format not in ['JPEG', 'PNG', 'WEBP']:
            output_format = 'JPEG'
        
        # Quality setting for JPEG
        quality = int(request.form.get('quality', 95))
        
        # Download face detection models if needed
        if config["ENABLE_FACE_ENHANCEMENT"] and config["USE_DNN_FACE_DETECTOR"]:
            download_face_detection_models(config)
        
        # Initialize the model if not already initialized
        if not initialized:
            initialize_model(config)
        
        # Open and process the image
        try:
            image = Image.open(input_path).convert("RGB")
            processed_image, debug_image_path, processing_temp_dir = process_image(image, config)
            
            # Save the result to a buffer
            output_buffer = io.BytesIO()
            processed_image.save(output_buffer, format=output_format, quality=quality)
            output_buffer.seek(0)
            
            # Encode as base64
            encoded_image = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            # Include debug image if available
            debug_image_data = None
            if debug_image_path and os.path.exists(debug_image_path):
                with open(debug_image_path, 'rb') as f:
                    debug_image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Clean up temporary directories
            shutil.rmtree(temp_dir)
            if processing_temp_dir and os.path.exists(processing_temp_dir):
                shutil.rmtree(processing_temp_dir)
            
            # Build response
            response = {
                'success': True,
                'image': encoded_image,
                'format': output_format.lower()
            }
            
            if debug_image_data:
                response['debug_image'] = debug_image_data
                
            return jsonify(response)
            
        except Exception as e:
            # Clean up temporary directories
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return jsonify({'error': f'Image processing error: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/process-file', methods=['POST'])
def process_file_endpoint():
    """
    API endpoint to process an image and return a file download.
    Similar to /process but returns the actual file instead of base64 encoding.
    """
    try:
        # Check if an image file was uploaded
        if 'Image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['Image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        
        # Save the uploaded file
        input_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(input_path)
        
        # Build configuration from request parameters
        config = DEFAULT_CONFIG.copy()
        
        # Update configuration with request parameters
        for key in DEFAULT_CONFIG.keys():
            if key in request.form:
                # Handle different parameter types
                if isinstance(DEFAULT_CONFIG[key], bool):
                    config[key] = get_bool_param(request.form, key, DEFAULT_CONFIG[key])
                elif isinstance(DEFAULT_CONFIG[key], float):
                    config[key] = get_float_param(request.form, key, DEFAULT_CONFIG[key])
                elif isinstance(DEFAULT_CONFIG[key], int):
                    config[key] = get_int_param(request.form, key, DEFAULT_CONFIG[key])
                else:
                    config[key] = request.form[key]
        
        # Print received parameters for debugging
        print(f"Received parameters for process-file:")
        for key in request.form:
            print(f"  {key}: {request.form[key]}")
            
        # Format selection - default is JPEG
        output_format = request.form.get('output_format', 'JPEG').upper()
        if output_format not in ['JPEG', 'PNG', 'WEBP']:
            output_format = 'JPEG'
        
        # Quality setting for JPEG
        quality = int(request.form.get('quality', 95))
        
        # Download face detection models if needed
        if config["ENABLE_FACE_ENHANCEMENT"] and config["USE_DNN_FACE_DETECTOR"]:
            download_face_detection_models(config)
        
        # Initialize the model if not already initialized
        if not initialized:
            initialize_model(config)
        
        # Open and process the image
        try:
            image = Image.open(input_path).convert("RGB")
            processed_image, debug_image_path, processing_temp_dir = process_image(image, config)
            
            # Save the result to a temporary file
            file_name, file_ext = os.path.splitext(file.filename)
            output_ext = f".{output_format.lower()}"
            output_filename = f"{file_name}-processed{output_ext}"
            output_path = os.path.join(temp_dir, output_filename)
            processed_image.save(output_path, format=output_format, quality=quality)
            
            # Return the file
            response = send_file(
                output_path,
                mimetype=f'image/{output_format.lower()}',
                as_attachment=True,
                download_name=output_filename
            )
            
            # Set a callback to clean up after the response is sent
            @response.call_on_close
            def cleanup():
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                if processing_temp_dir and os.path.exists(processing_temp_dir):
                    shutil.rmtree(processing_temp_dir)
            
            return response
            
        except Exception as e:
            # Clean up temporary directories
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            return jsonify({'error': f'Image processing error: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return jsonify({
        'status': 'ok',
        'initialized': initialized,
        'device': str(device)
    })

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """
    Simple test endpoint that works with any HTTP method.
    Useful for testing basic connectivity from clients like Postman.
    """
    return jsonify({
        'message': 'Connection successful',
        'method': request.method,
        'timestamp': str(datetime.datetime.now())
    })

@app.route('/settings', methods=['GET'])
def get_settings():
    """
    Return the current default settings.
    """
    return jsonify(DEFAULT_CONFIG)

@app.route('/check-params', methods=['POST'])
def check_params():
    """
    Debugging endpoint to check what parameters are being received without processing an image.
    Useful for testing parameter passing from clients like Postman.
    """
    # Collect all parameters from the request
    received_params = {}
    
    # Get parameters from form data
    for key in request.form:
        received_params[key] = request.form[key]
    
    # Check if any files were uploaded
    files = {}
    for key in request.files:
        files[key] = request.files[key].filename
    
    # Return summary of received data
    return jsonify({
        'received_params': received_params,
        'files': files,
        'content_type': request.content_type,
        'method': request.method,
        'parsed_parameters': {
            # Show how parameters would be interpreted
            key: get_bool_param(request.form, key, DEFAULT_CONFIG[key]) 
            if isinstance(DEFAULT_CONFIG.get(key), bool)
            else get_float_param(request.form, key, DEFAULT_CONFIG[key]) 
            if isinstance(DEFAULT_CONFIG.get(key), float)
            else get_int_param(request.form, key, DEFAULT_CONFIG[key])
            if isinstance(DEFAULT_CONFIG.get(key), int)
            else request.form.get(key)
            for key in request.form if key in DEFAULT_CONFIG
        }
    })

if __name__ == '__main__':
    # Try to download face detection models on startup
    download_face_detection_models()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)