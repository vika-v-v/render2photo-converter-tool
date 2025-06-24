import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import random
import torch
import copy
import struct
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageFile, ImageFilter # Added ImageFilter here
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup # Specifically import this
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import get_scheduler # Generic scheduler

# Add imports for HuggingFace dataset access
from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv # Import for loading .env file

# Training Configuration
STARTING_EPOCH = 0  # Start from scratch or set to continue from a checkpoint
TOTAL_EPOCHS = 50
MAX_IMG_SIZE = 1024  # Maximum image size for resizing
BASE_IMAGE_SIZE = 1024  # Base size for data loading

# Enhanced prompts
PROMPT = "high quality photograph, photorealistic, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighting, consistent shadows, preserve as many details from the original image as possible, 8k, 4k, sharp focus"
NEGATIVE_PROMPT = "low quality, bad anatomy, bad hands, text, error, blurry, out of focus, low resolution, cropped, worst quality, jpeg artifacts, signature, watermark, distorted"

# Training hyperparameters - UPDATED FOR STABILITY
LEARNING_RATE = 2e-5  # Reduced from 2e-4 to prevent NaN losses
UNET_RANK = 32
TEXT_ENCODER_RANK = 8
USE_EMA = True
EMA_DECAY = 0.9995
GRADIENT_CHECKPOINTING = True
MIXED_PRECISION = "fp16" 
GRADIENT_ACCUMULATION_STEPS = 4
LR_SCHEDULER = "cosine_with_restarts"
LR_NUM_CYCLES = 3
LR_WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 0.5  # Reduced from 1.0 for more aggressive clipping

# Controls for the descaling effect (blur amount)
DESCALE_BLUR_RADIUS_MIN = 1.0
DESCALE_BLUR_RADIUS_MAX = 2.5
DESCALE_PROBABILITY = 0.5  # Probability of applying descaling

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

def get_random_sized_transform(size):
    """Get transform with a specific image size"""
    # Use a fixed size for all images in a batch to ensure compatibility
    return transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        # First make the image square via center crop
        transforms.CenterCrop(min(size, size)),
        # Then resize to the target size
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def load_office_render2photos_dataset(hf_token, cache_dir=None):
    """
    Load the office render2photos dataset from Hugging Face
    """
    login(token=hf_token)

    # Download the repository contents
    print("Downloading dataset files from Hugging Face...")
    repo_dir = snapshot_download(
        repo_id="vika-v/office-render2photos-pairs",
        repo_type="dataset", 
        token=hf_token,
        cache_dir=cache_dir # If None, uses default HF cache
    )

    # Get paths to renders and photos directories
    renders_dir = os.path.join(repo_dir, "renders")
    photos_dir = os.path.join(repo_dir, "photos")

    # Verify directories exist
    if not os.path.exists(renders_dir):
        raise ValueError(f"Renders directory not found at {renders_dir}")
    if not os.path.exists(photos_dir):
        raise ValueError(f"Photos directory not found at {photos_dir}")

    # Get list of files
    render_files = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    photo_files = sorted([f for f in os.listdir(photos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Found {len(render_files)} render files and {len(photo_files)} photo files")
    
    return renders_dir, photos_dir, render_files, photo_files

class EnhancedPairedImageDataset(Dataset):
    def __init__(self, renders_dir, photos_dir, render_files, photo_files, transform=None, 
                 img_size=BASE_IMAGE_SIZE, generate_descaled=True): # generate_descaled was unused
        self.renders_dir = renders_dir
        self.photos_dir = photos_dir
        self.render_files = render_files
        self.photo_files = photo_files
        # self.transform = transform # This transform was not used, get_random_sized_transform is used directly
        self.img_size = img_size # This was also somewhat redundant as BASE_IMAGE_SIZE is used
        # self.generate_descaled = generate_descaled # Unused in this class
        
        # Verify matching files
        assert len(self.render_files) == len(self.photo_files), "Number of renders and photos do not match"
        
        # Track valid indices to allow filtering of corrupted files
        self.valid_indices = list(range(len(self.render_files)))
        self.skip_indices = set()
        
        # Pre-check image files to filter corrupted ones
        self._validate_files()
        
    def _validate_files(self):
        """Pre-check image files to identify corrupted ones"""
        print(f"Validating image files in dataset...")
        for idx in range(len(self.render_files)):
            render_path = os.path.join(self.renders_dir, self.render_files[idx])
            photo_path = os.path.join(self.photos_dir, self.photo_files[idx])
            
            try:
                # Try to open the images, but don't process them fully
                with Image.open(render_path) as img:
                    img.verify()  # Verify it's a valid image
                with Image.open(photo_path) as img:
                    img.verify()  # Verify it's a valid image
            except (OSError, struct.error, IOError, Image.UnidentifiedImageError) as e: # Added UnidentifiedImageError
                print(f"⚠️ Corrupted or problematic image found: {render_path} or {photo_path}")
                print(f"   Error: {str(e)}")
                self.skip_indices.add(idx)
        
        # Update valid indices by removing corrupted ones
        self.valid_indices = [i for i in self.valid_indices if i not in self.skip_indices]
        print(f"Found {len(self.skip_indices)} corrupted image pairs. {len(self.valid_indices)} valid pairs remaining.")
        
    def _generate_descaled_image(self, img): # This method is part of EnhancedPairedImageDataset but only used by DescaledRenderDataset
        """Generate a descaled (blurred) version of the input image"""
        # Apply a random Gaussian blur to simulate descaling
        blur_radius = random.uniform(DESCALE_BLUR_RADIUS_MIN, DESCALE_BLUR_RADIUS_MAX)
        return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map the requested index to our valid indices
        real_idx = self.valid_indices[idx]
        
        render_path = os.path.join(self.renders_dir, self.render_files[real_idx])
        photo_path = os.path.join(self.photos_dir, self.photo_files[real_idx])
        
        try:
            render_img = Image.open(render_path).convert("RGB")
            photo_img = Image.open(photo_path).convert("RGB")
            
            # Resize large images to prevent CUDA OOM errors
            render_img = resize_if_needed(render_img)
            photo_img = resize_if_needed(photo_img)
            
            # Store original paths for visualization
            paths = {
                'render_path': render_path,
                'photo_path': photo_path,
                'render_filename': self.render_files[real_idx],
                'photo_filename': self.photo_files[real_idx],
                'is_descaled': False  # Default not descaled
            }
            
            # Use a consistent size for batching
            current_img_size = BASE_IMAGE_SIZE # Use BASE_IMAGE_SIZE consistently
            
            # Create transform with fixed size
            transform_fn = get_random_sized_transform(current_img_size) # Renamed for clarity
            render_img_transform = transform_fn(render_img)
            photo_img_transform = transform_fn(photo_img)
            
            return {'render': render_img_transform, 'photo': photo_img_transform, 'paths': paths}
            
        except (OSError, struct.error, IOError, Image.UnidentifiedImageError) as e: # Added UnidentifiedImageError
            # This is a fallback in case a corrupted file was missed during validation
            print(f"❌ Error loading file that passed validation: {render_path} or {photo_path}")
            print(f"   Error: {str(e)}")
            
            # Create a simple replacement with blank images
            blank_render = Image.new("RGB", (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), color=(0, 0, 0))
            blank_photo = Image.new("RGB", (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), color=(0, 0, 0))
            
            paths = {
                'render_path': render_path,
                'photo_path': photo_path,
                'render_filename': self.render_files[real_idx],
                'photo_filename': self.photo_files[real_idx],
                'is_descaled': False
            }
            
            transform_fn = get_random_sized_transform(BASE_IMAGE_SIZE) # Renamed for clarity
            blank_render_transformed = transform_fn(blank_render) # Renamed for clarity
            blank_photo_transformed = transform_fn(blank_photo)  # Renamed for clarity
                
            return {'render': blank_render_transformed, 'photo': blank_photo_transformed, 'paths': paths}


class DescaledRenderDataset(EnhancedPairedImageDataset):
    """Dataset that generates descaled (blurred) versions of renders on-the-fly"""
    def __getitem__(self, idx):
        # Map the requested index to our valid indices
        real_idx = self.valid_indices[idx]
        
        render_path = os.path.join(self.renders_dir, self.render_files[real_idx])
        photo_path = os.path.join(self.photos_dir, self.photo_files[real_idx])
        
        try:
            # Load both images
            render_img = Image.open(render_path).convert("RGB")
            photo_img = Image.open(photo_path).convert("RGB")
            
            # Resize large images to prevent CUDA OOM errors
            render_img = resize_if_needed(render_img)
            photo_img = resize_if_needed(photo_img)
            
            # Apply descaling (blur) to the render image only
            descaled_render = self._generate_descaled_image(render_img) # Calls parent method
            
            # Apply transforms to both images
            current_img_size = BASE_IMAGE_SIZE # Use BASE_IMAGE_SIZE consistently
            transform_fn = get_random_sized_transform(current_img_size) # Renamed for clarity
            descaled_render_transform = transform_fn(descaled_render)
            photo_transform = transform_fn(photo_img)
            
            # Store path information with descaled flag set to True
            paths = {
                'render_path': render_path,
                'photo_path': photo_path,
                'render_filename': self.render_files[real_idx],
                'photo_filename': self.photo_files[real_idx],
                'is_descaled': True  # Mark as descaled version
            }
            
            # Return with the target photo remaining unchanged
            return {
                'render': descaled_render_transform,  # Blurred render (input)
                'photo': photo_transform,            # Original photo (target)
                'paths': paths
            }
            
        except (OSError, struct.error, IOError, Image.UnidentifiedImageError) as e: # Added UnidentifiedImageError
            # Handle errors by creating blank images
            print(f"❌ Error creating descaled version for {render_path}: {str(e)}")
            
            blank_render = Image.new("RGB", (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), color=(0, 0, 0))
            blank_photo = Image.new("RGB", (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), color=(0, 0, 0))
            
            paths = {
                'render_path': render_path,
                'photo_path': photo_path,
                'render_filename': self.render_files[real_idx],
                'photo_filename': self.photo_files[real_idx],
                'is_descaled': True
            }
            
            transform_fn = get_random_sized_transform(BASE_IMAGE_SIZE) # Renamed for clarity
            blank_render_transformed = transform_fn(blank_render) # Renamed for clarity
            blank_photo_transformed = transform_fn(blank_photo) # Renamed for clarity
                
            return {'render': blank_render_transformed, 'photo': blank_photo_transformed, 'paths': paths}

class EMA:
    """
    Exponential Moving Average for model parameters
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # Detach the model from the current graph and ensure it's on the CPU for deepcopy
        # to avoid potential CUDA OOM during EMA model creation if original model is large.
        model_on_cpu = copy.deepcopy(model).cpu()
        self.model = model_on_cpu 
        self.model.requires_grad_(False)
        self.model.eval()
        
    def update(self, model):
        # Ensure the EMA model is on the same device as the training model before updating
        self.model.to(model.device)
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
                
    def state_dict(self):
        # Ensure model is on CPU before saving state_dict to avoid device mismatches later
        return self.model.cpu().state_dict()

def fine_tune_model(hf_token, model_name, 
                    output_dir, 
                    batch_size, num_epochs, 
                    learning_rate, save_freq,
                    test_images):
    """
    Enhanced fine-tuning for SDXL image-to-image model with LoRA for both UNet and text encoders,
    now using the Hugging Face dataset and with on-the-fly descaling.
    """
    # Determine torch_dtype based on mixed_precision setting
    if MIXED_PRECISION == "fp16":
        torch_dtype = torch.float16
    elif MIXED_PRECISION == "bf16":
        torch_dtype = torch.bfloat16
    else: # "no" or other
        torch_dtype = torch.float32

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
        log_with="tensorboard", # Optional: for logging
        project_dir=os.path.join(output_dir, "logs") # Optional: for logging
    )
    
    print("Loading SDXL pipeline components...")
    # Load pre-trained SDXL model
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, # Use determined dtype
        use_safetensors=True, # Recommended for faster loading and safety
        token=hf_token # Pass token for model loading
    )
    
    # Use the DDPM scheduler for fine-tuning (or DDPMScheduler as per original code)
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    
    # Get the components
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    tokenizer = pipeline.tokenizer
    tokenizer_2 = pipeline.tokenizer_2
    
    # Keep VAE in fp32 for stability during training, especially with mixed precision.
    vae.to(accelerator.device, dtype=torch.float32) 

    # Enable gradient checkpointing for memory efficiency
    if GRADIENT_CHECKPOINTING:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()
    
    # Configure LoRA for UNet
    unet_lora_config = LoraConfig(
        r=UNET_RANK,
        lora_alpha=UNET_RANK, 
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Configure LoRA for text encoders
    text_encoder_lora_config = LoraConfig(
        r=TEXT_ENCODER_RANK,
        lora_alpha=TEXT_ENCODER_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Apply LoRA to models
    unet = get_peft_model(unet, unet_lora_config)
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
    text_encoder_2 = get_peft_model(text_encoder_2, text_encoder_lora_config)
    
    if accelerator.is_main_process:
        unet.print_trainable_parameters()
        text_encoder.print_trainable_parameters()
        text_encoder_2.print_trainable_parameters()

    # Load checkpoint if continuing training
    if STARTING_EPOCH > 0:
        checkpoint_path = os.path.join(output_dir, f"lora-weights-epoch-{STARTING_EPOCH}")
        if os.path.exists(checkpoint_path):
            if accelerator.is_main_process:
                print(f"Loading checkpoint from: {checkpoint_path}")
            try:
                # Ensure adapter loading is done correctly for PEFT models
                unet.load_adapter(os.path.join(checkpoint_path, "unet"), "default")
                text_encoder.load_adapter(os.path.join(checkpoint_path, "text_encoder"), "default")
                text_encoder_2.load_adapter(os.path.join(checkpoint_path, "text_encoder_2"), "default")
                if accelerator.is_main_process:
                    print(f"Successfully loaded LoRA weights from {checkpoint_path}")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Error loading LoRA weights: {e}. Starting fresh training.")
        elif accelerator.is_main_process:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting fresh training.")
    
    # Set models to training mode (except VAE)
    unet.train()
    text_encoder.train()
    text_encoder_2.train()
    vae.requires_grad_(False) 
    vae.eval() 
    
    if USE_EMA:
        ema_unet = EMA(unet, decay=EMA_DECAY)
        ema_text_encoder = EMA(text_encoder, decay=EMA_DECAY) 
        ema_text_encoder_2 = EMA(text_encoder_2, decay=EMA_DECAY)
        if accelerator.is_main_process:
            print(f"Initialized EMA models with decay {EMA_DECAY}")
    
    # Load dataset from HuggingFace
    renders_dir, photos_dir, render_files, photo_files = load_office_render2photos_dataset(
        hf_token=hf_token,
        cache_dir=None # Use default Hugging Face cache directory
    )
    
    regular_dataset = EnhancedPairedImageDataset(
        renders_dir=renders_dir, photos_dir=photos_dir,
        render_files=render_files, photo_files=photo_files,
        img_size=BASE_IMAGE_SIZE
    )
    if accelerator.is_main_process:
        print(f"Created regular dataset with {len(regular_dataset)} image pairs")
    
    descaled_dataset = DescaledRenderDataset(
        renders_dir=renders_dir, photos_dir=photos_dir,
        render_files=render_files, photo_files=photo_files, 
        img_size=BASE_IMAGE_SIZE
    )
    if accelerator.is_main_process:
        print(f"Created descaled dataset with {len(descaled_dataset)} image pairs")
    
    combined_dataset = ConcatDataset([regular_dataset, descaled_dataset])
    if accelerator.is_main_process:
        print(f"Combined dataset has {len(combined_dataset)} image pairs")
    
    dataloader = DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True,
        collate_fn=lambda batch: {
            'render': torch.stack([item['render'] for item in batch]),
            'photo': torch.stack([item['photo'] for item in batch]),
            'paths': [item['paths'] for item in batch],
        }
    )
    
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    trainable_params.extend(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    trainable_params.extend(filter(lambda p: p.requires_grad, text_encoder_2.parameters()))
    
    optimizer = torch.optim.AdamW(
        trainable_params, lr=learning_rate,
        betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY,
    )
    
    num_update_steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION_STEPS
    effective_num_epochs = num_epochs - STARTING_EPOCH
    num_training_steps = num_update_steps_per_epoch * effective_num_epochs
    num_warmup_steps = int(LR_WARMUP_RATIO * num_training_steps)
    
    # Use get_cosine_schedule_with_warmup for "cosine_with_restarts"
    # and the generic get_scheduler for other types.
    if LR_SCHEDULER == "cosine_with_restarts":
        lr_scheduler = get_cosine_schedule_with_warmup( # Imported from diffusers.optimization
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * accelerator.num_processes,
            num_training_steps=num_training_steps * accelerator.num_processes,
            num_cycles=LR_NUM_CYCLES
        )
    else:
        lr_scheduler = get_scheduler( # Imported from transformers
            name=LR_SCHEDULER, # Use 'name' keyword argument for clarity
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * accelerator.num_processes,
            num_training_steps=num_training_steps * accelerator.num_processes
        )
    
    unet, text_encoder, text_encoder_2, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, text_encoder_2, optimizer, dataloader, lr_scheduler
    )
    vae.to(accelerator.device) # Ensure VAE is on the accelerator device

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    
    def process_vis_samples(epoch):
        if not accelerator.is_main_process: 
            return

        print(f"Processing visualization samples for epoch {epoch+1}...")
        
        vis_unet = accelerator.unwrap_model(ema_unet.model if USE_EMA else unet)
        vis_text_encoder = accelerator.unwrap_model(ema_text_encoder.model if USE_EMA else text_encoder)
        vis_text_encoder_2 = accelerator.unwrap_model(ema_text_encoder_2.model if USE_EMA else text_encoder_2)

        inference_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        pipeline_t = StableDiffusionXLImg2ImgPipeline(
            vae=accelerator.unwrap_model(vae), # Use unwrapped VAE
            text_encoder=vis_text_encoder,
            text_encoder_2=vis_text_encoder_2,
            tokenizer=tokenizer, 
            tokenizer_2=tokenizer_2, 
            unet=vis_unet,
            scheduler=inference_scheduler,
        ).to(accelerator.device) 
        pipeline_t.set_progress_bar_config(disable=True)

        pipeline_t.eval() # Set all components to eval mode
        
        sample_dir = os.path.join(output_dir, f"samples/epoch_{epoch+1}")
        os.makedirs(sample_dir, exist_ok=True)
        
        for i, image_path in enumerate(test_images):
            try:
                # print(f"Processing test image {i+1}/{len(test_images)}: {image_path}")
                if not os.path.isfile(image_path):
                    print(f"Warning: Test image file does not exist: {image_path}")
                    continue
                
                input_image = Image.open(image_path).convert("RGB")
                input_image_resized = resize_if_needed(input_image)
                filename = os.path.splitext(os.path.basename(image_path))[0]
                
                with torch.no_grad():
                    generated_image = pipeline_t(
                        prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT,
                        image=input_image_resized, strength=0.3, 
                        guidance_scale=7.5, num_inference_steps=30,
                    ).images[0]
                
                input_image_resized.save(os.path.join(sample_dir, f"{filename}_input.png"))
                generated_image.save(os.path.join(sample_dir, f"{filename}_output_epoch{epoch+1}.png"))
                # print(f"Completed processing and saved: {filename}")
            except Exception as e:
                print(f"Error processing visualization image {image_path}: {e}")
                import traceback
                traceback.print_exc()
        del pipeline_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def save_lora_checkpoint(epoch, is_final=False):
        if not accelerator.is_main_process: 
            return None

        unet_to_save = accelerator.unwrap_model(ema_unet.model if USE_EMA and not is_final else unet)
        text_encoder_to_save = accelerator.unwrap_model(ema_text_encoder.model if USE_EMA and not is_final else text_encoder)
        text_encoder_2_to_save = accelerator.unwrap_model(ema_text_encoder_2.model if USE_EMA and not is_final else text_encoder_2)
            
        save_dir_base = os.path.join(output_dir, "lora-weights-final" if is_final else f"lora-weights-epoch-{epoch+1}")
        os.makedirs(save_dir_base, exist_ok=True)
        
        unet_to_save.save_pretrained(os.path.join(save_dir_base, "unet"))
        text_encoder_to_save.save_pretrained(os.path.join(save_dir_base, "text_encoder"))
        text_encoder_2_to_save.save_pretrained(os.path.join(save_dir_base, "text_encoder_2"))
        
        print(f"Saved LoRA weights for all models at epoch {epoch+1} to {save_dir_base}")
        return save_dir_base
            
    global_step = 0
    for epoch in range(STARTING_EPOCH, num_epochs):
        unet.train()
        text_encoder.train()
        text_encoder_2.train()

        epoch_loss = 0.0
        skipped_batches = 0
        
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        for step, batch in enumerate(dataloader):
            try:
                render_images = batch['render'] 
                target_images = batch['photo']  
                
                if torch.isnan(render_images).any() or torch.isnan(target_images).any():
                    if accelerator.is_main_process:
                        print(f"WARNING: NaN values in input image tensors at epoch {epoch+1}, step {step}. Skipping batch.")
                    skipped_batches += 1
                    continue
                
                with accelerator.accumulate(unet, text_encoder, text_encoder_2): 
                    with torch.no_grad(): 
                        vae.to(target_images.device, dtype=torch.float32) 
                        target_latents = vae.encode(target_images.to(dtype=torch.float32)).latent_dist.sample()
                        target_latents = target_latents * vae.config.scaling_factor

                    noise = torch.randn_like(target_latents)
                    bsz = target_latents.shape[0]
                    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device).long()
                    noisy_latents = pipeline.scheduler.add_noise(target_latents, noise, timesteps)

                    prompt_ids = tokenizer(
                        [PROMPT] * bsz, padding="max_length",
                        max_length=tokenizer.model_max_length, truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)

                    prompt_ids_2 = tokenizer_2(
                        [PROMPT] * bsz, padding="max_length",
                        max_length=tokenizer_2.model_max_length, truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)
                    
                    # Get text embeddings from text_encoder (CLIPTextModel)
                    # Output is CLIPTextModelOutput, which has 'last_hidden_state' and 'pooler_output'
                    text_encoder_output = text_encoder(prompt_ids, output_hidden_states=True) # Ensure output_hidden_states for consistency if needed, though not strictly for LoRA target modules here
                    encoder_hidden_states = text_encoder_output.last_hidden_state # Use last_hidden_state
                    
                    # Get text embeddings from text_encoder_2 (CLIPTextModelWithProjection)
                    # Output is BaseModelOutputWithPoolingAndProjection if output_hidden_states=True
                    text_encoder_2_output = text_encoder_2(prompt_ids_2, output_hidden_states=True)
                    # For SDXL, 'add_text_embeds' (pooled_prompt_embeds) comes from 'text_embeds' attribute
                    pooled_prompt_embeds = text_encoder_2_output.text_embeds 
                    # Sequence embeddings from text_encoder_2 are often from the penultimate layer
                    encoder_hidden_states_2 = text_encoder_2_output.hidden_states[-2] 
                    
                    final_encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=-1)

                    add_text_embeds = pooled_prompt_embeds
                    original_size = (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE)
                    crops_coords_top_left = (0,0)
                    target_size = (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE)

                    add_time_ids_list = [torch.tensor(list(original_size + crops_coords_top_left + target_size))] * bsz
                    add_time_ids = torch.stack(add_time_ids_list).to(accelerator.device, dtype=final_encoder_hidden_states.dtype)

                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                    model_pred = unet(
                        noisy_latents, timesteps, 
                        encoder_hidden_states=final_encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs
                    ).sample
                    
                    if torch.isnan(model_pred).any():
                        if accelerator.is_main_process:
                            print(f"WARNING: NaN in model prediction at epoch {epoch+1}, step {step}. Skipping batch.")
                        skipped_batches += 1
                        if accelerator.sync_gradients: optimizer.zero_grad()
                        continue
                        
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    if torch.isnan(loss).any():
                        if accelerator.is_main_process:
                            print(f"WARNING: NaN loss detected at epoch {epoch+1}, step {step}. Skipping batch.")
                        skipped_batches += 1
                        if accelerator.sync_gradients: optimizer.zero_grad()
                        continue
                            
                    epoch_loss += loss.detach().item() 
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad() 
                        
                if USE_EMA and accelerator.sync_gradients: 
                    ema_unet.update(accelerator.unwrap_model(unet))
                    ema_text_encoder.update(accelerator.unwrap_model(text_encoder))
                    ema_text_encoder_2.update(accelerator.unwrap_model(text_encoder_2))
                
                progress_bar.update(1)
                current_loss = loss.detach().item() if 'loss' in locals() and loss is not None else float('nan')
                current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler else learning_rate
                progress_bar.set_postfix({"loss": current_loss, "lr": current_lr})
                
                if global_step % 100 == 0 and accelerator.is_main_process:
                     print(f"Epoch {epoch+1}, Step {step}/{len(dataloader)}, Global Step: {global_step}, Loss: {current_loss:.6f}, LR: {current_lr:.2e}")

                global_step += 1
                
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"ERROR in batch processing at epoch {epoch+1}, step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                skipped_batches += 1
                if 'optimizer' in locals() and optimizer is not None: optimizer.zero_grad()
                continue
            
        progress_bar.close()
        avg_epoch_loss = epoch_loss / (len(dataloader) - skipped_batches) if (len(dataloader) - skipped_batches) > 0 else float('nan')
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.6f}")
            print(f"Skipped batches: {skipped_batches}/{len(dataloader)} ({(skipped_batches/len(dataloader))*100:.2f}%)")
        
        if accelerator.is_main_process:
            if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                checkpoint_dir = save_lora_checkpoint(epoch) 
                if checkpoint_dir: 
                    print(f"Saved checkpoint to {checkpoint_dir}")
                if test_images and len(test_images) > 0:
                    process_vis_samples(epoch) 
    
    if accelerator.is_main_process:
        final_checkpoint_dir = save_lora_checkpoint(num_epochs - 1, is_final=True)
        if final_checkpoint_dir:
            print(f"Training complete. Final LoRA weights saved at {final_checkpoint_dir}")
    
    accelerator.wait_for_everyone() 

    if accelerator.is_main_process:
        final_unet = accelerator.unwrap_model(ema_unet.model if USE_EMA else unet)
        final_text_encoder = accelerator.unwrap_model(ema_text_encoder.model if USE_EMA else text_encoder)
        final_text_encoder_2 = accelerator.unwrap_model(ema_text_encoder_2.model if USE_EMA else text_encoder_2)
        final_vae = accelerator.unwrap_model(vae) # VAE was not part of EMA

        inference_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        final_pipeline = StableDiffusionXLImg2ImgPipeline(
            vae=final_vae.cpu(), 
            text_encoder=final_text_encoder.cpu(),
            text_encoder_2=final_text_encoder_2.cpu(),
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=final_unet.cpu(),
            scheduler=inference_scheduler,
        )
        print("Final pipeline created on CPU. Move to a device for inference.")
        return final_pipeline
    return None

def main():
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")

    if not hf_token:
        print("Error: HUGGING_FACE_TOKEN not found. Please set it in .env or environment.")
        return
    
    output_dir = 'hf_fine_tuned_render2photo_sdxl_lora_env_v3' 
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    
    test_image_folder = "task-images" 
    test_image_files = [
        "cgi-rendering-1.jpg", "cgi-rendering-2.jpg", "cgi-rendering-3.jpg",
        "cgi-rendering-4.jpg", "cgi-rendering-5.jpg", "KI_03.jpg",
        "KI_04.png", "KI_05.jpg", "KI_06.jpg", "simple_rendering_1.jpeg",
        "simple_rendering_2.jpeg", "FBR_IND_E10_E15_C01_H01_E11_300dpi.jpg",
        "FBR_SCA_B05_300dpi.jpg", "NB_EG_Cam_0008.jpg", "NB_EG_Cam_0014.jpg",
    ]
    
    test_images_full_paths = []
    if os.path.isdir(test_image_folder):
        for img_file in test_image_files:
            path = os.path.join(test_image_folder, img_file)
            if os.path.exists(path): test_images_full_paths.append(path)
            else: print(f"Warning: Test image not found: {path}")
    else:
        print(f"Warning: Test image folder '{test_image_folder}' not found.")

    if not test_images_full_paths:
        print("No valid test images found. Visualization will be skipped.")
    
    trained_pipeline = fine_tune_model(
        hf_token=hf_token, model_name=model_name, output_dir=output_dir,
        batch_size=1, # Further reduced batch_size for SDXL on potentially limited VRAM
        num_epochs=TOTAL_EPOCHS, learning_rate=LEARNING_RATE, 
        save_freq=5, test_images=test_images_full_paths
    )
    
    if trained_pipeline:
        print("Training completed successfully and pipeline returned!")
        print("\nExample usage of the trained pipeline (ensure it's moved to GPU if needed):")
        # ... (example usage print statements)
    else:
        print("Training process finished (or not on main process to return pipeline).")
    
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will run on CPU (very slow).")
    main()
