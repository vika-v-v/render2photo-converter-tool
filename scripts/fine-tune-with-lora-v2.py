import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import random
import torch
import copy
import struct
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import get_scheduler

# Configure PIL to handle truncated images more gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Training Configuration
STARTING_EPOCH = 0  # Start from scratch or set to continue from a checkpoint
TOTAL_EPOCHS = 15
MAX_IMG_SIZE = 1024  # Maximum image size for resizing
BASE_IMAGE_SIZE = 1024  # Base size for data loading

# Enhanced prompts
PROMPT = "high quality photograph, photorealistic, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighting, consistent shadows, preserve as many details from the original image as possible, 8k, 4k, sharp focus"
NEGATIVE_PROMPT = "low quality, bad anatomy, bad hands, text, error, blurry, out of focus, low resolution, cropped, worst quality, jpeg artifacts, signature, watermark, distorted"

# Training hyperparameters
LEARNING_RATE = 2e-4
UNET_RANK = 32  # Increased from 16
TEXT_ENCODER_RANK = 8
USE_EMA = True  # Enable EMA for more stable results
EMA_DECAY = 0.9995
GRADIENT_CHECKPOINTING = True
MIXED_PRECISION = "fp16"
GRADIENT_ACCUMULATION_STEPS = 4
LR_SCHEDULER = "cosine_with_restarts"  # "linear", "cosine", "cosine_with_restarts"
LR_NUM_CYCLES = 3  # For cosine with restarts
LR_WARMUP_RATIO = 0.1  # Percentage of total steps for warmup
WEIGHT_DECAY = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0

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
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        # First make the image square via center crop
        transforms.CenterCrop(min(size, size)),
        # Then resize to the target size
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=BASE_IMAGE_SIZE):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.render_dir = os.path.join(root_dir, "renders")
        self.photo_dir = os.path.join(root_dir, "photos")
        
        # Get all the render files
        self.render_files = sorted([f for f in os.listdir(self.render_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.photo_files = sorted([f for f in os.listdir(self.photo_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Verify matching files
        assert len(self.render_files) == len(self.photo_files), "Number of renders and photos do not match"
        
        # Track valid indices to allow filtering of corrupted files
        self.valid_indices = list(range(len(self.render_files)))
        self.skip_indices = set()
        
        # Pre-check image files to filter corrupted ones
        self._validate_files()
        
    def _validate_files(self):
        """Pre-check image files to identify corrupted ones"""
        print(f"Validating image files in {self.root_dir}...")
        for idx in range(len(self.render_files)):
            render_path = os.path.join(self.render_dir, self.render_files[idx])
            photo_path = os.path.join(self.photo_dir, self.photo_files[idx])
            
            try:
                # Try to open the images, but don't process them fully
                with Image.open(render_path) as img:
                    img.verify()  # Verify it's a valid image
                with Image.open(photo_path) as img:
                    img.verify()  # Verify it's a valid image
            except (OSError, struct.error, IOError) as e:
                print(f"⚠️ Corrupted or problematic image found: {render_path} or {photo_path}")
                print(f"   Error: {str(e)}")
                self.skip_indices.add(idx)
        
        # Update valid indices by removing corrupted ones
        self.valid_indices = [i for i in self.valid_indices if i not in self.skip_indices]
        print(f"Found {len(self.skip_indices)} corrupted image pairs. {len(self.valid_indices)} valid pairs remaining.")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map the requested index to our valid indices
        real_idx = self.valid_indices[idx]
        
        render_path = os.path.join(self.render_dir, self.render_files[real_idx])
        photo_path = os.path.join(self.photo_dir, self.photo_files[real_idx])
        
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
                'photo_filename': self.photo_files[real_idx]
            }
            
            # Use a consistent size for batching - we'll disable dynamic resolution for now
            # to avoid the tensor shape issues
            img_size = BASE_IMAGE_SIZE
            
            # Create transform with fixed size
            transform = get_random_sized_transform(img_size)
            render_img = transform(render_img)
            photo_img = transform(photo_img)
            
            return {'render': render_img, 'photo': photo_img, 'paths': paths}
            
        except (OSError, struct.error, IOError) as e:
            # This is a fallback in case a corrupted file was missed during validation
            print(f"❌ Error loading file that passed validation: {render_path} or {photo_path}")
            print(f"   Error: {str(e)}")
            
            # Create a simple replacement with blank images
            # For training, it's better to return something than to crash
            blank_render = Image.new("RGB", (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), color=(0, 0, 0))
            blank_photo = Image.new("RGB", (BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), color=(0, 0, 0))
            
            paths = {
                'render_path': render_path,
                'photo_path': photo_path,
                'render_filename': self.render_files[real_idx],
                'photo_filename': self.photo_files[real_idx]
            }
            
            transform = get_random_sized_transform(BASE_IMAGE_SIZE)
            blank_render = transform(blank_render)
            blank_photo = transform(blank_photo) 
                
            return {'render': blank_render, 'photo': blank_photo, 'paths': paths}

class EMA:
    """
    Exponential Moving Average for model parameters
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.model.eval()
        
    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
                
    def state_dict(self):
        return self.model.state_dict()

def fine_tune_model(dataset_paths, model_name, 
                    output_dir, 
                    batch_size, num_epochs, 
                    learning_rate, save_freq,
                    test_images):
    """
    Enhanced fine-tuning for SDXL image-to-image model with LoRA for both UNet and text encoders,
    multi-resolution training, EMA, and more advanced optimization techniques.
    """
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision=MIXED_PRECISION,
    )
    
    print("Loading SDXL pipeline components...")
    # Load pre-trained SDXL model
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if MIXED_PRECISION == "fp16" else torch.float32
    )
    
    # Use the DDPM scheduler for fine-tuning
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # Get the components
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    
    # Enable gradient checkpointing for memory efficiency
    if GRADIENT_CHECKPOINTING:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()
    
    # Move all model components to the same device as accelerator
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    
    # Configure LoRA for UNet with expanded target modules
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
    
    # Configure LoRA for text encoders
    text_encoder_lora_config = LoraConfig(
        r=TEXT_ENCODER_RANK,
        lora_alpha=2 * TEXT_ENCODER_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Apply LoRA to models
    unet = get_peft_model(unet, unet_lora_config)
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
    text_encoder_2 = get_peft_model(text_encoder_2, text_encoder_lora_config)
    
    # Load checkpoint if continuing training
    if STARTING_EPOCH > 0:
        checkpoint_path = os.path.join(output_dir, f"lora-weights-epoch-{STARTING_EPOCH}")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load UNet LoRA weights
            unet_adapter_path = os.path.join(checkpoint_path, "unet")
            if os.path.exists(unet_adapter_path):
                unet.load_adapter(unet_adapter_path, adapter_name="default")
                print(f"Loaded UNet LoRA weights from {unet_adapter_path}")
            
            # Load text encoder LoRA weights
            text_encoder_adapter_path = os.path.join(checkpoint_path, "text_encoder")
            if os.path.exists(text_encoder_adapter_path):
                text_encoder.load_adapter(text_encoder_adapter_path, adapter_name="default")
                print(f"Loaded Text Encoder LoRA weights from {text_encoder_adapter_path}")
            
            # Load text encoder 2 LoRA weights
            text_encoder_2_adapter_path = os.path.join(checkpoint_path, "text_encoder_2")
            if os.path.exists(text_encoder_2_adapter_path):
                text_encoder_2.load_adapter(text_encoder_2_adapter_path, adapter_name="default")
                print(f"Loaded Text Encoder 2 LoRA weights from {text_encoder_2_adapter_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting fresh training.")
    
    # Prepare models for training/inference
    unet.train()
    text_encoder.train()
    text_encoder_2.train()
    vae.requires_grad_(False)
    vae.eval()
    
    # Create EMA model if enabled
    if USE_EMA:
        ema_unet = EMA(unet, decay=EMA_DECAY)
        ema_text_encoder = EMA(text_encoder, decay=EMA_DECAY)
        ema_text_encoder_2 = EMA(text_encoder_2, decay=EMA_DECAY)
        print(f"Initialized EMA models with decay {EMA_DECAY}")
    
    # Create datasets and data loaders for each path
    all_datasets = []
    for path in dataset_paths:
        dataset = PairedImageDataset(root_dir=path, transform=None, img_size=BASE_IMAGE_SIZE)
        all_datasets.append(dataset)
        print(f"Loaded {len(dataset)} image pairs from {path}")
    
    # Combine datasets
    if len(all_datasets) > 1:
        combined_dataset = ConcatDataset(all_datasets)
    else:
        combined_dataset = all_datasets[0]
    
    # Create the main dataloader for training
    dataloader = DataLoader(
        combined_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        # Add a custom collate function to handle variable sized tensors
        collate_fn=lambda batch: {
            'render': torch.stack([item['render'] for item in batch]),
            'photo': torch.stack([item['photo'] for item in batch]),
            'paths': [item['paths'] for item in batch],
            # No need to include 'size' in the batch as all tensors are already resized
        }
    )
    
    # Set up optimizer with all trainable parameters
    all_params = []
    all_params.extend(unet.parameters())
    all_params.extend(text_encoder.parameters())
    all_params.extend(text_encoder_2.parameters())
    
    optimizer = torch.optim.AdamW(
        all_params,
        lr=learning_rate,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Calculate number of training steps and warmup steps
    remaining_epochs = num_epochs - STARTING_EPOCH
    num_training_steps = len(dataloader) * remaining_epochs // GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(LR_WARMUP_RATIO * num_training_steps)
    
    # Set up learning rate scheduler
    if LR_SCHEDULER == "cosine_with_restarts":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=LR_NUM_CYCLES,
        )
    else:
        lr_scheduler = get_scheduler(
            LR_SCHEDULER,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    # Prepare model, optimizer, dataloader and scheduler for accelerator
    unet, text_encoder, text_encoder_2, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, text_encoder_2, optimizer, dataloader, lr_scheduler
    )
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    
    def process_vis_samples(epoch):
        print(f"Processing visualization samples for epoch {epoch+1}...")
        
        # Create a temporary pipeline for inference
        # Use EMA models if available
        if USE_EMA:
            inference_unet = accelerator.unwrap_model(ema_unet.model)
            inference_text_encoder = accelerator.unwrap_model(ema_text_encoder.model)
            inference_text_encoder_2 = accelerator.unwrap_model(ema_text_encoder_2.model)
        else:
            inference_unet = accelerator.unwrap_model(unet)
            inference_text_encoder = accelerator.unwrap_model(text_encoder)
            inference_text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
        
        # For inference, use a faster scheduler
        inference_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        pipeline_t = StableDiffusionXLImg2ImgPipeline(
            vae=vae,
            text_encoder=inference_text_encoder,
            text_encoder_2=inference_text_encoder_2,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            unet=inference_unet,
            scheduler=inference_scheduler,
        ).to(accelerator.device)
        
        # Make sure pipeline is in eval mode
        pipeline_t.unet.eval()
        pipeline_t.text_encoder.eval()
        pipeline_t.text_encoder_2.eval()
        
        # Create output directory for this epoch
        sample_dir = os.path.join(output_dir, f"samples/epoch_{epoch+1}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Process each test image directly from the file paths
        for i, image_path in enumerate(test_images):
            try:
                print(f"Processing test image {i+1}/{len(test_images)}: {image_path}")
                
                # Check if file exists
                if not os.path.isfile(image_path):
                    print(f"Warning: File does not exist: {image_path}")
                    continue
                
                # Load the image directly
                render_img = resize_if_needed(Image.open(image_path).convert("RGB"))
                
                # Get the filename without extension
                filename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Generate output with the model
                with torch.no_grad():
                    # For inference with negative prompt
                    model_output = pipeline_t(
                        prompt=PROMPT,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=render_img,
                        strength=0.3,
                        guidance_scale=7.5,
                        num_inference_steps=55,
                    ).images[0]
                
                # Save the input and output images
                render_img.save(os.path.join(sample_dir, f"{filename}_input.png"))
                model_output.save(os.path.join(sample_dir, f"{filename}_output.png"))
                
                print(f"Completed processing: {filename}")
            except Exception as e:
                print(f"Error processing visualization image {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
    def save_lora_checkpoint(epoch, is_final=False):
        """Save LoRA weights for all models"""
        # Get unwrapped models
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
        unwrapped_text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
            
        # Get EMA models if enabled
        if USE_EMA and not is_final:
            save_unet = ema_unet.model
            save_text_encoder = ema_text_encoder.model
            save_text_encoder_2 = ema_text_encoder_2.model
        else:
            save_unet = unwrapped_unet
            save_text_encoder = unwrapped_text_encoder
            save_text_encoder_2 = unwrapped_text_encoder_2
            
        # Determine save path
        if is_final:
            save_dir = os.path.join(output_dir, "lora-weights-final")
        else:
            save_dir = os.path.join(output_dir, f"lora-weights-epoch-{epoch+1}")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Save UNet LoRA weights
        unet_save_dir = os.path.join(save_dir, "unet")
        os.makedirs(unet_save_dir, exist_ok=True)
        save_unet.save_pretrained(unet_save_dir)
        
        # Save text encoder LoRA weights
        text_encoder_save_dir = os.path.join(save_dir, "text_encoder")
        os.makedirs(text_encoder_save_dir, exist_ok=True)
        save_text_encoder.save_pretrained(text_encoder_save_dir)
        
        # Save text encoder 2 LoRA weights
        text_encoder_2_save_dir = os.path.join(save_dir, "text_encoder_2")
        os.makedirs(text_encoder_2_save_dir, exist_ok=True)
        save_text_encoder_2.save_pretrained(text_encoder_2_save_dir)
        
        print(f"Saved LoRA weights for all models at epoch {epoch+1}")
        
        return save_dir
            
    # Training loop
    global_step = 0
    for epoch in range(STARTING_EPOCH, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        for step, batch in enumerate(dataloader):
            # Get the input images
            render_images = batch['render'].to(accelerator.device)
            target_images = batch['photo'].to(accelerator.device)
            
            # Encode the images with VAE (using proper precision)
            with torch.no_grad():
                # Use autocast to match precision
                with torch.autocast("cuda", dtype=torch.float16, enabled=MIXED_PRECISION == "fp16"):
                    # Scale down the latents
                    latents_render = vae.encode(render_images).latent_dist.sample() * 0.18215
                    latents_target = vae.encode(target_images).latent_dist.sample() * 0.18215
            
            # Sample noise with biased timesteps (focus training on details)
            noise = torch.randn_like(latents_target)
            
            # Sample timesteps with bias toward lower noise (better detail learning)
            batch_size = latents_target.shape[0]
            timesteps = torch.tensor(
                [pipeline.scheduler.config.num_train_timesteps * 
                 (1 - np.sqrt(np.random.uniform(0.0, 1.0))) 
                 for _ in range(batch_size)],
                device=accelerator.device
            ).long()
            
            # Add noise to the target latents
            noisy_latents = pipeline.scheduler.add_noise(latents_target, noise, timesteps)
            
            # Get text embeddings for SDXL 
            with torch.no_grad():
                # Format for SDXL requires specific handling of text embeddings
                prompt = [PROMPT] * batch_size
                
                # Original text conditioning
                text_inputs = pipeline.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)
                
                text_inputs_2 = pipeline.tokenizer_2(
                    prompt,
                    padding="max_length",
                    max_length=pipeline.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)
                
                # Get text embeddings from both encoders - now using LoRA-enabled text encoders
                prompt_embeds = text_encoder(text_inputs.input_ids)[0]
                pooled_prompt_embeds = text_encoder_2(text_inputs_2.input_ids)[0]
                
                # CRITICAL: The encoder_hidden_states need to be 2048-dimensional for SDXL
                # We need to concatenate both text encoder outputs to make 2048-dim embeddings
                # (768 from first encoder + 1280 from second encoder)
                prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids, output_hidden_states=True).hidden_states[-2]
                # Now concatenate to get the full 2048-dim embeddings needed for cross-attention
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                
                # Time ids should be shape [batch_size, 6]
                time_ids = torch.zeros((batch_size, 6), device=accelerator.device)
                # Setting width and height to normalized values (1.0)
                time_ids[:, 2] = 1.0  # width
                time_ids[:, 3] = 1.0  # height
                
                # These are the added condition kwargs for the SDXL UNet
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,  # Pooled text embeddings
                    "time_ids": time_ids  # Time embeddings
                }
            
            # UNet forward pass
            with accelerator.accumulate(unet):
                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                
                # Calculate loss
                loss = torch.nn.functional.mse_loss(model_pred, noise)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    params_to_clip = list(unet.parameters())
                    params_to_clip.extend(text_encoder.parameters())
                    params_to_clip.extend(text_encoder_2.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, MAX_GRAD_NORM)
                
                # Optimize
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update EMA models if enabled
                if USE_EMA and accelerator.sync_gradients:
                    ema_unet.update(accelerator.unwrap_model(unet))
                    ema_text_encoder.update(accelerator.unwrap_model(text_encoder))
                    ema_text_encoder_2.update(accelerator.unwrap_model(text_encoder_2))
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
            
            global_step += 1
        
        # Process visualization samples after each epoch
        if(epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs or epoch == STARTING_EPOCH:
            process_vis_samples(epoch)
        
        # Save checkpoint at the specified frequency
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
            save_dir = save_lora_checkpoint(epoch)
            
            # Create a complete pipeline for the final epoch
            if (epoch + 1) == num_epochs:
                # Get the latest saved models
                if USE_EMA:
                    final_unet = ema_unet.model
                    final_text_encoder = ema_text_encoder.model
                    final_text_encoder_2 = ema_text_encoder_2.model
                else:
                    final_unet = accelerator.unwrap_model(unet)
                    final_text_encoder = accelerator.unwrap_model(text_encoder)
                    final_text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
                    
                # Create a temporary pipeline for saving
                final_pipeline = StableDiffusionXLImg2ImgPipeline(
                    vae=vae,
                    text_encoder=final_text_encoder,
                    text_encoder_2=final_text_encoder_2,
                    tokenizer=pipeline.tokenizer,
                    tokenizer_2=pipeline.tokenizer_2,
                    unet=final_unet,
                    scheduler=DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config),
                )
                
                # Save the full pipeline
                pipeline_save_dir = os.path.join(output_dir, f"checkpoint-full-epoch-{epoch+1}")
                os.makedirs(pipeline_save_dir, exist_ok=True)
                final_pipeline.save_pretrained(pipeline_save_dir)
                print(f"Saved full pipeline at epoch {epoch+1}")
    
    # Save the final LoRA weights
    final_save_dir = save_lora_checkpoint(num_epochs-1, is_final=True)
    
    # Save configuration for reference
    config = {
        "base_model": model_name,
        "dataset_paths": dataset_paths,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "unet_rank": UNET_RANK,
        "text_encoder_rank": TEXT_ENCODER_RANK,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "ema_used": USE_EMA,
        "ema_decay": EMA_DECAY,
        "epochs": num_epochs,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "mixed_precision": MIXED_PRECISION,
        "lr_scheduler": LR_SCHEDULER,
        "lr_warmup_ratio": LR_WARMUP_RATIO,
    }
    
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        import json
        json.dump(config, f, indent=2)
        
    print(f"Fine-tuning completed. All models saved to {output_dir}")
    
    return final_pipeline

def main():
    # Configuration with updated paths
    dataset_paths = [
        'datasets/ikea_3d_pairs', 
        'datasets/interior_3d_pairs', 
        'datasets/unsplash_office_3d_pairs', 
        'datasets/unsplash_more_3d_pairs', 
        'datasets/ai_generated_office_3d_pairs', 
        'datasets/own_photos_3d_pairs',
        'datasets/unsplash_plus_office',
        'datasets/unsplash_plus_elements_3d_pairs',
        'datasets/ikea_3d_pairs_mirrored', 
        'datasets/interior_3d_pairs_mirrored', 
        'datasets/unsplash_office_3d_pairs_mirrored', 
        'datasets/unsplash_more_3d_pairs_mirrored', 
        'datasets/ai_generated_office_3d_pairs_mirrored',
        'datasets/own_photos_3d_pairs_mirrored',
        'datasets/unsplash_plus_office_mirrored',
        'datasets/unsplash_plus_elements_3d_pairs_mirrored',
    ]

    output_dir = 'fine_tuned_render2photo_sdxl_lora_enhanced'
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL base model
    
    # Updated test images for visualization
    test_images = [
        "task-images/cgi-rendering-1.jpg",
        "task-images/cgi-rendering-2.jpg",
        "task-images/cgi-rendering-3.jpg",
        "task-images/cgi-rendering-4.jpg",
        "task-images/cgi-rendering-5.jpg",
        "task-images/KI_03.jpg",
        "task-images/KI_04.png",
        "task-images/KI_05.jpg",
        "task-images/KI_06.jpg",
        "task-images/simple_rendering_1.jpeg",
        "task-images/simple_rendering_2.jpeg",
    ]
    
    # Verify test images exist
    valid_test_images = []
    for image_path in test_images:
        if os.path.exists(image_path):
            valid_test_images.append(image_path)
        else:
            print(f"Warning: Test image not found: {image_path}")
    
    if not valid_test_images:
        print("No valid test images found. Please check the paths.")
        valid_test_images = test_images  # Keep the original paths for structure
    
    # Fine-tune the model and get the returned pipeline
    pipeline = fine_tune_model(
        dataset_paths=dataset_paths,
        model_name=model_name,
        output_dir=output_dir,
        batch_size=2,  # Reduced batch size for SDXL due to memory considerations
        num_epochs=TOTAL_EPOCHS,
        learning_rate=2e-4,  # Slightly higher learning rate since we have fewer epochs
        save_freq=1,  # Save every epoch since we only have 5
        test_images=valid_test_images
    )
    
    # Check if the pipeline was returned
    if pipeline is None:
        print("Warning: No pipeline returned from fine_tune_model.")
    
if __name__ == "__main__":
    main()