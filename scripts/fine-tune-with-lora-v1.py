import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import random
import torch
import struct  # Add this for handling struct errors
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import get_scheduler

# Configure PIL to handle truncated images more gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set the starting epoch - we'll continue from epoch 12
STARTING_EPOCH = 5
# Keep original total epochs
TOTAL_EPOCHS = 15
MAX_IMG_SIZE = 2048  # Maximum image size for resizing

PROMPT = "high quality photograph, photorealistic, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighning, consistent shadows, preserve as many details from the original image as possible"

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

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=MAX_IMG_SIZE):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size  # Target size for all images
        self.render_dir = os.path.join(root_dir, "renders")
        self.photo_dir = os.path.join(root_dir, "photos")
        
        # Get all the render files (both png and jpg)
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
            
            # Resize to a fixed size to ensure all tensors have the same dimensions
            # This uses a center crop approach to maintain aspect ratio as much as possible
            render_img = transforms.CenterCrop(min(render_img.size))(render_img)  # Make square first
            render_img = render_img.resize((self.img_size, self.img_size), Image.LANCZOS)
            
            photo_img = transforms.CenterCrop(min(photo_img.size))(photo_img)  # Make square first
            photo_img = photo_img.resize((self.img_size, self.img_size), Image.LANCZOS)
            
            # Store original paths for visualization
            paths = {
                'render_path': render_path,
                'photo_path': photo_path,
                'render_filename': self.render_files[real_idx],
                'photo_filename': self.photo_files[real_idx]
            }
            
            if self.transform:
                render_img = self.transform(render_img)
                photo_img = self.transform(photo_img)
            
            return {'render': render_img, 'photo': photo_img, 'paths': paths}
            
        except (OSError, struct.error, IOError) as e:
            # This is a fallback in case a corrupted file was missed during validation
            print(f"❌ Error loading file that passed validation: {render_path} or {photo_path}")
            print(f"   Error: {str(e)}")
            
            # Create a simple replacement with blank images
            # For training, it's better to return something than to crash
            blank_render = Image.new("RGB", (self.img_size, self.img_size), color=(0, 0, 0))
            blank_photo = Image.new("RGB", (self.img_size, self.img_size), color=(0, 0, 0))
            
            paths = {
                'render_path': render_path,
                'photo_path': photo_path,
                'render_filename': self.render_files[real_idx],
                'photo_filename': self.photo_files[real_idx]
            }
            
            if self.transform:
                blank_render = self.transform(blank_render)
                blank_photo = self.transform(blank_photo)
                
            return {'render': blank_render, 'photo': blank_photo, 'paths': paths}

def fine_tune_model(dataset_paths, model_name, 
                    output_dir, 
                    batch_size, num_epochs, 
                    learning_rate, save_freq,
                    test_images):
    """
    Fine-tune a pre-trained SDXL image-to-image model on 3D render to photo dataset
    using LoRA for efficient training
    """
    # Set a fixed image size for consistency across all datasets
    IMG_SIZE = 512  # Common size that should work for most GPUs
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="fp16",
    )
    
    print("Loading SDXL pipeline components...")
    # Load pre-trained SDXL model
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name)
    
    # Use the DDPM scheduler for fine-tuning
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # Get the components
    unet = pipeline.unet
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    text_encoder_2 = pipeline.text_encoder_2
    
    # Move all model components to the same device as accelerator
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    
    # Configure LoRA for UNet with SDXL-specific target modules
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        # Targeting specific modules for SDXL compatibility
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # Main attention blocks
            "ff.net.0.proj", "ff.net.2",         # Feed-forward blocks
        ],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    
    # Load the latest checkpoint (if available)
    checkpoint_path = os.path.join(output_dir, f"lora-weights-epoch-{STARTING_EPOCH}")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        unet.load_adapter(checkpoint_path, adapter_name="default")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting fresh training.")
    
    # Enable gradient calculation for the UNet with LoRA
    unet.train()
    
    # Freeze VAE and text encoders
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    text_encoder_2.requires_grad_(False)
    text_encoder_2.eval()
    
    transform = transforms.Compose([
        # transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Now handled in the Dataset classes
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Create datasets and data loaders for each path
    all_datasets = []
    for path in dataset_paths:
        dataset = PairedImageDataset(root_dir=path, transform=transform, img_size=IMG_SIZE)
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
        num_workers=4
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Set up learning rate scheduler - use cosine scheduler with warmup
    # Modified to calculate remaining steps based on starting from STARTING_EPOCH
    remaining_epochs = num_epochs - STARTING_EPOCH
    num_training_steps = len(dataloader) * remaining_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Prepare model, optimizer, dataloader and scheduler for accelerator
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    
    def process_vis_samples(epoch):
        print(f"Processing visualization samples for epoch {epoch+1}...")
        
        # Create a temporary pipeline for inference
        unwrapped_unet = accelerator.unwrap_model(unet)
        
        # For inference, use a faster scheduler
        inference_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        pipeline_t = StableDiffusionXLImg2ImgPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            unet=unwrapped_unet,
            scheduler=inference_scheduler,
        ).to(accelerator.device)
        
        # Make sure pipeline is in eval mode
        pipeline_t.unet.eval()
        
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
                    # For inference, we need to prepare the embeddings properly
                    text_inputs = pipeline_t.tokenizer(
                        PROMPT,
                        padding="max_length",
                        max_length=pipeline_t.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    text_inputs_2 = pipeline_t.tokenizer_2(
                        PROMPT,
                        padding="max_length",
                        max_length=pipeline_t.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    # Ensure proper full prompt embedding by using pipeline class's internal methods
                    # The pipeline internally handles the correct concatenation and processing
                    model_output = pipeline_t(
                        prompt=PROMPT,
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
            
    # Training loop
    global_step = 0
    max_grad_norm = 1.0  # For gradient clipping
    
    # Start from the next epoch after the checkpoint
    for epoch in range(STARTING_EPOCH, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}")
        
        for step, batch in enumerate(dataloader):
            # Get the input images
            render_images = batch['render'].to(accelerator.device)
            target_images = batch['photo'].to(accelerator.device)
            
            # Encode the images with VAE (on the same device as the images)
            with torch.no_grad():
                # Scale down the latents
                latents_render = vae.encode(render_images).latent_dist.sample() * 0.18215
                latents_target = vae.encode(target_images).latent_dist.sample() * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents_target)
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, pipeline.scheduler.config.num_train_timesteps, 
                (latents_target.shape[0],), 
                device=accelerator.device
            ).long()
            
            # Add noise to the target latents
            noisy_latents = pipeline.scheduler.add_noise(latents_target, noise, timesteps)
            
            # Create consistent prompt embeddings for SDXL
            prompt = [PROMPT] * render_images.shape[0]
            
            # Get text embeddings for SDXL (properly handling both encoders)
            with torch.no_grad():
                # Format for SDXL requires specific handling of text embeddings
                
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
                
                # Get text embeddings from both encoders
                prompt_embeds = text_encoder(text_inputs.input_ids)[0]
                pooled_prompt_embeds = text_encoder_2(text_inputs_2.input_ids)[0]
                
                # CRITICAL: The encoder_hidden_states need to be 2048-dimensional for SDXL
                # We need to concatenate both text encoder outputs to make 2048-dim embeddings
                # (768 from first encoder + 1280 from second encoder)
                prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids, output_hidden_states=True).hidden_states[-2]
                # Now concatenate to get the full 2048-dim embeddings needed for cross-attention
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                
                # SDXL uses 2048 size original embeddings + 1280 size pooled embeddings in added_cond_kwargs
                # Added cond kwargs must contain:
                # - text_embeds: The pooled text embeddings from text_encoder_2 (shape: batch_size x 1280)
                # - time_ids: Original SDXL uses this for img2img with shape batch_size x 6, with values [0, 0, width, height, crop_top, crop_left]
                
                # Time ids should be shape [batch_size, 6]
                time_ids = torch.zeros((render_images.shape[0], 6), device=accelerator.device)
                # Setting width and height to normalized values (1.0)
                time_ids[:, 2] = 1.0  # width
                time_ids[:, 3] = 1.0  # height
                
                # These are the added condition kwargs for the SDXL UNet
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,  # Pooled text embeddings (batch_size x 1280)
                    "time_ids": time_ids  # Time embeddings (batch_size x 6)
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
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                
                # Optimize
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
            
            global_step += 1
        
        # Process visualization samples after each epoch
        if(epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs or epoch == STARTING_EPOCH:
            process_vis_samples(epoch)
        
        # Save checkpoint at the specified frequency
        if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
            # Get unwrapped model for saving
            unwrapped_unet = accelerator.unwrap_model(unet)
            
            # Save LoRA weights separately
            unwrapped_unet.save_pretrained(os.path.join(output_dir, f"lora-weights-epoch-{epoch+1}"))
            print(f"Saved LoRA weights at epoch {epoch+1}")
            
            # Create a full pipeline for testing/inference
            if (epoch + 1) == num_epochs:  # Only save full pipeline for the final epoch to save disk space
                # Create a temporary pipeline for saving
                pipeline_t = StableDiffusionXLImg2ImgPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=pipeline.tokenizer,
                    tokenizer_2=pipeline.tokenizer_2,
                    unet=unwrapped_unet,
                    scheduler=DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config),
                )
                
                # Save the pipeline
                pipeline_t.save_pretrained(os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}"))
                print(f"Saved full pipeline checkpoint at epoch {epoch+1}")
    
    # Get unwrapped model for the final save
    unwrapped_unet = accelerator.unwrap_model(unet)
    
    # Save LoRA weights separately for the final model
    unwrapped_unet.save_pretrained(os.path.join(output_dir, "lora-weights-final"))
    
    # Create the final pipeline with SDXL and LoRA
    final_pipeline = StableDiffusionXLImg2ImgPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=pipeline.tokenizer,
        tokenizer_2=pipeline.tokenizer_2,
        unet=unwrapped_unet,
        scheduler=DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config),
    )
    
    # Save the final pipeline
    final_pipeline.save_pretrained(output_dir)
    print(f"Fine-tuning completed. Model saved to {output_dir}")
    
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

    output_dir = 'fine_tuned_render2photo_sdxl_lora_more_data'
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