import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import random
import torch
import copy
import struct
import datetime # Added for loss logging timestamp
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, Sampler
from torchvision import transforms
from PIL import Image, ImageFile, ImageFilter 
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup 
from accelerate import Accelerator
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import get_scheduler 

from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv 

ImageFile.LOAD_TRUNCATED_IMAGES = True # Handle potential truncated images

# Training Configuration
STARTING_EPOCH = 9
TOTAL_EPOCHS = 50
SAVE_FREQ = 1 # Save model weights every 1 epoch
MAX_IMG_SIZE = 1024 # Max image size for visualization resizing (can be different from training)
BASE_IMAGE_SIZE = 1024  # Base size to constrain the longest dimension of training images
DIM_QUANTIZATION_FACTOR = 64 # Quantize training image dimensions to multiples of this

OUTPUT_DIR = "hf_fine_tuned_render2photo_sdxl_lora_aspect_v4_final" # Directory to save model weights and logs

# Enhanced prompts
PROMPT = "high quality photograph, photorealistic, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighting, consistent shadows, preserve as many details from the original image as possible, 8k, 4k, sharp focus, distant objects have consistent detailes, distant objects have consistent shapes"
NEGATIVE_PROMPT = "low quality, bad anatomy, bad hands, error, blurry, out of focus, low resolution, cropped, worst quality, jpeg artifacts, signature, watermark, distorted, inconsistent shapes, inconsistent lighting, inconsistent shadows, descaled, descaling, descaled image, descaled photo"


# Training hyperparameters
LEARNING_RATE = 1e-4
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
MAX_GRAD_NORM = 0.5

DESCALE_BLUR_RADIUS_MIN = 1.0
DESCALE_BLUR_RADIUS_MAX = 2.5
DESCALE_PROBABILITY = 1 # Probability of applying descaling for items from the "descaled" dataset part

# Logging frequency for batch details
LOG_FREQ = 50 # Log AR bucket and shapes every 50 steps

def quantize_dimension(dim, factor):
    """Quantizes a dimension to the nearest multiple of factor."""
    return int(round(dim / factor) * factor)

def resize_if_needed_for_vis(img):
    """Resize image for visualization if either dimension exceeds max_size while preserving aspect ratio"""
    width, height = img.size
    if width <= MAX_IMG_SIZE and height <= MAX_IMG_SIZE:
        return img
    if width > height:
        new_width = MAX_IMG_SIZE
        new_height = int(height * (MAX_IMG_SIZE / width))
    else:
        new_height = MAX_IMG_SIZE
        new_width = int(width * (MAX_IMG_SIZE / height))
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def load_office_render2photos_dataset(hf_token, cache_dir=None):
    login(token=hf_token)
    print("Downloading dataset files from Hugging Face...")
    repo_dir = snapshot_download(
        repo_id="vika-v/office-render2photos-pairs",
        repo_type="dataset", 
        token=hf_token,
        cache_dir=cache_dir
    )
    renders_dir = os.path.join(repo_dir, "renders")
    photos_dir = os.path.join(repo_dir, "photos")
    if not os.path.exists(renders_dir):
        raise ValueError(f"Renders directory not found at {renders_dir}")
    if not os.path.exists(photos_dir):
        raise ValueError(f"Photos directory not found at {photos_dir}")
    render_files = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    photo_files = sorted([f for f in os.listdir(photos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(render_files)} render files and {len(photo_files)} photo files")
    return renders_dir, photos_dir, render_files, photo_files

class EnhancedPairedImageDataset(Dataset):
    def __init__(self, renders_dir, photos_dir, render_files, photo_files,
                 base_image_size, dim_quantization_factor,
                 is_descaled_dataset=False, ar_precision=1):
        self.renders_dir = renders_dir
        self.photos_dir = photos_dir
        self.base_image_size = base_image_size
        self.dim_quantization_factor = dim_quantization_factor
        self.is_descaled_dataset = is_descaled_dataset
        self.ar_precision = ar_precision

        self.image_metas = [] 
        
        print(f"Initializing EnhancedPairedImageDataset ({'Descaled Candidates' if is_descaled_dataset else 'Regular Images'})...")
        
        if len(render_files) != len(photo_files):
            print("Warning: Render and photo file counts differ. Attempting naive matching.")
            photo_map = {os.path.splitext(name)[0]: name for name in photo_files}
            matched_render_files, matched_photo_files = [], []
            for r_file in render_files:
                r_base = os.path.splitext(r_file)[0]
                if r_base in photo_map:
                    matched_render_files.append(r_file); matched_photo_files.append(photo_map[r_base])
            render_files, photo_files = matched_render_files, matched_photo_files
            print(f"After matching: {len(render_files)} render files and {len(photo_files)} photo files.")
            if not render_files:
                 print("Error: No matching files found. Dataset part will be empty.")
                 return

        skipped_due_to_corruption_or_size = 0
        for original_idx, (render_fname, photo_fname) in enumerate(zip(render_files, photo_files)):
            render_path = os.path.join(self.renders_dir, render_fname)
            photo_path = os.path.join(self.photos_dir, photo_fname)
            try:
                with Image.open(render_path) as r_img_verify: r_img_verify.verify()
                with Image.open(render_path) as r_img_data: w_orig, h_orig = r_img_data.size
                with Image.open(photo_path) as p_img_verify: p_img_verify.verify()

                if w_orig == 0 or h_orig == 0:
                    skipped_due_to_corruption_or_size +=1; continue

                # Calculate scaled dimensions maintaining aspect ratio
                ratio = min(self.base_image_size / w_orig, self.base_image_size / h_orig) if w_orig > 0 and h_orig > 0 else 1.0
                if w_orig > self.base_image_size or h_orig > self.base_image_size:
                    ratio = min(self.base_image_size / w_orig, self.base_image_size / h_orig)
                else: 
                    ratio = 1.0 

                scaled_w = int(w_orig * ratio)
                scaled_h = int(h_orig * ratio)

                final_train_w = max(self.dim_quantization_factor, quantize_dimension(scaled_w, self.dim_quantization_factor))
                final_train_h = max(self.dim_quantization_factor, quantize_dimension(scaled_h, self.dim_quantization_factor))
                
                ar = w_orig / h_orig if h_orig > 0 else 1.0
                ar_bucket_key_part = self._quantize_ar(ar)
                bucket_key = (ar_bucket_key_part, final_train_w, final_train_h)

                self.image_metas.append({
                    'render_path': render_path, 'photo_path': photo_path,
                    'render_filename': render_fname, 'photo_filename': photo_fname,
                    'original_width': w_orig, 'original_height': h_orig,
                    'final_train_width': final_train_w, 'final_train_height': final_train_h,
                    'aspect_ratio_bucket_key': bucket_key, 
                })
            except (OSError, struct.error, IOError, Image.UnidentifiedImageError, SyntaxError) as e:
                skipped_due_to_corruption_or_size += 1
        
        print(f"Dataset part ({'Descaled Candidates' if is_descaled_dataset else 'Regular Images'}): "
              f"{len(self.image_metas)} valid pairs. {skipped_due_to_corruption_or_size} skipped.")
        if not self.image_metas:
            print(f"WARNING: No valid image pairs found for this dataset part.")

    def _quantize_ar(self, ar):
        return round(ar * (10**self.ar_precision)) / (10**self.ar_precision)

    def _generate_descaled_image(self, img_pil):
        blur_radius = random.uniform(DESCALE_BLUR_RADIUS_MIN, DESCALE_BLUR_RADIUS_MAX)
        return img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def __len__(self):
        return len(self.image_metas)

    def __getitem__(self, idx):
        meta = self.image_metas[idx]
        render_path = meta['render_path']
        photo_path = meta['photo_path']
        
        try:
            render_img_pil = Image.open(render_path).convert("RGB")
            photo_img_pil = Image.open(photo_path).convert("RGB")

            item_was_descaled = False
            if self.is_descaled_dataset and random.random() < DESCALE_PROBABILITY:
                render_img_pil = self._generate_descaled_image(render_img_pil)
                item_was_descaled = True
            
            target_w, target_h = meta['final_train_width'], meta['final_train_height']
            
            render_transformed = self._transform_image(render_img_pil, target_w, target_h)
            photo_transformed = self._transform_image(photo_img_pil, target_w, target_h)

            paths_dict = {
                'render_path': render_path, 'photo_path': photo_path,
                'render_filename': meta['render_filename'], 'photo_filename': meta['photo_filename'],
                'is_descaled': item_was_descaled, # This flag indicates if blur was applied for THIS item
                'original_width': meta['original_width'], 'original_height': meta['original_height'],
                'final_train_width': target_w, 'final_train_height': target_h,
                'aspect_ratio_bucket_key': meta['aspect_ratio_bucket_key']
            }
            return {'render': render_transformed, 'photo': photo_transformed, 'paths': paths_dict}

        except Exception as e:
            default_error_dim = self.dim_quantization_factor * 4 
            blank_tensor = torch.zeros((3, default_error_dim, default_error_dim)) 
            paths_dict = {
                'render_path': render_path, 'photo_path': photo_path,
                'render_filename': meta.get('render_filename', 'err_render'),
                'photo_filename': meta.get('photo_filename', 'err_photo'),
                'is_descaled': False, 'original_width': 0, 'original_height': 0,
                'final_train_width': default_error_dim, 'final_train_height': default_error_dim,
                'aspect_ratio_bucket_key': 'error'
            }
            return {'render': blank_tensor, 'photo': blank_tensor, 'paths': paths_dict}

    def _transform_image(self, pil_img, target_w, target_h):
        resized_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return img_transforms(resized_img)

class AspectRatioBatchSampler(Sampler):
    def __init__(self, image_metas_with_global_indices, batch_size, drop_last=True):
        super().__init__(image_metas_with_global_indices)
        self.image_metas_with_global_indices = image_metas_with_global_indices 
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.indices_by_bucket_key = {}

        if not self.image_metas_with_global_indices:
            print("Warning: AspectRatioBatchSampler received no image metadata.")
            self._num_batches = 0; return

        for item_meta in self.image_metas_with_global_indices:
            bucket_key = item_meta['aspect_ratio_bucket_key'] 
            global_idx = item_meta['global_idx']
            if bucket_key not in self.indices_by_bucket_key:
                self.indices_by_bucket_key[bucket_key] = []
            self.indices_by_bucket_key[bucket_key].append(global_idx)

        if not self.indices_by_bucket_key:
             print("WARNING: AspectRatioBatchSampler: No aspect ratio buckets were formed."); self._num_batches = 0; return

        print("AspectRatioBatchSampler: Buckets (AR, W, H) and Image Counts:")
        for bucket_k, indices in sorted(self.indices_by_bucket_key.items()):
            print(f"  Bucket {bucket_k}: {len(indices)} images")

        self.valid_buckets = {}
        for bucket_k, indices in self.indices_by_bucket_key.items():
            if len(indices) >= self.batch_size or (not self.drop_last and len(indices) > 0):
                self.valid_buckets[bucket_k] = indices
            elif self.drop_last:
                print(f"  Bucket {bucket_k} has {len(indices)} images, less than batch_size {self.batch_size}. Will be dropped.")
        
        if not self.valid_buckets:
            print("WARNING: AspectRatioBatchSampler: No buckets are large enough or contain images."); self._num_batches = 0; return

        self._num_batches = 0
        for bucket_k in self.valid_buckets:
            count = len(self.valid_buckets[bucket_k])
            self._num_batches += count // self.batch_size if self.drop_last else (count + self.batch_size - 1) // self.batch_size
        
        print(f"AspectRatioBatchSampler: Total number of batches calculated: {self._num_batches}")
        if self._num_batches == 0: print("WARNING: AspectRatioBatchSampler will produce no batches.")

    def __iter__(self):
        if self._num_batches == 0: return iter([])
        all_batches = []
        for bucket_k, indices_in_bucket in self.valid_buckets.items():
            current_indices = list(indices_in_bucket); random.shuffle(current_indices)
            for i in range(0, len(current_indices), self.batch_size):
                batch = current_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    all_batches.append(batch) 
        random.shuffle(all_batches)
        for batch in all_batches: yield batch

    def __len__(self):
        return self._num_batches

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        model_on_cpu = copy.deepcopy(model).cpu()
        self.model = model_on_cpu 
        self.model.requires_grad_(False); self.model.eval()
    def update(self, model):
        self.model.to(model.device)
        with torch.no_grad():
            for ema_param, model_param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    def state_dict(self): return self.model.cpu().state_dict()

def fine_tune_model(hf_token, model_name, 
                    batch_size, num_epochs, 
                    learning_rate, save_freq,
                    test_images):
    torch_dtype = torch.float16 if MIXED_PRECISION == "fp16" else (torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float32)
    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, mixed_precision=MIXED_PRECISION,
                              log_with="tensorboard", project_dir=os.path.join(OUTPUT_DIR, "logs"))
    
    # Initialize loss log file
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    print("Loading SDXL pipeline components...")
    pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, use_safetensors=True, token=hf_token)
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    unet, vae, text_encoder, text_encoder_2 = pipeline.unet, pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2
    tokenizer, tokenizer_2 = pipeline.tokenizer, pipeline.tokenizer_2
    vae.to(accelerator.device, dtype=torch.float32) 

    if GRADIENT_CHECKPOINTING:
        unet.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()
        text_encoder_2.gradient_checkpointing_enable()
    
    unet_lora_config = LoraConfig(r=UNET_RANK, lora_alpha=UNET_RANK, target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"], lora_dropout=0.05, bias="none")
    text_encoder_lora_config = LoraConfig(r=TEXT_ENCODER_RANK, lora_alpha=TEXT_ENCODER_RANK, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], lora_dropout=0.05, bias="none")
    
    unet = get_peft_model(unet, unet_lora_config)
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
    text_encoder_2 = get_peft_model(text_encoder_2, text_encoder_lora_config)
    
    if accelerator.is_main_process:
        unet.print_trainable_parameters()
        text_encoder.print_trainable_parameters()
        text_encoder_2.print_trainable_parameters()
    
    unet.train()
    text_encoder.train()
    text_encoder_2.train()
    vae.requires_grad_(False)
    vae.eval() 
    
    # Define helper functions first
    def save_training_state(epoch, optimizer, lr_scheduler, global_step, output_dir):
        """Save optimizer and scheduler state"""
        state_path = os.path.join(output_dir, f"training_state_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
        }, state_path)

    def load_training_state(epoch, optimizer, lr_scheduler, output_dir):
        """Load optimizer and scheduler state"""
        state_path = os.path.join(output_dir, f"training_state_epoch_{epoch}.pt")
        if os.path.exists(state_path):
            checkpoint = torch.load(state_path, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return checkpoint['global_step']
        return 0
    
    def process_vis_samples(epoch): 
        if not accelerator.is_main_process: return
        print(f"Processing visualization samples for epoch {epoch+1}...")
        vis_unet = accelerator.unwrap_model(ema_unet.model if USE_EMA else unet)
        vis_text_encoder = accelerator.unwrap_model(ema_text_encoder.model if USE_EMA else text_encoder)
        vis_text_encoder_2 = accelerator.unwrap_model(ema_text_encoder_2.model if USE_EMA else text_encoder_2)
        inference_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline_t = StableDiffusionXLImg2ImgPipeline(vae=accelerator.unwrap_model(vae), text_encoder=vis_text_encoder,
            text_encoder_2=vis_text_encoder_2, tokenizer=tokenizer, tokenizer_2=tokenizer_2, 
            unet=vis_unet, scheduler=inference_scheduler).to(accelerator.device)
        pipeline_t.set_progress_bar_config(disable=True)
        sample_dir = os.path.join(OUTPUT_DIR, f"samples/epoch_{epoch+1}"); os.makedirs(sample_dir, exist_ok=True)
        for i, image_path in enumerate(test_images):
            try:
                if not os.path.isfile(image_path): print(f"Vis Warning: File not found: {image_path}"); continue
                input_image = Image.open(image_path).convert("RGB")
                input_image_resized = resize_if_needed_for_vis(input_image) 
                filename = os.path.splitext(os.path.basename(image_path))[0]
                with torch.no_grad():
                    generated_image = pipeline_t(PROMPT, negative_prompt=NEGATIVE_PROMPT, image=input_image_resized, 
                                                 strength=0.4, guidance_scale=7.5, num_inference_steps=30).images[0]
                input_image_resized.save(os.path.join(sample_dir, f"{filename}_input.png"))
                generated_image.save(os.path.join(sample_dir, f"{filename}_output_epoch{epoch+1}.png"))
            except Exception as e: print(f"Vis Error {image_path}: {e}"); import traceback; traceback.print_exc()
        del pipeline_t; torch.cuda.empty_cache() if torch.cuda.is_available() else None

            
    def save_lora_checkpoint(epoch, is_final=False): 
        if not accelerator.is_main_process: return None
        unet_s = accelerator.unwrap_model(ema_unet.model if USE_EMA and not is_final else unet)
        txt_enc_s = accelerator.unwrap_model(ema_text_encoder.model if USE_EMA and not is_final else text_encoder)
        txt_enc2_s = accelerator.unwrap_model(ema_text_encoder_2.model if USE_EMA and not is_final else text_encoder_2)
        save_dir = os.path.join(OUTPUT_DIR, "lora-weights-final" if is_final else f"lora-weights-epoch-{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)
        unet_s.save_pretrained(os.path.join(save_dir, "unet"))
        txt_enc_s.save_pretrained(os.path.join(save_dir, "text_encoder"))
        txt_enc2_s.save_pretrained(os.path.join(save_dir, "text_encoder_2"))

        print(f"Saved LoRA weights epoch {epoch+1} to {save_dir}")
        return save_dir
    

    def validate_model(unet, text_encoder, text_encoder_2, vae, pipeline, validation_samples, accelerator):
        """Calculate validation loss on a small subset"""
        unet.eval()
        text_encoder.eval() 
        text_encoder_2.eval()
        
        total_val_loss = 0.0
        num_val_samples = 0
        
        with torch.no_grad():
            for val_batch in validation_samples:
                try:
                    render_images, target_images = val_batch['render'], val_batch['photo']
                    
                    target_latents = vae.encode(target_images.to(dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(target_latents)
                    bsz = target_latents.shape[0]
                    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device).long()
                    noisy_latents = pipeline.scheduler.add_noise(target_latents, noise, timesteps)
                    
                    prompt_ids = tokenizer([PROMPT]*bsz, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                    prompt_ids_2 = tokenizer_2([PROMPT]*bsz, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                    
                    txt_enc_out = text_encoder(prompt_ids, output_hidden_states=True)
                    txt_enc2_out = text_encoder_2(prompt_ids_2, output_hidden_states=True)
                    final_enc_hidden = torch.cat([txt_enc_out.last_hidden_state, txt_enc2_out.hidden_states[-2]], dim=-1)
                    
                    add_time_ids_list = []
                    for item_paths in val_batch['paths']:
                        orig_h, orig_w = item_paths['original_height'], item_paths['original_width']
                        target_h, target_w = item_paths['final_train_height'], item_paths['final_train_width']
                        add_time_ids_list.append(torch.tensor([orig_h, orig_w, 0, 0, target_h, target_w]))
                    add_time_ids = torch.stack(add_time_ids_list).to(accelerator.device, dtype=final_enc_hidden.dtype)
                    added_conds = {"text_embeds": txt_enc2_out.text_embeds, "time_ids": add_time_ids}
                    
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=final_enc_hidden, added_cond_kwargs=added_conds).sample
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    total_val_loss += loss.item()
                    num_val_samples += 1
                    
                except Exception as e:
                    continue
        
        avg_val_loss = total_val_loss / num_val_samples if num_val_samples > 0 else float('inf')
        return avg_val_loss
    
    if USE_EMA:
        ema_unet = EMA(unet)
        ema_text_encoder = EMA(text_encoder)
        ema_text_encoder_2 = EMA(text_encoder_2)
        if accelerator.is_main_process:
            print(f"Initialized EMA models with decay {EMA_DECAY}")
    
    renders_dir, photos_dir, render_files, photo_files = load_office_render2photos_dataset(hf_token=hf_token)
    
    regular_dataset_part = EnhancedPairedImageDataset(
        renders_dir, photos_dir, render_files, photo_files,
        base_image_size=BASE_IMAGE_SIZE, dim_quantization_factor=DIM_QUANTIZATION_FACTOR,
        is_descaled_dataset=False
    )
    descaled_dataset_part = EnhancedPairedImageDataset(
        renders_dir, photos_dir, render_files, photo_files,
        base_image_size=BASE_IMAGE_SIZE, dim_quantization_factor=DIM_QUANTIZATION_FACTOR,
        is_descaled_dataset=True
    )

    all_image_metas_for_sampler = []
    for i, meta_item in enumerate(regular_dataset_part.image_metas):
        all_image_metas_for_sampler.append({'global_idx': i, 'aspect_ratio_bucket_key': meta_item['aspect_ratio_bucket_key']})
    offset_idx = len(regular_dataset_part)
    for i, meta_item in enumerate(descaled_dataset_part.image_metas):
        all_image_metas_for_sampler.append({'global_idx': offset_idx + i, 'aspect_ratio_bucket_key': meta_item['aspect_ratio_bucket_key']})
        
    if not all_image_metas_for_sampler and accelerator.is_main_process:
        raise ValueError("No images found in dataset parts for sampler. Cannot train.")

    combined_dataset = ConcatDataset([regular_dataset_part, descaled_dataset_part])
    if accelerator.is_main_process:
        print(f"Combined dataset: {len(combined_dataset)} pairs. Regular: {len(regular_dataset_part)}, Descaled: {len(descaled_dataset_part)}")
    if len(combined_dataset) == 0 and accelerator.is_main_process:
         raise ValueError("Combined dataset is empty.")
    
    val_size = max(1, len(combined_dataset) // 20)  # 5% for validation
    train_size = len(combined_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    
    # Create validation samples
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    import itertools
    val_samples = list(itertools.islice(val_dataloader, 10))  # Take 10 samples for validation
    best_val_loss = float('inf')

    batch_sampler = AspectRatioBatchSampler(all_image_metas_for_sampler, batch_size, drop_last=True)
    if accelerator.is_main_process and len(batch_sampler) == 0:
        print("WARNING: AspectRatioBatchSampler will produce no batches.")

    dataloader = DataLoader(combined_dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
        collate_fn=lambda items: {
            'render': torch.stack([item['render'] for item in items if item and 'render' in item]),
            'photo': torch.stack([item['photo'] for item in items if item and 'photo' in item]),
            'paths': [item['paths'] for item in items if item and 'paths' in item],
        })
    
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters())) + \
                       list(filter(lambda p: p.requires_grad, text_encoder.parameters())) + \
                       list(filter(lambda p: p.requires_grad, text_encoder_2.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, betas=(ADAM_BETA1, ADAM_BETA2), eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY)
    
    num_update_steps_per_epoch = (len(batch_sampler) // GRADIENT_ACCUMULATION_STEPS) if hasattr(batch_sampler, '__len__') and len(batch_sampler) > 0 else \
                                 ((len(combined_dataset) // batch_size) // GRADIENT_ACCUMULATION_STEPS if batch_size > 0 else 0)
    if num_update_steps_per_epoch == 0 and accelerator.is_main_process:
        print("Warning: num_update_steps_per_epoch is 0.")
    
    num_training_steps = num_update_steps_per_epoch * (num_epochs - STARTING_EPOCH)
    if num_training_steps == 0 and accelerator.is_main_process:
        print("ERROR: num_training_steps is 0.")
        return None
    num_warmup_steps = int(LR_WARMUP_RATIO * num_training_steps)
    
    lr_scheduler_cls = get_cosine_schedule_with_warmup if LR_SCHEDULER == "cosine_with_restarts" else get_scheduler
    lr_scheduler_args = {"optimizer": optimizer, "num_warmup_steps": num_warmup_steps * accelerator.num_processes, "num_training_steps": num_training_steps * accelerator.num_processes}
    if LR_SCHEDULER == "cosine_with_restarts":
        lr_scheduler_args["num_cycles"] = LR_NUM_CYCLES
    else:
        lr_scheduler_args["name"] = LR_SCHEDULER
    lr_scheduler = lr_scheduler_cls(**lr_scheduler_args)
    
    unet, text_encoder, text_encoder_2, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, text_encoder_2, optimizer, dataloader, lr_scheduler)

    # Load checkpoint after accelerator.prepare()
    global_step = 0
    if STARTING_EPOCH > 1:
        checkpoint_path = os.path.join(OUTPUT_DIR, f"lora-weights-epoch-{STARTING_EPOCH}")
        if os.path.exists(checkpoint_path):
            accelerator.wait_for_everyone()
            print(f"Loading checkpoint from: {checkpoint_path}")
            try:
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                unwrapped_text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
                
                unwrapped_unet.load_adapter(os.path.join(checkpoint_path, "unet"), "default")
                unwrapped_text_encoder.load_adapter(os.path.join(checkpoint_path, "text_encoder"), "default")
                unwrapped_text_encoder_2.load_adapter(os.path.join(checkpoint_path, "text_encoder_2"), "default")
                
                global_step = load_training_state(STARTING_EPOCH, optimizer, lr_scheduler, OUTPUT_DIR)
                
                if accelerator.is_main_process:
                    print(f"Successfully loaded LoRA weights from {checkpoint_path}")
                    print(f"Resuming training from global_step: {global_step}")
                    
                    # Check file existence and sizes
                    print("\n=== CHECKPOINT FILE DIAGNOSTICS ===")
                    for component in ["unet", "text_encoder", "text_encoder_2"]:
                        adapter_path = os.path.join(checkpoint_path, component, "adapter_model.safetensors")
                        if os.path.exists(adapter_path):
                            size_mb = os.path.getsize(adapter_path) / (1024 * 1024)
                            print(f"✓ {component}/adapter_model.safetensors exists: {size_mb:.2f} MB")
                        else:
                            print(f"✗ ERROR: {component}/adapter_model.safetensors NOT FOUND")
                    
                    # Detailed parameter analysis
                    print("\n=== UNET LORA PARAMETER DIAGNOSTICS ===")
                    with torch.no_grad():
                        # Check different parameter name patterns
                        lora_patterns = ['lora', 'adapter']
                        for pattern in lora_patterns:
                            params = [(n, p) for n, p in unwrapped_unet.named_parameters() if pattern in n.lower()]
                            print(f"\nFound {len(params)} parameters containing '{pattern}'")
                            if len(params) > 0:
                                # Show some parameter names as examples
                                print(f"Example parameter names: {[n for n, _ in params[:3]]}")
                                
                                # Check if parameters are trainable
                                trainable = sum(1 for _, p in params if p.requires_grad)
                                print(f"Trainable parameters: {trainable}/{len(params)}")
                                
                                # Statistics on parameter values
                                non_zero_params = sum(1 for _, p in params if torch.any(p != 0))
                                print(f"Non-zero parameters: {non_zero_params}/{len(params)}")
                                
                                if len(params) > 0:
                                    # Calculate more detailed statistics
                                    param_means = [p.mean().item() for _, p in params]
                                    param_stds = [p.std().item() for _, p in params]
                                    param_mins = [p.min().item() for _, p in params]
                                    param_maxs = [p.max().item() for _, p in params]
                                    
                                    # Overall statistics
                                    mean_of_means = sum(param_means) / len(param_means) if param_means else 0
                                    mean_of_stds = sum(param_stds) / len(param_stds) if param_stds else 0
                                    overall_min = min(param_mins) if param_mins else 0
                                    overall_max = max(param_maxs) if param_maxs else 0
                                    
                                    print(f"Parameter statistics:")
                                    print(f"  Mean of means: {mean_of_means:.8f}")
                                    print(f"  Mean of standard deviations: {mean_of_stds:.8f}")
                                    print(f"  Overall min: {overall_min:.8f}")
                                    print(f"  Overall max: {overall_max:.8f}")
                                    
                                    # Sample some individual parameters
                                    print("\nSample of individual parameters:")
                                    for i, (name, param) in enumerate(params[:5]):  # Look at first 5
                                        print(f"  {name}:")
                                        print(f"    Shape: {param.shape}")
                                        print(f"    Mean: {param.mean().item():.8f}")
                                        print(f"    Std: {param.std().item():.8f}")
                                        print(f"    Min: {param.min().item():.8f}")
                                        print(f"    Max: {param.max().item():.8f}")
                                        print(f"    Has non-zeros: {torch.any(param != 0).item()}")
                                        if param.numel() < 50:
                                            print(f"    Values: {param.flatten().tolist()}")
                    
                    print("\n=== TEXT ENCODER PARAMETER DIAGNOSTICS ===")
                    # Similar analysis for text encoder
                    with torch.no_grad():
                        text_params = [(n, p) for n, p in unwrapped_text_encoder.named_parameters() if 'lora' in n.lower() or 'adapter' in n.lower()]
                        if text_params:
                            print(f"Found {len(text_params)} LoRA parameters in text_encoder")
                            non_zero = sum(1 for _, p in text_params if torch.any(p != 0))
                            print(f"Non-zero parameters: {non_zero}/{len(text_params)}")
                            
                            if len(text_params) > 0:
                                param_means = [p.mean().item() for _, p in text_params]
                                mean_of_means = sum(param_means) / len(param_means)
                                print(f"Mean of means: {mean_of_means:.8f}")
                    
                    print("\n=== WEIGHT VERIFICATION SUMMARY ===")
                    # Determine if weights appear to be loaded correctly
                    all_params = [(n, p) for n, p in unwrapped_unet.named_parameters() 
                                if ('lora' in n.lower() or 'adapter' in n.lower()) and p.requires_grad]
                    non_zero_count = sum(1 for _, p in all_params if torch.any(p != 0))
                    
                    if len(all_params) == 0:
                        print("❌ CRITICAL ISSUE: No LoRA parameters found in UNet!")
                    elif non_zero_count == 0:
                        print("❌ CRITICAL ISSUE: All LoRA parameters are ZERO! Weights not loaded correctly.")
                    elif non_zero_count < len(all_params) * 0.5:
                        print(f"⚠️ WARNING: Only {non_zero_count}/{len(all_params)} LoRA parameters have non-zero values.")
                    else:
                        print(f"✅ SUCCESS: {non_zero_count}/{len(all_params)} LoRA parameters have non-zero values.")
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
                import traceback
                traceback.print_exc()
                print("Starting fresh training")
                global_step = 0
        else:
            if accelerator.is_main_process:
                print(f"Checkpoint not found at {checkpoint_path}. Starting fresh.")
    
    vae.to(accelerator.device)
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)
    
    # Training loop
    for epoch in range(STARTING_EPOCH, num_epochs):
        unet.train()
        text_encoder.train()
        text_encoder_2.train()
        epoch_loss = 0.0
        skipped_batches = 0
        
        if accelerator.is_main_process:
            epoch_losses_not_descaled = []
            epoch_losses_descaled = []
            loss_log_file_path = os.path.join(OUTPUT_DIR, f"training_losses_epoch_{epoch + 1}.txt")
            with open(loss_log_file_path, "w") as f_loss:
                f_loss.write(f"Training Loss Log - Epoch {epoch + 1} started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f_loss.write("Format: <type>: loss1, loss2, ...;\n")
                f_loss.write("="*40 + "\n")
        
        prog_total = len(dataloader.batch_sampler) if hasattr(dataloader.batch_sampler, '__len__') and len(dataloader.batch_sampler) > 0 else \
                     (len(combined_dataset) // batch_size if batch_size > 0 else 0)
        if prog_total == 0 and accelerator.is_main_process:
            print("ERROR: Dataloader is empty or batch_sampler length is 0. Cannot continue epoch.")
            break

        progress_bar = tqdm(total=prog_total, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process or prog_total == 0)
        
        for step, batch in enumerate(dataloader):
            render_tensor = batch.get('render')
            photo_tensor = batch.get('photo')

            if not batch or render_tensor is None or photo_tensor is None or \
               render_tensor.shape[0] == 0 or photo_tensor.shape[0] == 0:
                if accelerator.is_main_process: 
                    print(f"WARNING: Empty/malformed batch E{epoch+1} S{step}. Skipping. Batch content: {batch}")
                skipped_batches += 1
                continue
            try:
                render_images, target_images = render_tensor, photo_tensor 
                if torch.isnan(render_images).any() or torch.isnan(target_images).any():
                    if accelerator.is_main_process:
                        print(f"WARNING: NaN in input E{epoch+1} S{step}. Skipping.")
                    skipped_batches += 1
                    continue
                
                with accelerator.accumulate(unet, text_encoder, text_encoder_2): 
                    with torch.no_grad(): 
                        vae.to(target_images.device, dtype=torch.float32) 
                        target_latents = vae.encode(target_images.to(dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor

                    noise = torch.randn_like(target_latents)
                    bsz = target_latents.shape[0]
                    timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=target_latents.device).long()
                    noisy_latents = pipeline.scheduler.add_noise(target_latents, noise, timesteps)

                    prompt_ids = tokenizer([PROMPT]*bsz, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                    prompt_ids_2 = tokenizer_2([PROMPT]*bsz, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
                    
                    txt_enc_out = text_encoder(prompt_ids, output_hidden_states=True)
                    enc_hidden = txt_enc_out.last_hidden_state
                    txt_enc2_out = text_encoder_2(prompt_ids_2, output_hidden_states=True)
                    pooled_embeds = txt_enc2_out.text_embeds 
                    enc_hidden2 = txt_enc2_out.hidden_states[-2] 
                    final_enc_hidden = torch.cat([enc_hidden, enc_hidden2], dim=-1)

                    add_time_ids_list = []
                    for item_paths in batch['paths']: 
                        orig_h, orig_w = item_paths['original_height'], item_paths['original_width']
                        batch_target_h, batch_target_w = item_paths['final_train_height'], item_paths['final_train_width']
                        current_orig_size = (max(1, orig_h), max(1, orig_w)) 
                        current_crop_coords = (0,0) 
                        current_target_size = (batch_target_h, batch_target_w)
                        add_time_ids_list.append(torch.tensor(list(current_orig_size + current_crop_coords + current_target_size)))
                    
                    add_time_ids = torch.stack(add_time_ids_list).to(accelerator.device, dtype=final_enc_hidden.dtype)
                    added_conds = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=final_enc_hidden, added_cond_kwargs=added_conds).sample
                    
                    current_loss_val = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    if torch.isnan(model_pred).any() or torch.isnan(current_loss_val).any():
                        if accelerator.is_main_process: 
                            print(f"WARNING: NaN in model_pred or loss at E{epoch+1} S{step}. Skipping batch.")
                        skipped_batches += 1
                        accelerator.skip_gradient_accumulation()
                        optimizer.zero_grad()
                        continue
                            
                    epoch_loss += current_loss_val.detach().item()
                    accelerator.backward(current_loss_val)
                    
                    if accelerator.is_main_process:
                        if batch['paths']:
                            descaled_count = sum(1 for path in batch['paths'] if path.get('is_descaled', False))
                            regular_count = len(batch['paths']) - descaled_count
                            loss_val = current_loss_val.detach().item()
                            
                            if descaled_count > 0 and regular_count > 0:
                                epoch_losses_descaled.extend([loss_val] * descaled_count)
                                epoch_losses_not_descaled.extend([loss_val] * regular_count)
                            elif descaled_count > 0:
                                epoch_losses_descaled.append(loss_val)
                            else:
                                epoch_losses_not_descaled.append(loss_val)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad() 
                        
                if USE_EMA and accelerator.sync_gradients: 
                    ema_unet.update(accelerator.unwrap_model(unet))
                    ema_text_encoder.update(accelerator.unwrap_model(text_encoder))
                    ema_text_encoder_2.update(accelerator.unwrap_model(text_encoder_2))
                
                if prog_total > 0:
                    progress_bar.update(1)
                curr_lr = lr_scheduler.get_last_lr()[0]
                if prog_total > 0:
                    progress_bar.set_postfix({"loss": current_loss_val.detach().item(), "lr": curr_lr})
                
                if global_step % LOG_FREQ == 0 and accelerator.is_main_process:
                    if batch['paths']: 
                        descaled_count = sum(1 for path in batch['paths'] if path.get('is_descaled', False))
                        total_batch_size = len(batch['paths'])
                        
                        first_path_info = batch['paths'][0]
                        bucket_key_str = str(first_path_info['aspect_ratio_bucket_key'])
                        orig_dims_str = f"{first_path_info['original_width']}x{first_path_info['original_height']}"
                        final_dims_str = f"{first_path_info['final_train_width']}x{first_path_info['final_train_height']}"
                        
                        print(f"E{epoch+1} S{step}/{prog_total if prog_total > 0 else 'N/A'}, GS:{global_step}, "
                              f"Loss:{current_loss_val.detach().item():.5f}, LR:{curr_lr:.2e}, "
                              f"BucketKey:{bucket_key_str} (Orig:{orig_dims_str} -> Final:{final_dims_str}), "
                              f"BatchSize:{total_batch_size} (Descaled:{descaled_count}, Regular:{total_batch_size-descaled_count}), "
                              f"Tensor Shapes: R={render_images.shape} P={target_images.shape}")
                    else: 
                         print(f"E{epoch+1} S{step}/{prog_total if prog_total > 0 else 'N/A'}, GS:{global_step}, Loss:{current_loss_val.detach().item():.5f}, LR:{curr_lr:.2e}, "
                               f"Batch Shapes: R={render_images.shape} P={target_images.shape} (Paths info missing)")
                         
                global_step += 1
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"ERROR batch E{epoch+1} S{step}: {e}")
                    import traceback
                    traceback.print_exc()
                skipped_batches += 1
                optimizer.zero_grad() if 'optimizer' in locals() else None
                continue
            
        if prog_total > 0:
            progress_bar.close()
        
        # Log collected losses for the epoch to file
        if accelerator.is_main_process:
            with open(loss_log_file_path, "a") as f_loss:
                if epoch_losses_not_descaled:
                    losses_str = ", ".join([f"{l:.4f}" for l in epoch_losses_not_descaled])
                    f_loss.write(f"not descaled: {losses_str};\n")
                if epoch_losses_descaled:
                    losses_str = ", ".join([f"{l:.4f}" for l in epoch_losses_descaled])
                    f_loss.write(f"descaled: {losses_str};\n")
                if not epoch_losses_not_descaled and not epoch_losses_descaled:
                    f_loss.write(f"No losses recorded this epoch.\n")

        num_proc_batches = (prog_total if prog_total > 0 else step + 1) - skipped_batches
        avg_epoch_loss = epoch_loss / num_proc_batches if num_proc_batches > 0 else float('nan')
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} done. Avg loss: {avg_epoch_loss:.6f}. Skipped: {skipped_batches}/{prog_total if prog_total > 0 else step + 1}")
            
            # Validation
            val_loss = validate_model(unet, text_encoder, text_encoder_2, vae, pipeline, val_samples, accelerator)
            print(f"Epoch {epoch+1} - Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            with open(loss_log_file_path, "a") as f_loss:
                 f_loss.write(f"validation: {val_loss:.6f};\n")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {val_loss:.6f}")
            
        if accelerator.is_main_process and ((epoch + 1) % save_freq == 0 or epoch == num_epochs - 1):
            chkpt_dir = save_lora_checkpoint(epoch)
            save_training_state(epoch, optimizer, lr_scheduler, global_step, OUTPUT_DIR)
            if chkpt_dir and test_images:
                process_vis_samples(epoch)
    
    if accelerator.is_main_process:
        final_chkpt_dir = save_lora_checkpoint(num_epochs - 1, is_final=True)
        if final_chkpt_dir:
            print(f"Training complete. Final LoRA weights at {final_chkpt_dir}")
    accelerator.wait_for_everyone() 

    if accelerator.is_main_process: 
        final_unet = accelerator.unwrap_model(ema_unet.model if USE_EMA else unet)
        final_text_encoder = accelerator.unwrap_model(ema_text_encoder.model if USE_EMA else text_encoder)
        final_text_encoder_2 = accelerator.unwrap_model(ema_text_encoder_2.model if USE_EMA else text_encoder_2)
        print("Final pipeline can be constructed with the saved LoRA weights.")
        return "Pipeline construction placeholder" 
    return None

def main():
    load_dotenv(); hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token: print("Error: HUGGING_FACE_TOKEN not found."); return
    
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    test_image_folder = "task-images"; test_image_files = [ 
        "cgi-rendering-1.jpg", "cgi-rendering-2.jpg", "KI_03.jpg", "simple_rendering_1.jpeg" 
    ] 
    test_images_full_paths = [os.path.join(test_image_folder, f) for f in test_image_files if os.path.exists(os.path.join(test_image_folder, f))]
    if not test_images_full_paths: print("No valid test images found.")

    script_batch_size = 1 
    trained_pipeline = fine_tune_model(hf_token, model_name, script_batch_size, 
                                       TOTAL_EPOCHS, LEARNING_RATE, SAVE_FREQ, test_images_full_paths)
    if trained_pipeline: print("Training completed successfully!")
    else: print("Training process finished.")
    
if __name__ == "__main__":
    if not torch.cuda.is_available(): print("WARNING: CUDA not available. CPU training is very slow.")
    main()
