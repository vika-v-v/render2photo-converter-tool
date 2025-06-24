"""
FLUX.1-dev Fine-tuning Script for Office Render-to-Photo

************************************************************************************
IMPORTANT NOTE: The errors encountered (TypeError related to argument passing
within FluxTransformer2DModel, and persistent shape mismatches) strongly suggest
an issue with the diffusers library version being used (e.g., a development
version like 0.30.0.dev0) or a fundamental incompatibility/misconfiguration
of the loaded model checkpoint with the current fine-tuning setup.

THE STRONGLY RECOMMENDED FIRST STEP IS TO UPDATE YOUR LIBRARIES:
pip install --upgrade diffusers transformers accelerate torch
If issues persist, consider trying a different, stable version of diffusers.
************************************************************************************

Required installations:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
# pip install diffusers transformers accelerate datasets huggingface_hub pillow tqdm sentencepiece protobuf bitsandbytes

Note: FLUX.1-dev is a gated model. You need to:
1. Accept the license at https://huggingface.co/black-forest-labs/FLUX.1-dev
2. Use a HuggingFace token with access to the model (set as HUGGING_FACE_TOKEN environment variable)
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import FluxPipeline 
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
DATASET_NAME = "vika-v/office-render2photos-pairs" 
OUTPUT_DIR = "./flux-office-finetuned"
LOGGING_DIR = Path(OUTPUT_DIR, "logs") 

# Training hyperparameters
TRAIN_BATCH_SIZE = 1  
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5 
NUM_TRAIN_EPOCHS = 5
MAX_TRAIN_STEPS = None  
MIXED_PRECISION = "fp16"  
GRADIENT_CHECKPOINTING = False # Kept False as it was part of debugging
USE_8BIT_ADAM = False  
MAX_GRAD_NORM = 1.0

# Model parameters
RESOLUTION = 1024  

# Dataset parameters
VALIDATION_SPLIT = 0.1 
SEED = 42


class OfficeRenderDataset(Dataset):
    """Dataset for office render to photo pairs"""
    
    def __init__(self, repo_id, token, resolution=1024):
        self.repo_id = repo_id
        self.token = token
        self.resolution = resolution
        self.api = HfApi(token=token)
        self.photo_files = []
        self.render_files = []
        self._load_file_list()
        self.metadata = self._load_metadata()

        if len(self.photo_files) == 0 or len(self.render_files) == 0:
            logger.error(f"No image files found in dataset {self.repo_id}. Check paths (photos/, renders/) and file types.")
            raise ValueError("Dataset is empty or file paths are incorrect.")
        if len(self.photo_files) != len(self.render_files):
            logger.warning(f"Mismatch in number of photo files ({len(self.photo_files)}) and render files ({len(self.render_files)}). Using the minimum.")

    def _load_file_list(self):
        logger.info(f"Loading file list from {self.repo_id}")
        try:
            all_files = list_repo_files(self.repo_id, token=self.token, repo_type="dataset")
        except Exception as e:
            logger.error(f"Failed to list files from repo {self.repo_id}: {e}")
            raise
        
        for file_path in all_files:
            normalized_file_path = file_path.replace("\\", "/")
            if normalized_file_path.startswith("photos/") and (normalized_file_path.lower().endswith((".png", ".jpg", ".jpeg"))):
                self.photo_files.append(file_path) 
            elif normalized_file_path.startswith("renders/") and (normalized_file_path.lower().endswith((".png", ".jpg", ".jpeg"))):
                self.render_files.append(file_path) 
        
        self.photo_files.sort()
        self.render_files.sort()
        logger.info(f"Found {len(self.photo_files)} photo files and {len(self.render_files)} render files")
        
    def _load_metadata(self):
        try:
            metadata_path = hf_hub_download(repo_id=self.repo_id, filename="metadata.json", repo_type="dataset", token=self.token)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata with {len(metadata)} entries")
            return metadata
        except Exception as e: 
            logger.warning(f"Could not load metadata.json from {self.repo_id}: {e}. Proceeding without metadata.")
            return {} 
    
    def __len__(self):
        return min(len(self.photo_files), len(self.render_files))
    
    def _get_image(self, file_path):
        try:
            local_path = hf_hub_download(repo_id=self.repo_id, filename=file_path, repo_type="dataset", token=self.token)
            return Image.open(local_path)
        except Exception as e:
            logger.error(f"Failed to download or open image {file_path} from {self.repo_id}: {e}")
            raise
    
    def _resize_and_pad(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        original_width, original_height = image.size
        image_aspect_ratio = original_width / original_height
        if image_aspect_ratio > 1.0: 
            new_width = self.resolution
            new_height = int(new_width / image_aspect_ratio)
        else: 
            new_height = self.resolution
            new_width = int(new_height * image_aspect_ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        new_image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))
        paste_x = (self.resolution - new_width) // 2
        paste_y = (self.resolution - new_height) // 2
        new_image.paste(image, (paste_x, paste_y))
        return new_image
    
    def __getitem__(self, idx):
        render_path = self.render_files[idx]
        photo_path = self.photo_files[idx]
        render_image = self._get_image(render_path)
        photo_image = self._get_image(photo_path)
        render_image_processed = self._resize_and_pad(render_image)
        photo_image_processed = self._resize_and_pad(photo_image)
        render_tensor = torch.from_numpy(np.array(render_image_processed)).permute(2, 0, 1).float() / 127.5 - 1.0
        photo_tensor = torch.from_numpy(np.array(photo_image_processed)).permute(2, 0, 1).float() / 127.5 - 1.0
        prompt = "A photorealistic office interior photograph with professional lighting and composition." 
        image_filename_key = Path(photo_path).stem 
        if isinstance(self.metadata, dict) and image_filename_key in self.metadata:
            item_metadata = self.metadata[image_filename_key]
            if isinstance(item_metadata, dict): 
                 prompt = item_metadata.get('description', item_metadata.get('render_prompt', prompt))
        elif isinstance(self.metadata, list) and idx < len(self.metadata): 
            item_metadata = self.metadata[idx]
            if isinstance(item_metadata, dict):
                prompt = item_metadata.get('description', item_metadata.get('render_prompt', prompt))
        return {'render': render_tensor, 'photo': photo_tensor, 'prompt': prompt, 'idx': idx}

def load_flux_pipeline(model_name, device, dtype, token=None):
    logger.info(f"Loading FLUX pipeline from {model_name}")
    pipeline = FluxPipeline.from_pretrained(model_name, torch_dtype=dtype, use_safetensors=True, token=token, low_cpu_mem_usage=False)
    pipeline.vae.to(device=device, dtype=dtype)
    pipeline.text_encoder.to(device=device, dtype=dtype)
    pipeline.text_encoder_2.to(device=device, dtype=dtype)
    pipeline.transformer.to(device=device, dtype=dtype) 
    logger.info(f"VAE latent channels: {pipeline.vae.config.latent_channels}") 
    logger.info(f"VAE scaling factor (internal, not spatial): {pipeline.vae.config.scaling_factor}") 
    return pipeline

def prepare_model_for_training(pipeline): 
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(True)
    pipeline.transformer.train()
    if GRADIENT_CHECKPOINTING:
        if hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
            pipeline.transformer.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled for the transformer.")
        else:
            logger.warning("Transformer module does not have 'enable_gradient_checkpointing' method. Skipping.")
    else:
        logger.info("Gradient checkpointing is DISABLED for the transformer.")
    return pipeline

def compute_loss(model_pred_velocity, target_velocity, reduction="mean"):
    if reduction == "mean":
        return F.mse_loss(model_pred_velocity.float(), target_velocity.float(), reduction="mean")
    else:
        return F.mse_loss(model_pred_velocity.float(), target_velocity.float(), reduction="none").mean([1, 2, 3])

def train_flux(pipeline, train_dataloader, accelerator, num_epochs, output_dir):
    optimizer_class = torch.optim.AdamW
    if USE_8BIT_ADAM:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam optimizer.")
        except ImportError:
            logger.warning("bitsandbytes not installed or failed to load. Falling back to standard AdamW.")
    trainable_params = list(filter(lambda p: p.requires_grad, pipeline.transformer.parameters()))
    if not trainable_params:
        logger.error("No trainable parameters found in the transformer. Check requires_grad_() calls.")
        raise ValueError("No parameters to optimize.")
    optimizer = optimizer_class(trainable_params, lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-08)
    
    transformer, optimizer, train_dataloader = accelerator.prepare(
        pipeline.transformer, optimizer, train_dataloader
    )

    # Removed surgical modification of x_embedder as the primary issue is likely library version.
    # If library updates don't solve shape mismatches, diagnostics for the prepared `transformer` might be needed again.
    if accelerator.is_local_main_process:
        logger.info("--- DIAGNOSTICS (ON ACCELERATOR-PREPARED TRANSFORMER, NO MODIFICATION APPLIED IN THIS VERSION) ---")
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        logger.info(f"Type of transformer from accelerator (unwrapped): {type(unwrapped_transformer)}")
        if hasattr(unwrapped_transformer, 'x_embedder') and isinstance(unwrapped_transformer.x_embedder, torch.nn.Linear):
            logger.info(f"Accelerator-prepared transformer x_embedder config - "
                        f"In features: {unwrapped_transformer.x_embedder.in_features}, "
                        f"Out features: {unwrapped_transformer.x_embedder.out_features}")
        else:
            logger.warning("Could not log x_embedder config for accelerator-prepared transformer.")
        logger.info("--- DIAGNOSTICS END ---")
    
    accelerator.wait_for_everyone()


    if MAX_TRAIN_STEPS is None:
        num_update_steps_per_epoch = len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS
        total_training_steps = num_epochs * num_update_steps_per_epoch
    else:
        total_training_steps = MAX_TRAIN_STEPS
        num_epochs = MAX_TRAIN_STEPS // (len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS) + 1 
        logger.info(f"MAX_TRAIN_STEPS set to {MAX_TRAIN_STEPS}. Will run for approximately {num_epochs} epochs.")
    logger.info(f"Starting training for {num_epochs} epochs, {total_training_steps} total update steps.")
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        num_items_in_epoch = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(progress_bar):
            if batch['render'].shape[0] != TRAIN_BATCH_SIZE:
                 logger.warning(f"Skipping incomplete batch of size {batch['render'].shape[0]}")
                 continue
            with accelerator.accumulate(transformer): 
                render_images, target_images, prompts = batch['render'], batch['photo'], batch['prompt']
                with torch.no_grad(): 
                    render_images_vae = render_images.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
                    target_images_vae = target_images.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
                    render_latents = pipeline.vae.encode(render_images_vae).latent_dist.sample() * pipeline.vae.config.scaling_factor
                    target_latents = pipeline.vae.encode(target_images_vae).latent_dist.sample() * pipeline.vae.config.scaling_factor
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt=prompts, prompt_2=prompts, device=accelerator.device, num_images_per_prompt=1)
                
                render_latents = render_latents.to(accelerator.device, dtype=transformer.dtype) 
                target_latents = target_latents.to(accelerator.device, dtype=transformer.dtype)

                if global_step == 0 and accelerator.is_local_main_process: 
                    logger.info(f"Shape of render_latents (from VAE, on accel device, transformer dtype): {render_latents.shape}")
                    logger.info(f"Shape of target_latents (from VAE, on accel device, transformer dtype): {target_latents.shape}")
                    logger.info(f"Shape of prompt_embeds: {prompt_embeds.shape}") 
                    logger.info(f"Shape of pooled_prompt_embeds: {pooled_prompt_embeds.shape}")
                    logger.info(f"Shape of text_ids: {text_ids.shape}")

                noise = torch.randn_like(target_latents) 
                timesteps_flow = torch.rand(target_latents.shape[0], device=target_latents.device) 
                noisy_latents_for_model = (1 - timesteps_flow[:, None, None, None]) * render_latents + timesteps_flow[:, None, None, None] * noise
                target_velocity_for_loss = target_latents - render_latents

                if global_step == 0 and accelerator.is_local_main_process:
                    logger.info(f"Shape of noisy_latents_for_model (input to transformer): {noisy_latents_for_model.shape}")
                    logger.info(f"Shape of target_velocity_for_loss (target for loss): {target_velocity_for_loss.shape}")

                model_timesteps = (timesteps_flow * 1000.0).to(dtype=transformer.dtype) 
                noisy_latents_for_model = noisy_latents_for_model.to(dtype=transformer.dtype)
                prompt_embeds = prompt_embeds.to(dtype=transformer.dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=transformer.dtype)
                text_ids = text_ids.to(device=accelerator.device) 

                model_pred = transformer(hidden_states=noisy_latents_for_model, encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds, timestep=model_timesteps, txt_ids=text_ids, img_ids=None, return_dict=False)[0] 
                
                loss = compute_loss(model_pred, target_velocity_for_loss)
                accelerator.backward(loss)
                if accelerator.sync_gradients: 
                    accelerator.clip_grad_norm_(transformer.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss_sum += loss.item() * render_images.size(0) 
                num_items_in_epoch += render_images.size(0)

            if accelerator.sync_gradients:
                global_step += 1
                avg_loss_so_far = epoch_loss_sum / num_items_in_epoch if num_items_in_epoch > 0 else 0.0
                progress_bar.set_postfix(loss=f"{avg_loss_so_far:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if accelerator.is_main_process and global_step > 0 and global_step % 500 == 0: 
                save_path = Path(output_dir, f"checkpoint-{global_step}")
                unwrapped_transformer_to_save = accelerator.unwrap_model(transformer)
                pipeline.transformer = unwrapped_transformer_to_save 
                pipeline.save_pretrained(save_path)
                logger.info(f"Saved checkpoint to {save_path}")
            if MAX_TRAIN_STEPS is not None and global_step >= MAX_TRAIN_STEPS: break
        
        if MAX_TRAIN_STEPS is not None and global_step >= MAX_TRAIN_STEPS:
            logger.info(f"Reached MAX_TRAIN_STEPS ({MAX_TRAIN_STEPS}). Stopping training.")
            break
        if accelerator.is_local_main_process and num_items_in_epoch > 0:
            avg_epoch_loss = epoch_loss_sum / num_items_in_epoch
            logger.info(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

    if accelerator.is_main_process:
        pipeline.transformer = accelerator.unwrap_model(transformer)
    return pipeline

def main():
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_TOKEN environment variable not set.")
        raise ValueError("HUGGING_FACE_TOKEN not found in environment variables.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=OUTPUT_DIR, logging_dir=LOGGING_DIR)
    accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, mixed_precision=MIXED_PRECISION, log_with="tensorboard", project_config=accelerator_project_config)
    logger.info(f"Accelerator device: {accelerator.device}")
    logger.info(f"Mixed precision: {accelerator.mixed_precision}")
    weights_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = OfficeRenderDataset(repo_id=DATASET_NAME, token=hf_token, resolution=RESOLUTION)
    if len(dataset) == 0:
        logger.error("Dataset is empty. Please check dataset configuration and paths.")
        return
    train_dataset = dataset 
    logger.info(f"Using {len(train_dataset)} samples for training.")
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    pipeline = load_flux_pipeline(MODEL_NAME, accelerator.device, weights_dtype, token=hf_token)
    
    if accelerator.is_local_main_process: 
        logger.info("--- DIAGNOSTIC PRINTS (ON PIPELINE BEFORE ACCELERATOR PREPARE) ---")
        initial_transformer_module = pipeline.transformer
        logger.info(f"Type of pipeline.transformer: {type(initial_transformer_module)}")

        if hasattr(initial_transformer_module, 'config'):
            logger.info(f"Config of initial_transformer_module: {initial_transformer_module.config}")
            if hasattr(initial_transformer_module.config, 'in_channels'):
                 logger.info(f"Initial transformer model config 'in_channels': {initial_transformer_module.config.in_channels}") 
            else:
                logger.warning("initial_transformer_module.config does not have 'in_channels' attribute.")
        else:
            logger.warning("initial_transformer_module does not have 'config' attribute.")

        if hasattr(initial_transformer_module, 'pos_embed'):
            pos_embed_layer = initial_transformer_module.pos_embed
            logger.info(f"Type of initial_transformer_module.pos_embed: {type(pos_embed_layer)}")
            if hasattr(pos_embed_layer, 'proj') and isinstance(pos_embed_layer.proj, torch.nn.Conv2d):
                logger.info(f"Initial pos_embed.proj (Conv2d) config - "
                            f"in_channels: {pos_embed_layer.proj.in_channels}, "
                            f"out_channels (embed_dim): {pos_embed_layer.proj.out_channels}")
            else: 
                 logger.warning(f"Initial pos_embed.proj not found or not Conv2d. Type: {type(getattr(pos_embed_layer, 'proj', None))}")
        else:
            logger.warning("Initial_transformer_module does not have 'pos_embed' attribute.")
        
        if hasattr(initial_transformer_module, 'x_embedder'):
            x_embedder_layer = initial_transformer_module.x_embedder
            if isinstance(x_embedder_layer, torch.nn.Linear):
                logger.info(f"Initial transformer x_embedder (Linear) config - "
                            f"In features: {x_embedder_layer.in_features}, "
                            f"Out features: {x_embedder_layer.out_features}")
            else:
                logger.warning(f"Initial_transformer_module.x_embedder is not a Linear layer. Type: {type(x_embedder_layer)}")
        else:
            logger.warning("Initial_transformer_module does not have 'x_embedder' attribute.")
        logger.info("--- INITIAL DIAGNOSTIC PRINTS END ---")

    pipeline = prepare_model_for_training(pipeline) 
    
    logger.info("Starting training process...")
    trained_pipeline = train_flux(pipeline, train_dataloader, accelerator, NUM_TRAIN_EPOCHS, OUTPUT_DIR)
    
    if accelerator.is_main_process:
        logger.info(f"Saving final model to {OUTPUT_DIR}")
        trained_pipeline.save_pretrained(OUTPUT_DIR)
        logger.info("Training completed and model saved successfully!")
    else:
        logger.info("Training completed on non-main process.")

    accelerator.wait_for_everyone() 
    logger.info("All processes finished.")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
