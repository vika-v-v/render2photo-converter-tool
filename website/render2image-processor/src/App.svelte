<!-- App.svelte -->
<script>
  // Import necessary components
  import { onMount } from 'svelte';
  
  // Form state
  let formData = new FormData();
  let imageFile = null;
  let imagePreview = null;
  let processedImage = null;
  let isLoading = false;
  let error = null;
  let isDraggingOver = false;
  
  // Modal state
  let showResultModal = false;
  let showComparisonView = false;
  
  // Default settings from your API
  let settings = {
    LORA_DIR: "loras/final/lora-weights-epoch-10",
    PROMPT: "high quality photograph, photorealistic, masterpiece, high quality, detailed, realistic, photorealistic, consistent shapes, consistent lighting, consistent shadows, preserve as many details from the original image as possible, 8k, 4k, sharp focus",
    NEGATIVE_PROMPT: "low quality, bad anatomy, bad hands, text, error, blurry, out of focus, low resolution, cropped, worst quality, jpeg artifacts, signature, watermark, distorted",
    STRENGTH: "0.25",
    GUIDANCE_SCALE: "10.5",
    MAX_IMG_SIZE: "2048",
    LORA_SCALE: "0.8",
    NUM_STEPS: "100",
    USE_CUSTOM_NOISE: "True",
    SEED: "42",
    POST_PROCESS: "True",
    CONTRAST_FACTOR: "1.2",
    SHARPNESS_FACTOR: "1.7",
    SATURATION_FACTOR: "1.1",
    ENABLE_FACE_ENHANCEMENT: "False",
    FACE_DETECTION_CONFIDENCE: "0.7",
    FACE_PADDING_PERCENT: "10",
    FACE_PROMPT: "high quality photograph, photorealistic, masterpiece, perfect face details, realistic face features, high quality, detailed face, ultra realistic human face, perfect eyes, perfect skin texture, perfect facial proportions, clean render",
    FACE_NEGATIVE_PROMPT: "low quality, bad anatomy, distorted face, deformed face, disfigured face, unrealistic face, bad eyes, crossed eyes, misaligned eyes, bad nose, bad mouth, bad teeth, bad skin, ugly"
  };
  
  // Create boolean variables for checkbox state
  let useCustomNoise = settings.USE_CUSTOM_NOISE === "True";
  let enablePostProcess = settings.POST_PROCESS === "True";
  let enableFaceEnhancement = settings.ENABLE_FACE_ENHANCEMENT === "True";
  
  // Update settings when checkbox state changes
  $: settings.USE_CUSTOM_NOISE = useCustomNoise ? "True" : "False";
  $: settings.POST_PROCESS = enablePostProcess ? "True" : "False";
  $: settings.ENABLE_FACE_ENHANCEMENT = enableFaceEnhancement ? "True" : "False";
  
  // Advanced settings visibility toggle
  let showAdvancedSettings = false;
  
  // Initialize range styles function
  function initializeRangeStyles() {
    document.querySelectorAll('input[type="range"]').forEach(input => {
      const min = parseFloat(input.min) || 0;
      const max = parseFloat(input.max) || 1;
      const value = parseFloat(input.value);
      const percentage = ((value - min) / (max - min)) * 100;
      input.style.setProperty('--value', `${percentage}%`);
    });
  }
  
  // Update range style on input
  function updateRangeStyle(event) {
    const input = event.target;
    const min = parseFloat(input.min) || 0;
    const max = parseFloat(input.max) || 1;
    const value = parseFloat(input.value);
    const percentage = ((value - min) / (max - min)) * 100;
    input.style.setProperty('--value', `${percentage}%`);
  }
  
  // Reactive statement to reinitialize styles when advanced settings are shown
  $: if (showAdvancedSettings) {
    setTimeout(() => {
      initializeRangeStyles();
    }, 0);
  }
  
  // Handle file selection
  function handleFileChange(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    imageFile = file;
    
    // Create preview
    const reader = new FileReader();
    reader.onload = e => {
      imagePreview = e.target.result;
    };
    reader.readAsDataURL(file);
  }
  
  // Build FormData with all settings
  function buildFormData() {
    const data = new FormData();
    
    // Add the image file
    if (imageFile) {
      data.append('Image', imageFile);
    }
    
    // Add all settings
    Object.entries(settings).forEach(([key, value]) => {
      data.append(key, value);
    });
    
    return data;
  }
  
  // Process the image
  async function processImage() {
    if (!imageFile) {
      error = "Please select an image first";
      return;
    }
    
    isLoading = true;
    error = null;
    
    try {
      const apiUrl = "http://10.128.39.228:5000/process-file";
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: buildFormData(),
        // Add auth header if needed
        // headers: {
        //   'X-API-Key': 'your-api-key'
        // }
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      // The response is the processed image
      const blob = await response.blob();
      processedImage = URL.createObjectURL(blob);
      
      // Show the result modal
      showResultModal = true;
    } catch (err) {
      error = `Error processing image: ${err.message}`;
      console.error(err);
    } finally {
      isLoading = false;
    }
  }
  
  // Download the processed image
  function downloadImage() {
    if (!processedImage) return;
    
    const a = document.createElement('a');
    a.href = processedImage;
    a.download = `enhanced-${imageFile.name}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
  
  // Reset the form
  function resetForm() {
    imageFile = null;
    imagePreview = null;
    processedImage = null;
    error = null;
  }
  
  // Toggle comparison view
  function toggleComparisonView() {
    showComparisonView = !showComparisonView;
  }
  
  let compareHoverState = false;
  function handleCompareMouseDown() {
    compareHoverState = true;
    showComparisonView = true;
  }
  
  function handleCompareMouseUp() {
    compareHoverState = false;
    showComparisonView = false;
  }
  
  function handleCompareKeyDown(event) {
    if (event.key === ' ' || event.key === 'Enter') {
      showComparisonView = true;
    }
  }
  
  function handleCompareKeyUp(event) {
    if (event.key === ' ' || event.key === 'Enter') {
      showComparisonView = false;
    }
  }
  
  // Close the result modal
  function closeModal() {
    showResultModal = false;
    showComparisonView = false;
  }
  
  // Handle keyboard events
  function handleKeydown(event) {
    if (event.key === 'Escape' && showResultModal) {
      closeModal();
    }
  }

  // Drag and drop handlers
  function handleDragEnter(event) {
    event.preventDefault();
    event.stopPropagation();
    isDraggingOver = true;
  }
  
  function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    isDraggingOver = true;
  }
  
  function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    isDraggingOver = false;
  }
  
  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    isDraggingOver = false;
    
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      
      // Check if it's an image
      if (file.type.startsWith('image/')) {
        imageFile = file;
        
        // Create preview
        const reader = new FileReader();
        reader.onload = e => {
          imagePreview = e.target.result;
        };
        reader.readAsDataURL(file);
      } else {
        error = "Please upload an image file";
      }
    }
  }
  
  // Setup component lifecycle
  onMount(() => {
    // Setup keyboard event listener
    window.addEventListener('keydown', handleKeydown);
    
    // Initialize range styles for visible sliders
    initializeRangeStyles();
    
    // Add event listener for range inputs using event delegation
    document.addEventListener('input', (event) => {
      if (event.target.type === 'range') {
        updateRangeStyle(event);
      }
    });
    
    return () => {
      window.removeEventListener('keydown', handleKeydown);
    };
  });

</script>

<svelte:head>
  <!-- Add Font Awesome for icons -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
</svelte:head>

<main>
  <div class="header-container">
    <h1>3d-Render to photo converter - optimized for office interiors</h1>
  </div>
  
  <div class="container">
    <div class="upload-section glass-panel">
      <h2>Upload Image</h2>
      <div 
        class="image-upload" 
        class:has-image={imagePreview}
        class:dragging={isDraggingOver}
        on:dragenter={handleDragEnter}
        on:dragover={handleDragOver}
        on:dragleave={handleDragLeave}
        on:drop={handleDrop}
      >
        {#if imagePreview}
          <img src={imagePreview} alt="Preview" />
          <button class="remove-button" on:click={resetForm}>✕</button>
        {:else}
          <label for="image-input">
            <i class="fas fa-cloud-upload-alt"></i>
            <span>Click or drop the image here</span>
          </label>
          <input 
            id="image-input"
            type="file" 
            accept="image/*" 
            on:change={handleFileChange}
          />
        {/if}
      </div>
      <div class="action-buttons">
        <button 
          class="process-button" 
          on:click={processImage} 
          disabled={!imageFile || isLoading}
        >
          {#if isLoading}
            <i class="fas fa-spinner fa-spin"></i> Processing...
          {:else}
            <i class="fas fa-magic"></i> Process image
          {/if}
        </button>
      </div>
    </div>
    
    <div class="settings-section glass-panel">
      <h2>Settings</h2>
      
      <!-- Basic Settings -->
      <div class="settings-group">
        <h3>Basic settings</h3>
        
        <div class="form-group">
          <label for="prompt">Prompt:</label>
          <textarea id="prompt" bind:value={settings.PROMPT} rows="4" maxlength="500" class="textfield"></textarea>
        </div>
        
        <div class="form-group">
          <label for="negative-prompt">Negative prompt:</label>
          <textarea id="negative-prompt" bind:value={settings.NEGATIVE_PROMPT} rows="3" maxlength="500" class="textfield"></textarea>
        </div>
        
        <div class="form-group">
          <label for="strength">Strength (how much the image must be changed):</label>
          <input 
            type="range" 
            id="strength" 
            bind:value={settings.STRENGTH} 
            min="0" 
            max="1" 
            step="0.05"
          />
          <span>{settings.STRENGTH}</span>
        </div>
        
        <div class="form-group">
          <label for="guidance-scale">Guidance scale (prompt following):</label>
          <input 
            type="range" 
            id="guidance-scale" 
            bind:value={settings.GUIDANCE_SCALE} 
            min="1" 
            max="20" 
            step="0.5"
          />
          <span>{settings.GUIDANCE_SCALE}</span>
        </div>
        
        <div class="form-group">
          <label for="steps">Steps (more steps result in better quality but longer processing time):</label>
          <input 
            type="range" 
            id="steps" 
            bind:value={settings.NUM_STEPS} 
            min="10" 
            max="100" 
            step="5"
          />
          <span>{settings.NUM_STEPS}</span>
        </div>
        
        <div class="form-group">
          <label for="seed">Seed:</label>
          <input 
            type="number" 
            id="seed" 
            bind:value={settings.SEED} 
          />
        </div>
      </div>
      
      <!-- Advanced Settings Toggle -->
      <button class="toggle-button" on:click={() => showAdvancedSettings = !showAdvancedSettings}>
         {showAdvancedSettings ? 'Hide' : 'Show'} advanced settings
      </button>
      
      <!-- Advanced Settings -->
      {#if showAdvancedSettings}
        <div class="settings-group">
          <h3>Advanced Settings</h3>
          
          <div class="form-group">
            <label for="lora-dir">LoRA path:</label>
            <input type="text" id="lora-dir" bind:value={settings.LORA_DIR} />
          </div>
          
          <div class="form-group">
            <label for="lora-scale">LoRA scale:</label>
            <input 
              type="range" 
              id="lora-scale" 
              bind:value={settings.LORA_SCALE} 
              min="0" 
              max="1" 
              step="0.05"
            />
            <span>{settings.LORA_SCALE}</span>
          </div>
          
          <div class="form-group">
            <label for="max-img-size">Maximum image size:</label>
            <input 
              type="number" 
              id="max-img-size" 
              bind:value={settings.MAX_IMG_SIZE} 
              min="512" 
              max="4096" 
              step="128"
            />
          </div>
          
          <div class="form-group">
            <label>
              <input type="checkbox" bind:checked={useCustomNoise} />
              Use custom noise
            </label>
          </div>
          
          <div class="form-group">
            <label>
              <input type="checkbox" bind:checked={enablePostProcess} />
              Enable post-processing
            </label>
          </div>
          
          {#if enablePostProcess}
            <div class="form-group">
              <label for="contrast">Contrast factor:</label>
              <input 
                type="range" 
                id="contrast" 
                bind:value={settings.CONTRAST_FACTOR} 
                min="0.5" 
                max="2" 
                step="0.1"
              />
              <span>{settings.CONTRAST_FACTOR}</span>
            </div>
            
            <div class="form-group">
              <label for="sharpness">Sharpness:</label>
              <input 
                type="range" 
                id="sharpness" 
                bind:value={settings.SHARPNESS_FACTOR} 
                min="0.5" 
                max="3" 
                step="0.1"
              />
              <span>{settings.SHARPNESS_FACTOR}</span>
            </div>
            
            <div class="form-group">
              <label for="saturation">Saturation:</label>
              <input 
                type="range" 
                id="saturation" 
                bind:value={settings.SATURATION_FACTOR} 
                min="0.5" 
                max="2" 
                step="0.1"
              />
              <span>{settings.SATURATION_FACTOR}</span>
            </div>
          {/if}
        </div>
        
        <!-- Face Enhancement Settings 
        <div class="settings-group">
          <h3>Gesichtsverbesserung</h3>
          
          <div class="form-group">
            <label>
              <input type="checkbox" bind:checked={enableFaceEnhancement} />
              Aktivieren der Gesichtsverbesserung
            </label>
          </div>
          
          {#if enableFaceEnhancement}
            <div class="form-group">
              <label for="face-prompt">Gesicht Prompt:</label>
              <textarea id="face-prompt" bind:value={settings.FACE_PROMPT} rows="4" maxlength="500" class="textfield"></textarea>
            </div>
            
            <div class="form-group">
              <label for="face-negative-prompt">Gesicht Negatives Prompt:</label>
              <textarea id="face-negative-prompt" bind:value={settings.FACE_NEGATIVE_PROMPT} rows="3" maxlength="500" class="textfield"></textarea>
            </div>
            
            <div class="form-group">
              <label for="face-detection-confidence">Sicherheit bei der Gesichtserkennung:</label>
              <input 
                type="range" 
                id="face-detection-confidence" 
                bind:value={settings.FACE_DETECTION_CONFIDENCE} 
                min="0.1" 
                max="1" 
                step="0.05"
              />
              <span>{settings.FACE_DETECTION_CONFIDENCE}</span>
            </div>
            
            <div class="form-group">
              <label for="face-padding">Gesicht Padding (%):</label>
              <input 
                type="number" 
                id="face-padding" 
                bind:value={settings.FACE_PADDING_PERCENT} 
                min="0" 
                max="50"
              />
            </div>
          {/if}
        </div>-->
      {/if}
      
      {#if error}
        <div class="error-message">
          <i class="fas fa-exclamation-circle"></i> {error}
        </div>
      {/if}
    </div>
  </div>
</main>

<!-- Result Modal -->
{#if showResultModal && processedImage}
  <div class="modal-overlay" on:click={closeModal} on:keydown={handleKeydown} role="dialog" aria-modal="true" aria-labelledby="modal-title">
    <div class="modal-content" on:click|stopPropagation on:keydown={() => {}} role="document">
      <button class="close-modal" on:click={closeModal}>✕</button>
      
      <h2 id="modal-title">Processed Image</h2>
      
      <div class="result-container">
        <div class="result-image">
          <div class="image-container">
            {#if showComparisonView && imagePreview}
              <img src={imagePreview} alt="Original" class="image-layer" />
              <div class="image-label">Original</div>
            {:else}
              <img src={processedImage} alt="Processed" class="image-layer" />
              <div class="image-label">Processed</div>
            {/if}
          </div>
        </div>
      </div>
      
      <div class="modal-actions">
        <button 
          class="action-button compare-button" 
          on:mousedown={handleCompareMouseDown}
          on:mouseup={handleCompareMouseUp}
          on:mouseleave={handleCompareMouseUp}
          on:keydown={handleCompareKeyDown}
          on:keyup={handleCompareKeyUp}
          on:touchstart={handleCompareMouseDown}
          on:touchend={handleCompareMouseUp}
          aria-label="Hold to compare with original image"
        >
          <i class="fas fa-eye"></i> Hold to Compare
        </button>
        <button class="action-button download-button" on:click={downloadImage}>
          <i class="fas fa-download"></i> Download
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    background-image: url('/mesut-cicen-qZbht5_6Iko-unsplash.jpg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    min-height: 100vh;
  }
  
  main {
    max-width: 1200px;
    margin: 0 auto;
    /*padding: 20px;*/
    display: flex;
    flex-direction: column;
    justify-content: top;
    align-items: center;
    font-family: Geneva, Verdana, sans-serif;
    color: #000;
    backdrop-filter: blur(15px);
    width: 100%;
    min-width: 100%;
    min-height: 100vh;
  }

  .image-upload.dragging {
    border-color: rgba(255, 255, 255, 0.8);
    background-color: rgba(255, 255, 255, 0.1);
  }
  
  .header-container {
    text-align: center;
    margin-bottom: 30px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 0px 0px 12px 12px;
    padding: 0px 30px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    border: 2px solid rgba(255, 255, 255, 0.1);
    width: fit-content;
  }

  h1 {
    text-align: center;
    
    /*margin-bottom: 30px;*/
    text-shadow: none;
    font-size: 1.8rem;
    width: fit-content;
  }
  
  h2, h3 {
    text-shadow: none;
    font-size: 1.2rem;
    margin-bottom: 15px;
  }
  
  .container {
    display: grid;
    grid-template-columns: 1fr;
    gap: 30px;
    width: 90%;
  }
  
  @media (min-width: 1024px) {
    .container {
      grid-template-columns: 1fr 1fr;
    }
  }
  
  /* Glass Effect */
  .glass-panel {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 12px;
    padding: 5px 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.1);
    height: fit-content;
  }

  .settings-section {
    max-height: calc(100vh - 165px);
    overflow-y: auto;
    padding-bottom: 20px;
    padding-right: 20px;    /* Add padding on the right side */
    margin-right: -20px;    /* Use negative margin to compensate */
    box-sizing: content-box; /* Ensures the padding doesn't affect the width */
    background: rgba(255, 255, 255, 0.5);
  }
  
  /* Upload Section */
  .image-upload {
    height: 300px;
    border: 2px dashed rgba(117, 117, 117, 0.3);
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
    position: relative;
    overflow: hidden;
    background-color: rgba(255, 255, 255, 0.3);
    margin-bottom: 20px;
    transition: all 0.3s;
  }
  
  .image-upload:hover {
    border-color: rgba(135, 135, 135, 0.6);
  }
  
  .image-upload.has-image {
    border-style: solid;
    border-color: rgba(135, 135, 135, 0.6);
  }
  
  .image-upload label {
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    color: #000000;
  }
  
  .image-upload label i {
    font-size: 48px;
    margin-bottom: 15px;
    color: rgba(115, 115, 115, 0.7);
  }
  
  .image-upload input {
    display: none;
  }
  
  .image-upload img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }
  
  .remove-button {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(0, 0, 0, 0.5);
    color: white;
    border: none;
    border-radius: 50%;
    width: 25px;
    height: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 1;
    transition: background 0.2s;
  }
  
  .remove-button:hover {
    background: rgba(0, 0, 0, 0.7);
  }
  
  /* Settings Section */
  .settings-group {
    margin-bottom: 20px;
    padding: 0px 20px;
    background-color: rgba(255, 255, 255, 0.3);
    border-radius: 8px;
    border: 1px solid rgba(136, 136, 136, 0.1);
  }
  
  .form-group {
    margin-bottom: 15px;
  }
  
  label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
    color: rgba(0, 0, 0, 0.9);
    font-size: 14px;
    text-shadow: none;
  }
  
  input[type="text"],
  input[type="number"],
  textarea {
    width: 95%;
    padding: 8px;
    border: 1px solid rgba(96, 96, 96, 0.3);
    border-radius: 4px;
    font-size: 14px;
    background: rgba(174, 174, 174, 0.6);
    color: #4a4a4a;
    box-shadow: none;
    resize: none;
  }

  input[type="text"]:focus,
  input[type="number"]:focus,
  textarea:focus {
    border-color: #050505; /* Change this to your preferred focus color */
    outline: none; /* Remove the default browser outline */
  }
  
  input[type="number"] {
    width: 100px;
  }
  
  .textfield {
    max-width: 100%;
    overflow-y: auto;
  }
  
  /* Updated CSS for the range input */
  input[type="range"] {
    width: 80%;
    vertical-align: middle;
    position: relative;
    height: 6px;
    border-radius: 3px;
    border: 1px solid rgba(96, 96, 96, 0.7);
    -webkit-appearance: none;
    appearance: none;
    outline: none;
    background-image: linear-gradient(to right, rgb(182, 182, 182) 0%, rgb(182, 182, 182) var(--value, 0%), rgb(234, 234, 234) var(--value, 0%));
  }

  /* Style the range handle (thumb) for Webkit browsers */
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 1px solid rgba(96, 96, 96, 0.7);
    background: rgba(166, 166, 166, 1);
    cursor: pointer;

  }

  /* Style the range handle (thumb) for Firefox */
  input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 1px solid rgba(96, 96, 96, 0.7);
    background: rgba(166, 166, 166, 1);
    cursor: pointer;
    /* No hover effect as per requirement 2 */
  }

  /* Remove the hover effect from the range input */
  input[type="range"]:hover {
    /* No specific hover styling as per requirement 2 */
  }

  /* Ensure the styles are maintained when focusing the input */
  input[type="range"]:focus {
    outline: none;
  }

  input[type="range"] + span {
    margin-left: 10px;
    font-size: 14px;
    color: rgba(0, 0, 0, 0.8);
    min-width: 30px;
    display: inline-block;
    text-align: right;
  }

  input[type="checkbox"] {
    margin-right: 8px;
    accent-color: rgba(218, 218, 218, 0.8);
  }
  
  /* Toggle Button */
  .toggle-button {
    border: 1px solid rgba(96, 96, 96, 0.3);
    border-radius: 4px;
    background: rgba(174, 174, 174, 0.6);
    color: #323232;
    padding: 8px 16px;
    cursor: pointer;
    margin-bottom: 20px;
    font-size: 14px;
    transition: background-color 0.2s;
  }
  
  
  /* Action Buttons */
  .action-buttons {
    margin: 20px 0px;
    display: flex;
    justify-content: center;
    width: 100%;
  }
  
  .process-button {
    border: 1px solid rgba(96, 96, 96, 0.3);
    border-radius: 4px;
    background: rgba(174, 174, 174, 0.6);
    color: #4a4a4a;
    padding: 12px 24px;
    font-size: 16px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.3s;
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
  }
  
  .process-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  }
  
  .process-button:disabled {
    background: rgba(221, 221, 221, 0.6);
    cursor: not-allowed;
  }
  
  /* Error Message */
  .error-message {
    background-color: rgba(255, 87, 87, 0.2);
    color: #ff5757;
    padding: 10px;
    border-radius: 4px;
    margin-top: 20px;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  
  /* Modal */
  .modal-overlay {
    font-family: Geneva, Verdana, sans-serif;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(5px);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
    animation: fadeIn 0.3s;
  }
  
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  .modal-content {
    background: rgb(223, 223, 223);
    backdrop-filter: blur(20px);
    border-radius: 12px;
    padding: 25px;
    max-width: 90%;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    animation: slideUp 0.4s;
  }
  
  @keyframes slideUp {
    from { transform: translateY(50px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  .close-modal {
    position: absolute;
    top: 15px;
    right: 15px;
    background: rgba(0, 0, 0, 0.2);
    color: #000;
    border: none;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .close-modal:hover {
    background: rgba(0, 0, 0, 0.3);
    transform: scale(1.1);
  }
  
  .result-container {
    margin: 20px 0;
  }
  
  .result-image {
    text-align: center;
  }
  
  .image-container {
    position: relative;
    width: 100%;
    max-width: 100%;
    text-align: center;
  }
  
  .image-layer {
    max-width: 100%;
    max-height: 70vh;
    border-radius: 4px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    transition: opacity 0.3s ease;
  }
  
  .image-label {
    position: absolute;
    top: 10px;
    left: 10px;
    background-color: rgba(255, 255, 255, 0.6);
    color: black;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 14px;
  }
  
  .modal-actions {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 20px;
  }
  
  .action-button {
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.3s;
    border: 1px solid rgba(178, 178, 178, 0.2);
    justify-content: center;
  }
  
  .compare-button {
    border: 1px solid rgba(96, 96, 96, 0.3);
    border-radius: 4px;
    background: rgba(174, 174, 174, 0.6);
    color: #4a4a4a;
    user-select: none;
    -webkit-user-select: none;
    touch-action: manipulation;
    width: 50%;
  }
  
  .compare-button:hover {
    background: rgba(142, 142, 142, 0.6);
  }
  
  .compare-button:active {
    background: rgba(104, 104, 104, 0.7);
  }
  
  .download-button {
    border: 1px solid rgba(96, 96, 96, 0.3);
    border-radius: 4px;
    background: rgba(174, 174, 174, 0.6);
    color: #4a4a4a;
    width: 50%;
  }
  
  .download-button:hover {
    background: rgba(142, 142, 142, 0.6);
  }
</style>