import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from pathlib import Path
from PIL import Image

# -----------------------------------------------------------------------------
# Manual model loading to bypass extremely slow from_pretrained() behavior
#
# Reason:
# - Florence-2-large-ft requires `trust_remote_code=True`, which triggers custom
#   model logic when calling `from_pretrained()`. This can result in 2+ minutes
#   of delay during loading even for a ~1.5GB model.
# - Manually loading config, instantiating the model, and loading weights skips
#   unnecessary overhead (like dynamic backend registration or tokenizer validation).
# - This method gives full control over the loading process and is much faster.
# -----------------------------------------------------------------------------

# Set paths and device
model_path = Path("./model/Florence-2-large-ft")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load config and manually construct the model
print("Loading config and building model manually...")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Load model weights manually
state_dict_path = model_path / "pytorch_model.bin"  # Or "model.safetensors" if applicable
state_dict = torch.load(state_dict_path, map_location="cpu")
model.load_state_dict(state_dict)

# Optional: override attention backend if required by the model
if hasattr(model.config, "attn_implementation"):
    model.config.attn_implementation = "eager"  # Or "flash_attention_2", etc.

# Move model to desired dtype and device in one go
model = model.to(device, dtype=dtype)

# Load processor (tokenizer, image preprocessor, etc.)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

# Set up prompt and load local image
print("Setting up prompt and loading image...")
"""
region_caption - <OD>
dense_region_caption - <DENSE_REGION_CAPTION>
region_proposal - <REGION_PROPOSAL>
caption - <CAPTION>
detailed_caption - <DETAILED_CAPTION>
more_detailed_caption - <MORE_DETAILED_CAPTION>
caption_to_phrase_grounding - <CAPTION_TO_PHRASE_GROUNDING>
referring_expression_segmentation - <REFERRING_EXPRESSION_SEGMENTATION>
ocr - <OCR>
ocr_with_region - <OCR_WITH_REGION>
docvqa - <DocVQA>
prompt_gen_tags - <GENERATE_TAGS>
prompt_gen_mixed_caption - <MIXED_CAPTION>
prompt_gen_analyze - <ANALYZE>
prompt_gen_mixed_caption_plus - <MIXED_CAPTION_PLUS>
"""

prompt = "<CAPTION>"
local_image_path = "./input/woman-low-res.jpg"
image = Image.open(local_image_path).convert("RGB")

# Process inputs and generate predictions
print("Processing...")
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=4096,
    num_beams=3,
    do_sample=False
)

# Decode and post-process results
print("Decoding results...")
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
parsed_answer = processor.post_process_generation(
    generated_text,
    task="<CAPTION>",
    image_size=(image.width, image.height)
)
print(parsed_answer)
