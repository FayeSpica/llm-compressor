from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# MiniMax-M2 REAP-50 (pruned to ~50% experts)
# Change this to match your model path
MODEL_ID = "Akicou/MiniMax-M2-5-REAP-50"

# Load model.
# ignore_mismatched_sizes=True is needed for REAP models where
# e_score_correction_bias was not trimmed to match pruned expert count
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=[
        "lm_head",
        "re:.*block_sparse_moe.gate$",  # MoE router gate
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
