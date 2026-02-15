import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# MiniMax-M2 REAP-50 (pruned to ~50% experts)
# Change this to match your model path
MODEL_ID = "Akicou/MiniMax-M2-5-REAP-50"


def dequantize_fp8_to_bf16(model, block_size=(128, 128)):
    """
    Dequantize FP8 (float8_e4m3fn) weights to bfloat16 using weight_scale_inv.
    FP8 block-quantized weights need scale factors applied to recover original values.
    CUDA doesn't support min/max ops on FP8, so we must dequantize before NVFP4 quantization.
    """
    count = 0
    for name, module in model.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if module.weight.dtype != torch.float8_e4m3fn:
            continue

        weight = module.weight.data.to(torch.bfloat16)

        if hasattr(module, "weight_scale_inv") and weight.dim() == 2:
            scale_inv = module.weight_scale_inv.data
            if scale_inv.dim() == 2:
                bs_r, bs_c = block_size
                out_f, in_f = weight.shape
                scale = scale_inv.repeat_interleave(bs_r, dim=0).repeat_interleave(
                    bs_c, dim=1
                )
                scale = scale[:out_f, :in_f].to(
                    device=weight.device, dtype=weight.dtype
                )
                weight = weight * scale
            else:
                weight = weight * scale_inv.to(
                    device=weight.device, dtype=weight.dtype
                )

            if "weight_scale_inv" in dict(module.named_parameters(recurse=False)):
                del module._parameters["weight_scale_inv"]
            elif "weight_scale_inv" in dict(module.named_buffers(recurse=False)):
                del module._buffers["weight_scale_inv"]
            else:
                delattr(module, "weight_scale_inv")

        module.weight = torch.nn.Parameter(weight, requires_grad=False)
        count += 1

    print(f"Dequantized {count} FP8 layers to bfloat16")
    return model


# Load model.
# ignore_mismatched_sizes=True is needed for REAP models where
# e_score_correction_bias was not trimmed to match pruned expert count
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    trust_remote_code=True,
    ignore_mismatched_sizes=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Dequantize FP8 weights to bfloat16 before NVFP4 quantization
block_size = (128, 128)
if hasattr(model.config, "quantization_config"):
    qc = model.config.quantization_config
    if isinstance(qc, dict) and "weight_block_size" in qc:
        block_size = tuple(qc["weight_block_size"])
    elif hasattr(qc, "weight_block_size") and qc.weight_block_size:
        block_size = tuple(qc.weight_block_size)

print(f"Dequantizing FP8 weights with block_size={block_size}...")
dequantize_fp8_to_bf16(model, block_size=block_size)

# Remove FP8 quantization_config to avoid torch.fx tracing issues
if hasattr(model.config, "quantization_config"):
    del model.config.quantization_config


DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:.*block_sparse_moe.gate$",  # MoE router gate
    ],
)

# Apply quantization.
# MoE calibration is now handled automatically by the pipeline.
# We set `moe_calibrate_all_experts` to True to ensure all experts receive
# calibration data. This temporarily updates the model definition to use
# `CalibrationMiniMaxM2SparseMoeBlock` (from `llmcompressor.modeling.minimax_moe`)
# which replaces the original `MiniMaxM2SparseMoeBlock` class.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
)


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
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
