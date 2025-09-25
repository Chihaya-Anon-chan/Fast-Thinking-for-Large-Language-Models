import transformers
from models.codebook import CodebookAdapterModel, CodebookConfig
from peft import LoraConfig, get_peft_model
import re
from models.codebook.utils import disable_early_lora


def _freeze_early_lora_params(model, inserted_layer):
    """Disable gradients for LoRA params that belong to layers **before** `inserted_layer`.

    Args:
        model: A PEFT wrapped model after `get_peft_model`.
        inserted_layer: 1-indexed position where Codebook is inserted. Layers < inserted_layer will be frozen.
    """
    layer_regex = re.compile(r"\.layers\.(\d+)")

    for n, p in model.named_parameters():
        if "lora_" not in n:
            continue

        match = layer_regex.search(n)
        if match is None:
            # Could not find layer index; default to enabling grad (safer)
            p.requires_grad = True
            continue

        layer_id = int(match.group(1))

        # Enable grads for layers ≥ inserted_layer, freeze earlier ones
        if (layer_id + 1) < inserted_layer:
            p.requires_grad = False
        else:
            p.requires_grad = True


def load_codebook_model(model, codebook_size, inserted_layer, select_len):
    codebook_config = CodebookConfig(  
        codebook_size=codebook_size,
        inserted_layer=inserted_layer,
        select_len=select_len,
    )

    # 1. Build LoRA config (only q_proj / v_proj / gate/up/down_proj as default).
    target_modules = [
        "q_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Attach LoRA **before** wrapping with CodebookAdapterModel to avoid peft_config conflicts.
    lora_base = get_peft_model(model, lora_cfg)

    # 2. Wrap the LoRA-augmented model with CodebookAdapterModel (adds Codebook & Refiner)
    model = CodebookAdapterModel(lora_base, codebook_config)

    # 3. Disable early-layer LoRA during training forward (still trainable ones remain)
    disable_early_lora(model, inserted_layer)

    # 4. Re-enable LoRA params for layers ≥ inserted_layer, freeze earlier ones
    _freeze_early_lora_params(model, inserted_layer)

    # NOTE: CodebookAdapterModel already ensured Codebook/Refiner params are trainable, and
    # underlying backbone params remain frozen. Only LoRA params adjusted here.

    return model
