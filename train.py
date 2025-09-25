import os
import json
import argparse
import random
import numpy as np
import torch
import transformers
from transformers import TrainingArguments, TrainerCallback
from transformers import set_seed as transformers_set_seed

from trainer.model import load_codebook_model, _freeze_early_lora_params
from trainer import CustomDataset
from models import Qwen3ForCausalLM
from models.codebook.utils import disable_all_lora

def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")

MODEL_MAP = {
    "qwen3": "Qwen/Qwen3-4B-Instruct-2507",
}

TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float,
}

MODEL_TYPE = {
    "qwen3": Qwen3ForCausalLM,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", type=str, choices=["qwen3"], default="qwen3", help="Base LLM")
    parser.add_argument("-dtype", "--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16", help="torch dtype")
    parser.add_argument("-codebook_size", "--codebook_size", default=512, type=int, help="Size of codebook")
    parser.add_argument("-select_len", "--select_len", default=32, type=int, help="Length of thinking units")
    parser.add_argument("-inserted_layer", "--inserted_layer", default=25, type=int, help="Position of inserted layers")
    parser.add_argument("-lr1", "--lr1", type=float, default=5e-5, help="Learning rate for stage 1")
    parser.add_argument("-lr2", "--lr2", type=float, default=1e-5, help="Learning rate for stage 2")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=4)
    parser.add_argument("-gradient_accumulation_steps", "--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("-epochs1", "--epochs1", type=int, default=2, help="Epochs for stage 1")
    parser.add_argument("-epochs2", "--epochs2", type=int, default=3, help="Epochs for stage 2")
    parser.add_argument("-source_file", "--source_file", type=str, help="Training dataset file (.jsonl)")
    parser.add_argument("-save_path", "--save_path", type=str, help="Directory to save checkpoints")
    parser.add_argument("-max_length", "--max_length", type=int, default=3000, help="Maximum sequence length")
    parser.add_argument("--max_tokens", type=int, default=30720, help="Maximum tokens limit for training sequences")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA adapters during training & inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_lr_multipliers", action="store_true", help="Disable component-specific learning rate multipliers")
    parser.add_argument("--lr_config", type=str, default="lr_multipliers.json", help="Path to learning rate multiplier configuration")
    parser.add_argument("--lr_preset", type=str, choices=["aggressive", "conservative", "balanced"], default="balanced", help="Preset learning rate strategy")
    return parser.parse_args()

class LogToJSONLCallback(TrainerCallback):
    def __init__(self, output_file: str):
        self.output_file = output_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.output_file, "a") as f:
                f.write(json.dumps(logs) + "\n")

if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)

    print(f"Training configuration:")
    print(f"   Single GPU training")
    print(args)

    model_name = MODEL_MAP[args.model]
    torch_dtype = TORCH_DTYPE[args.dtype]
    model_cls = MODEL_TYPE[args.model]

    print("Loading model...")
    model = model_cls.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Injecting Codebook module...")
    model = load_codebook_model(
        model,
        codebook_size=args.codebook_size,
        inserted_layer=args.inserted_layer,
        select_len=args.select_len,
    )

    # Optionally disable LoRA entirely
    if args.no_lora:
        disable_all_lora(model)

    # Freeze ALL LoRA params during Stage-1 (alignment)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = False

    print("Loading reference model...")
    ref_model = model_cls.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Load dataset
    data = []
    with open(args.source_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    data = sorted(data, key=lambda x: x.get('question', ''))
    print(f"Loaded {len(data)} samples from {args.source_file}")

    dataset = CustomDataset(data, tokenizer, args.select_len, max_length=args.max_tokens)

    # Setup save path
    save_path = args.save_path or f"saves/{args.model}/layer_{args.inserted_layer}_codebooksize_{args.codebook_size}_selectlen_{args.select_len}/"
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, "logs.jsonl")
    if os.path.exists(log_file):
        os.remove(log_file)
    callback = LogToJSONLCallback(log_file)

    # Stage 1 Training
    print("#### Stage 1 Training ####")
    stage1_args = TrainingArguments(
        output_dir="./saves",
        overwrite_output_dir=True,
        report_to="none",
        bf16=(args.dtype == "bf16"),
        dataloader_pin_memory=False,
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type="linear",
        save_strategy="no",
        optim="adamw_torch_fused",
        warmup_ratio=0.1,
        learning_rate=args.lr1,
        num_train_epochs=args.epochs1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 3,
        gradient_checkpointing=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        weight_decay=0.01,
    )

    from trainer import CustomTrainer

    # Check if using component-level learning rates
    use_lr_multipliers = not args.no_lr_multipliers
    if use_lr_multipliers and not os.path.exists(args.lr_config):
        print(f"Warning: Learning rate config file {args.lr_config} not found, using default learning rate")
        use_lr_multipliers = False

    stage1_trainer = CustomTrainer(
        model=model,
        ref_model=ref_model,
        args=stage1_args,
        data_collator=dataset.collate_fn,
        train_dataset=dataset,
        tokenizer=tokenizer,
        mode="alignment",
        callbacks=[callback],
        use_lr_multipliers=use_lr_multipliers,
        lr_config_path=args.lr_config,
        inserted_layer=args.inserted_layer,
    )

    stage1_trainer.train()
    torch.cuda.empty_cache()

    # Stage 2: unfreeze LoRA for layers >= inserted_layer
    if not args.no_lora:
        _freeze_early_lora_params(model, args.inserted_layer)
        n_lora_trainable = sum(p.requires_grad for n,p in model.named_parameters() if "lora_" in n)
        print(f"Trainable LoRA params: {n_lora_trainable}")

    # Free reference model
    del ref_model
    torch.cuda.empty_cache()

    print("#### Stage 2 Training ####")
    stage2_args = TrainingArguments(
        output_dir="./saves",
        overwrite_output_dir=True,
        report_to="none",
        bf16=(args.dtype == "bf16"),
        dataloader_pin_memory=False,
        logging_strategy="steps",
        logging_steps=10,
        lr_scheduler_type="linear",
        save_strategy="no",
        optim="adamw_torch_fused",
        warmup_ratio=0.1,
        learning_rate=args.lr2,
        num_train_epochs=args.epochs2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 3,
        gradient_checkpointing=False,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        weight_decay=0.005,
    )

    stage2_trainer = CustomTrainer(
        model=model,
        ref_model=None,
        args=stage2_args,
        data_collator=dataset.collate_fn,
        train_dataset=dataset,
        tokenizer=tokenizer,
        mode="sft",
        callbacks=[callback],
        use_lr_multipliers=use_lr_multipliers,
        lr_config_path=args.lr_config,
        inserted_layer=args.inserted_layer,
    )

    stage2_trainer.train()

    # Save model
    stage2_trainer.model.save_pretrained(save_path)
    stage2_trainer.tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print("Training completed! Model saved to:", save_path)