#!/usr/bin/env python3
"""
Single-GPU evaluation script for codebook models
"""

import json
import argparse
import os
import torch
import transformers
from transformers.generation import GenerationConfig
from tqdm import tqdm
import warnings
import logging

# Suppress warnings and unnecessary output
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models import Qwen3ForCausalLM
from models import CodebookAdapterModel
from trainer import QwenEncoder
from generators import MATH_SIMPLE_ACTION_INSTRUCTION, PY_SIMPLE_ACTION_INSTRUCTION, OLYMPIAD_SIMPLE_ACTION_INSTRUCTION
from envs import programming_is_correct, math_reasoning_is_correct, olympiad_is_correct

MODEL_MAP = {
    'qwen3': 'Qwen/Qwen3-4B-Instruct-2507',
}

TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float
}

MODEL_TYPE = {
    'qwen3': Qwen3ForCausalLM
}

ENCODER_MAP = {
    'qwen3': QwenEncoder,
}

INSTRUCTION = {
    'math_reasoning': MATH_SIMPLE_ACTION_INSTRUCTION,
    'programming': PY_SIMPLE_ACTION_INSTRUCTION,
    'olympiad': OLYMPIAD_SIMPLE_ACTION_INSTRUCTION
}

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate trained codebook models")
    parser.add_argument(
        "-c", "--checkpoint-path", type=str, required=True,
        help="Trained codebook checkpoint path"
    )
    parser.add_argument(
        "-f", "--input-file", type=str, required=True,
        help="Input file, `.jsonl` format"
    )
    parser.add_argument(
        "-o", "--output-file", type=str, required=True,
        help="Output file, `.jsonl` format"
    )
    parser.add_argument(
        "-dtype", "--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"],
        help="torch dtype"
    )
    parser.add_argument(
        "-model", "--model", type=str, choices=["qwen3"], default="qwen3",
        help="Base LLM"
    )
    parser.add_argument(
        "-t", "--task", type=str, choices=["math_reasoning", "programming", "olympiad"],
        required=True, help="Task type"
    )
    parser.add_argument(
        "--max_token", type=int, default=30720,
        help="Maximum generation token length"
    )
    parser.add_argument(
        "--generation_timeout", type=int, default=7200,
        help="Generation timeout in seconds"
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0",
        help="GPU device ID to use"
    )

    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load model and tokenizer"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model_name = MODEL_MAP[args.model]
    torch_dtype = TORCH_DTYPE[args.dtype]
    model_cls = MODEL_TYPE[args.model]

    print(f"Loading base model: {model_name}")
    model = model_cls.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading codebook checkpoint: {args.checkpoint_path}")
    model = CodebookAdapterModel.from_pretrained(
        args.checkpoint_path,
        base_model=model,
        device_map="auto",
        torch_dtype=torch_dtype,
    )

    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, instruction, question, args):
    """Generate response for a single question"""
    prompt = f"{instruction}\n\n{question}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_token,
        do_sample=False,  # Use greedy decoding for deterministic results
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                timeout=args.generation_timeout
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

    except Exception as e:
        print(f"Generation error: {e}")
        return ""

def evaluate_response(response, item, task):
    """Evaluate if response is correct"""
    if task == 'math_reasoning':
        return math_reasoning_is_correct(response, item.get('answer', ''))
    elif task == 'programming':
        test_list = item.get('test_list', [])
        return programming_is_correct(response, test_list)
    elif task == 'olympiad':
        return olympiad_is_correct(response, item.get('answer', ''))
    else:
        raise NotImplementedError(f"Task {task} not supported")

def main():
    args = get_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Load dataset
    print(f"Loading dataset: {args.input_file}")
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} samples")

    # Get instruction for the task
    instruction = INSTRUCTION[args.task]

    # Evaluate
    results = []
    correct_count = 0

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i, item in enumerate(tqdm(data, desc="Evaluating")):
            question = item.get('question') or item.get('problem', '')

            if not question:
                print(f"Warning: Empty question in item {i}")
                continue

            # Generate response
            response = generate_response(model, tokenizer, instruction, question, args)

            # Evaluate correctness
            is_correct = evaluate_response(response, item, args.task)
            if is_correct:
                correct_count += 1

            # Save result
            result = {
                'question': question,
                'response': response,
                'is_correct': is_correct,
                'ground_truth': item.get('answer', ''),
                'index': i
            }

            results.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

    # Print final statistics
    total_samples = len(results)
    accuracy = correct_count / total_samples if total_samples > 0 else 0

    print(f"\n=== Evaluation Results ===")
    print(f"Total samples: {total_samples}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()