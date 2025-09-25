#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-process CoT data generation script using API models
Generates Chain-of-Thought reasoning data for training without multithreading
"""

import jsonlines
import json
import argparse
import os
from typing import List, Dict, Optional
import time
from tqdm import tqdm
from generators.instruction import *
from envs import math_reasoning_get_feedback, programming_get_feedback, olympiad_get_feedback
import warnings
import logging
from openai import OpenAI
import httpx

# Suppress warnings and unnecessary output
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key", type=str, required=True,
        help="API Key for Qwen-plus API (required)"
    )
    parser.add_argument(
        "--task", type=str, choices=["math_reasoning", "programming", "olympiad"],
        help="Task type"
    )
    parser.add_argument(
        "--input_file", type=str, default="benchmark/math/train.jsonl",
        help="Input file, `.jsonl` format"
    )
    parser.add_argument(
        "--output_file", type=str, default="datasets_cot/cot_dataset.jsonl",
        help="Output file, `.jsonl` format"
    )
    parser.add_argument(
        "--max_samples", type=int, default=1000,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=30720,
        help="Maximum tokens for API generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="API request timeout in seconds"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Maximum number of retry attempts for API calls"
    )

    return parser.parse_args()

def call_qwen_api(messages, api_key, max_tokens=2048, temperature=0.7, timeout=30):
    """
    Call Qwen API using OpenAI-compatible mode
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        http_client=httpx.Client(timeout=timeout)
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen-max-latest",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"API request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 3 + attempt * 2  # Increasing wait time
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"API request finally failed, returning empty string")
                return ""

    return ""

def generate_cot_thinking(question: str, api_key: str, task: str, args) -> str:
    """Generate Chain-of-Thought reasoning using API"""

    # Select appropriate CoT instruction based on task
    if task == "math_reasoning":
        instruction = MATH_COT_INSTRUCTION
    elif task == "programming":
        instruction = PY_COT_INSTRUCTION
    elif task == "olympiad":
        instruction = OLYMPIAD_COT_INSTRUCTION
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Create messages for API call
    messages = [
        {
            "role": "system",
            "content": "You are an expert problem solver. Generate detailed step-by-step reasoning."
        },
        {
            "role": "user",
            "content": f"{instruction}\n\nProblem: {question}\n\nReasoning:"
        }
    ]

    # Call API
    try:
        response = call_qwen_api(
            messages,
            api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout
        )
        return response.strip()

    except Exception as e:
        print(f"Generation error: {e}")
        return ""

def generate_final_answer(question: str, cot_thinking: str, api_key: str, task: str, args) -> str:
    """Generate final answer based on CoT reasoning"""

    # Get task-specific instruction
    if task == "math_reasoning":
        instruction = MATH_SIMPLE_ACTION_INSTRUCTION
    elif task == "programming":
        instruction = PY_SIMPLE_ACTION_INSTRUCTION
    elif task == "olympiad":
        instruction = OLYMPIAD_SIMPLE_ACTION_INSTRUCTION
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Create prompt for final answer
    cot_prompt = f"{instruction}\n\n{question}\n\nReasoning: {cot_thinking}\n\nAnswer:"

    messages = [
        {
            "role": "system",
            "content": "You are an expert problem solver. Provide the final answer based on the reasoning."
        },
        {
            "role": "user",
            "content": cot_prompt
        }
    ]

    try:
        response = call_qwen_api(
            messages,
            api_key,
            max_tokens=256,  # Shorter for final answer
            temperature=0.0,  # Deterministic for final answer
            timeout=args.timeout
        )
        return response.strip()

    except Exception as e:
        print(f"Final answer generation error: {e}")
        return ""

def get_task_instruction(task: str) -> str:
    """Get the appropriate simple instruction for the task"""
    if task == "math_reasoning":
        return MATH_SIMPLE_ACTION_INSTRUCTION
    elif task == "programming":
        return PY_SIMPLE_ACTION_INSTRUCTION
    elif task == "olympiad":
        return OLYMPIAD_SIMPLE_ACTION_INSTRUCTION
    else:
        raise ValueError(f"Unsupported task: {task}")

def validate_cot_data(cot_thinking: str, question: str, answer: str, task: str) -> bool:
    """Simple validation of CoT data quality"""
    if not cot_thinking or len(cot_thinking.strip()) < 10:
        return False

    # Basic checks for reasoning content
    if task in ["math_reasoning", "olympiad"]:
        # Should contain some mathematical reasoning keywords
        math_keywords = ["calculate", "solve", "equation", "formula", "therefore", "because", "since", "step"]
        if not any(keyword in cot_thinking.lower() for keyword in math_keywords):
            return False

    elif task == "programming":
        # Should contain some programming reasoning keywords
        prog_keywords = ["function", "algorithm", "implementation", "code", "return", "variable", "loop", "condition"]
        if not any(keyword in cot_thinking.lower() for keyword in prog_keywords):
            return False

    return True

def main():
    args = get_args()

    print(f"Using API-based CoT generation for task: {args.task}")
    print(f"API Key provided: {'Yes' if args.api_key else 'No'}")

    # Load input data
    print(f"Loading data from: {args.input_file}")
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Limit samples if specified
    if args.max_samples > 0:
        data = data[:args.max_samples]

    print(f"Processing {len(data)} samples")

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Process data
    successful_generations = 0
    failed_generations = 0

    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for i, item in enumerate(tqdm(data, desc="Generating CoT data")):
            question = item.get('question') or item.get('problem', '')
            answer = item.get('answer', '')

            if not question:
                print(f"Warning: Empty question in item {i}")
                failed_generations += 1
                continue

            # Generate CoT thinking
            retry_count = 0
            max_retries = args.max_retries

            while retry_count < max_retries:
                # Generate CoT reasoning
                cot_thinking = generate_cot_thinking(question, args.api_key, args.task, args)

                if not cot_thinking:
                    print(f"Warning: Empty CoT generated for item {i}, retry {retry_count + 1}")
                    retry_count += 1
                    time.sleep(2)  # Wait before retry
                    continue

                # Validate generated CoT
                if validate_cot_data(cot_thinking, question, answer, args.task):
                    break
                else:
                    print(f"Warning: Invalid CoT generated for item {i}, retry {retry_count + 1}")
                    retry_count += 1
                    time.sleep(2)
                    continue

            if retry_count >= max_retries:
                print(f"Failed to generate valid CoT for item {i} after {max_retries} retries")
                failed_generations += 1
                continue

            # Create training sample
            training_sample = {
                'question': question,
                'thinking': cot_thinking,
                'answer': answer,
                'instruction': get_task_instruction(args.task),
                'task': args.task,
                'original_index': i
            }

            # Save to output
            f_out.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
            f_out.flush()

            successful_generations += 1

            # Add small delay to respect API rate limits
            time.sleep(1.2)

    # Print statistics
    print(f"\n=== CoT Data Generation Complete ===")
    print(f"Total input samples: {len(data)}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    print(f"Success rate: {successful_generations/len(data)*100:.2f}%")
    print(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main()