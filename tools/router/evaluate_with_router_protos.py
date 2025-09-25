"""Evaluate with router-based chain selection."""

import os
import sys

# Set tokenizers parallelism to false before importing other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import jsonlines
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import transformers

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.codebook import CodebookAdapterModel
from tools.router.router_model import (
    RouterLite,
    extract_question_vector,
    extract_thinking_vectors,
    load_router
)
from envs import (
    programming_is_correct,
    math_is_correct, 
    math_reasoning_is_correct,
    olympiad_is_correct
)
from generators import MATH_SIMPLE_ACTION_INSTRUCTION, PY_SIMPLE_ACTION_INSTRUCTION, OLYMPIAD_SIMPLE_ACTION_INSTRUCTION

# Task-specific instructions (from evaluate_parallel.py)
INSTRUCTION_MAP = {
    'math': MATH_SIMPLE_ACTION_INSTRUCTION,
    'math_reasoning': MATH_SIMPLE_ACTION_INSTRUCTION,
    'programming': PY_SIMPLE_ACTION_INSTRUCTION,
    'olympiad': OLYMPIAD_SIMPLE_ACTION_INSTRUCTION  # Olympiad uses dedicated olympiad instruction
}

# Global variables to store loaded models (per process)
MODELS = {}

def init_worker(base_model_name, router_path, p_non_path, codebook_ckpt, dtype, device):
    """Initialize worker process with all required models."""
    global MODELS
    
    try:
        torch.cuda.set_device(device)
        device_str = f"cuda:{device}"
        torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16 if dtype == "bf16" else torch.float32
        
        
        # Load tokenizer for non-thinking path (will be reused)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load RouterLite model
        router = load_router(router_path, device=device_str)

        # Use FP32 for consistency with training
        router = router.float()

        router.eval()
        
        # Get config from checkpoint for router
        checkpoint = torch.load(router_path, map_location=device_str, weights_only=False)
        config = checkpoint['config']
        
        # Load CodebookAdapter model (contains base model internally + codebook layers)
        # Use Qwen3ForCausalLM like in evaluate_parallel.py for consistency
        from models import Qwen3ForCausalLM
        base_model_temp = Qwen3ForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_str,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        codebook_model = CodebookAdapterModel.from_pretrained(
            base_model_temp, codebook_ckpt
        ).to(device_str)
        codebook_model.eval()
        
        # Load thinking model - check for local model first
        import os

        # Check if local thinking model exists in parent directory
        current_dir = Path(__file__).parent.parent.parent  # Meta-qwen3 root
        local_thinking_path = current_dir / "Qwen_Qwen3-4B-Thinking-2507"

        if local_thinking_path.exists() and local_thinking_path.is_dir():
            thinking_model_path = str(local_thinking_path)
        else:
            thinking_model_path = "Qwen/Qwen3-4B-Thinking-2507"

        # Load thinking tokenizer
        thinking_tokenizer = transformers.AutoTokenizer.from_pretrained(
            thinking_model_path, trust_remote_code=True
        )
        if thinking_tokenizer.pad_token is None:
            thinking_tokenizer.pad_token = thinking_tokenizer.eos_token

        # Load thinking model
        thinking_model = Qwen3ForCausalLM.from_pretrained(
            thinking_model_path,
            device_map=device_str,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        )
        thinking_model.eval()
        
        # Create encoders for both paths
        from trainer.encoder import QwenEncoder
        # Non-thinking encoder: with codebook select_len (evaluate_parallel.py approach)
        # Get select_len from codebook model's peft_config, not router config
        encoder_non_thinking = QwenEncoder(tokenizer, codebook_model.peft_config.select_len)
        # Thinking encoder: no thinking tokens (evaluation_baseline_parallel.py approach)  
        encoder_thinking = QwenEncoder(thinking_tokenizer, select_len=0)
        
        # Store all models globally (base_model removed - it's contained within codebook_model)
        MODELS = {
            'tokenizer': tokenizer,
            'router': router,
            'router_config': config,
            'codebook_model': codebook_model,
            'thinking_model': thinking_model,
            'thinking_tokenizer': thinking_tokenizer,
            'encoder_non_thinking': encoder_non_thinking,
            'encoder_thinking': encoder_thinking,
            'device': device_str,
            'torch_dtype': torch_dtype
        }
        
        
    except Exception as e:
        print(f"Error initializing worker {device}: {str(e)}")
        raise


def filter_thinking_content(text):
    """Filter thinking content from response."""
    import re
    pattern = r'<think>.*?</think>'
    filtered_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    filtered_text = re.sub(r'\n{3,}', '\n\n', filtered_text)
    return filtered_text.strip()


def calculate_token_count(text, tokenizer):
    """Calculate token count for text (from evaluate_parallel.py)."""
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except:
        # If tokenizer not available, use simple estimation (4 chars = 1 token)
        return len(text) // 4


def evaluate_sample_worker(args):
    """Worker function for parallel evaluation - uses pre-loaded models."""
    global MODELS
    
    sample_idx, sample, task, max_new_tokens, max_new_tokens_thinking = args
    
    try:
        # Get models from global storage
        tokenizer = MODELS['tokenizer']
        router = MODELS['router']
        router_config = MODELS['router_config']
        codebook_model = MODELS['codebook_model']
        thinking_model = MODELS['thinking_model']
        thinking_tokenizer = MODELS['thinking_tokenizer']
        encoder_non_thinking = MODELS['encoder_non_thinking']
        encoder_thinking = MODELS['encoder_thinking']
        device = MODELS['device']
        
        # Get proper instruction for the task (from evaluate_parallel.py)
        instruction = INSTRUCTION_MAP.get(task, '')
        
        # Extract question text - compatible with different field names
        question = sample.get('question') or sample.get('problem', '') or sample.get('prompt', '')
        if not question:
            raise KeyError("Neither 'question' nor 'problem' nor 'prompt' field found in data")
        
        # Extract question vector using codebook_model (which contains the base model)
        q_vec = extract_question_vector(
            tokenizer, codebook_model, instruction, question, device=device
        )
        
        # Extract thinking vectors from non-thinking forward pass
        r_non = extract_thinking_vectors(
            tokenizer, codebook_model, instruction, question,
            codebook_model, max_len=2048, device=device
        )
        
        # Align dtype/device with router parameters before routing
        # Router is now in float32, so convert inputs accordingly
        rdtype = next(router.parameters()).dtype  # should be float32 now
        q_vec = q_vec.to(device).to(rdtype)
        r_non = r_non.to(device).to(rdtype)

        # Make routing decision using RouterLite
        use_think, scores = router.compute_scores(q_vec, r_non, router.len_scale)
        
        # Build question text (following evaluate_parallel.py format)
        question_text = f"\n[Question]: {question}"
        if task == "programming":
            test_code = sample.get('test_code', '')
            if test_code:
                question_text += f"Your code should pass these tests: {test_code}"
        
        # Generate response based on routing
        if use_think:
            # Use thinking chain (evaluation_baseline_parallel.py approach)
            chain_type = "thinking"
            
            # Encode input with thinking encoder (select_len=0, no thinking tokens)
            inputs = encoder_thinking.encode_inference(instruction, question_text)
            
            # Only keep keys accepted by standard HF generate/forward APIs
            cur_len = len(inputs[0]["input_ids"])
            allowed = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
            model_inputs = {
                k: torch.LongTensor(v).unsqueeze(0).to(device)
                for k, v in inputs[0].items()
                if k in allowed
            }
            # Ensure attention_mask exists
            if "attention_mask" not in model_inputs:
                model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
            
            # Set max_length like in evaluation_baseline_parallel.py
            max_length = cur_len + max_new_tokens_thinking
            
            with torch.no_grad():
                outputs = thinking_model.generate(
                    **model_inputs,
                    eos_token_id=[thinking_tokenizer.eos_token_id, thinking_tokenizer.pad_token_id],
                    max_new_tokens=max_new_tokens_thinking,
                    max_length=max_length,  # Hard limit on maximum length
                    do_sample=False,
                    pad_token_id=thinking_tokenizer.pad_token_id,
                    early_stopping=True,  # Add early stopping to speed up termination
                    num_beams=1,  # Use greedy search for speed
                )
            
            response = thinking_tokenizer.decode(outputs[0].tolist()[cur_len:], skip_special_tokens=True)
            actual_new_tokens = len(outputs[0].tolist()) - cur_len
            truncated_by_length = actual_new_tokens >= max_new_tokens_thinking
            
        else:
            # Use non-thinking chain (CodebookAdapter) - evaluate_parallel.py approach
            chain_type = "non-thinking"
            
            # Encode input with proper masks using non-thinking encoder
            inputs = encoder_non_thinking.encode_inference(instruction, question_text)
            cur_len = len(inputs[0]['input_ids'])
            
            # Prepare batch
            inputs_tensor = {k: torch.LongTensor(v).unsqueeze(0).to(device) for k, v in inputs[0].items()}
            
            # Set max_length like in evaluate_parallel.py
            max_length = cur_len + max_new_tokens
            
            # Generate
            with torch.no_grad():
                outputs = codebook_model.generate(
                    **inputs_tensor,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,  # Hard limit on maximum length
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    early_stopping=True,  # Add early stopping to speed up termination
                    num_beams=1,  # Use greedy search for speed
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0].tolist()[cur_len:], skip_special_tokens=True)
            actual_new_tokens = len(outputs[0].tolist()) - cur_len
            truncated_by_length = actual_new_tokens >= max_new_tokens
        
        # Filter thinking content from response
        filtered_response = filter_thinking_content(response)
        
        # Calculate token count (including thinking content, like in evaluate_parallel.py)
        tokenizer_for_count = thinking_tokenizer if use_think else tokenizer
        token_count = calculate_token_count(response, tokenizer_for_count)
        
        # Evaluate response
        if task == "programming":
            # Extract code from response
            from generators.generator_utils import parse_code_block
            func = parse_code_block(filtered_response)
            is_correct = programming_is_correct(func, sample.get('test_code', ''))
        elif task == "math":
            is_correct = math_is_correct(filtered_response, sample.get('answer', ''))
        elif task == "math_reasoning":
            is_correct = math_reasoning_is_correct(filtered_response, sample.get('answer', ''))
        elif task == "olympiad":
            # Extract olympiad-specific fields
            final_answer = sample.get('final_answer', [])
            if isinstance(final_answer, list):
                ground_truth = final_answer[0] if final_answer else ""
            else:
                ground_truth = final_answer
            
            answer_type = sample.get('answer_type', None)
            unit = sample.get('unit', None)
            
            is_correct = olympiad_is_correct(filtered_response, ground_truth, 
                                           answer_type=answer_type, unit=unit)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Prepare result (following evaluate_parallel.py format)
        result = dict(sample)  # Keep all original fields
        result['completion'] = response  # Raw response with thinking
        result['token_count'] = token_count  # Token count including thinking
        result['truncated_by_length'] = truncated_by_length
        result['acc'] = is_correct
        
        # Additional router-specific fields
        result['chain_type'] = chain_type
        result['routing_scores'] = scores
        
        # Add idx for sorting (will be removed before saving)
        result['_idx'] = sample_idx
        
        return result
        
    except Exception as e:
        import traceback
        print(f"Error processing sample {sample_idx}: {str(e)}")
        
        # Return error result in evaluate_parallel.py format
        result = dict(sample) if sample else {}
        result['completion'] = ""
        result['token_count'] = 0
        result['truncated_by_length'] = False
        result['acc'] = False
        result['error'] = str(e)
        result['chain_type'] = 'error'
        result['_idx'] = sample_idx
        
        return result


def evaluate_sample_worker_with_init(sample_args, init_args):
    """Worker function that initializes models on first call."""
    global MODELS
    
    # Initialize models if not already done
    if not MODELS:
        base_model_name, router_path, p_non_path, codebook_ckpt, dtype, device = init_args
        init_worker(base_model_name, router_path, p_non_path, codebook_ckpt, dtype, device)
    
    # Call the regular worker function
    return evaluate_sample_worker(sample_args)


def gpu_evaluate_sample_worker(gpu_args):
    """Worker function that handles GPU-specific initialization and sample processing."""
    global MODELS
    
    # Extract sample args + GPU ID
    sample_idx, sample, task, max_new_tokens, max_new_tokens_thinking, gpu_id = gpu_args
    
    try:
        # Initialize models if not already done for this process
        if not MODELS:
            # Get the initialization arguments from the main process
            # We'll need to pass these through the environment or reconstruct them
            import os
            
            # These should be set as environment variables by the main process
            base_model_name = os.environ.get('ROUTER_BASE_MODEL', 'Qwen/Qwen3-4B-Instruct-2507')
            router_path = os.environ.get('ROUTER_PATH')
            p_non_path = os.environ.get('ROUTER_P_NON_PATH') 
            codebook_ckpt = os.environ.get('ROUTER_CODEBOOK_CKPT')
            dtype = os.environ.get('ROUTER_DTYPE', 'bf16')
            
            # p_non_path is optional for RouterLite, but router_path and codebook_ckpt are required
            if not all([router_path, codebook_ckpt]):
                raise ValueError(f"Required environment variables not set")
                
            init_worker(base_model_name, router_path, p_non_path, codebook_ckpt, dtype, gpu_id)
        
        # Call the regular worker function with original args format
        original_args = (sample_idx, sample, task, max_new_tokens, max_new_tokens_thinking)
        return evaluate_sample_worker(original_args)
        
    except Exception as e:
        import traceback
        print(f"Error in GPU worker {gpu_id} for sample {sample_idx}: {e}")
        return {
            'completion': "",
            'token_count': 0,
            'truncated_by_length': False,
            'acc': False,
            'error': str(e),
            'chain_type': 'error',
            '_idx': sample_idx
        }


def evaluate_with_router(args):
    """Main evaluation function with routing."""
    
    # Load test data
    print(f"Loading test data from: {args.file}")
    test_data = []
    with jsonlines.open(args.file) as reader:
        for obj in reader:
            test_data.append(obj)
    print(f"Loaded {len(test_data)} samples")
    
    # Prepare GPU assignment
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = [0]
    
    processes_per_gpu = args.processes_per_gpu
    total_processes = len(gpu_ids) * processes_per_gpu
    
    # Prepare arguments for workers
    worker_args = []
    for idx, sample in enumerate(test_data):
        worker_args.append((
            idx, sample, args.task, args.max_new_tokens, args.max_new_tokens_thinking
        ))
    
    # Set environment variables for worker processes
    import os
    os.environ['ROUTER_BASE_MODEL'] = args.base_model
    os.environ['ROUTER_PATH'] = args.router
    os.environ['ROUTER_P_NON_PATH'] = args.p_non if args.p_non is not None else ""
    os.environ['ROUTER_CODEBOOK_CKPT'] = args.codebook_ckpt
    os.environ['ROUTER_DTYPE'] = args.dtype
    
    print(f"\n{'='*80}")
    print(f"Starting Router-based Evaluation")
    print(f"{'='*80}")
    print(f"📊 Configuration:")
    print(f"  - Total samples: {len(test_data)}")
    print(f"  - Task type: {args.task}")
    print(f"  - Max tokens (non-thinking): {args.max_new_tokens}")
    print(f"  - Max tokens (thinking): {args.max_new_tokens_thinking}")
    print(f"  - GPUs: {gpu_ids} ({total_processes} processes)")
    print(f"  - GPU assignment: round-robin")
    print(f"{'='*80}\n")
    
    # Create worker function with GPU assignment embedded in the task
    def create_gpu_assigned_worker_args(worker_args, gpu_ids, processes_per_gpu):
        """Assign GPU IDs to worker arguments."""
        gpu_assigned_args = []
        total_processes = len(gpu_ids) * processes_per_gpu
        
        for i, args in enumerate(worker_args):
            # Assign GPU in round-robin fashion across all processes
            process_id = i % total_processes
            gpu_id = gpu_ids[process_id // processes_per_gpu]
            # Add GPU ID to the arguments
            gpu_assigned_args.append(args + (gpu_id,))
        return gpu_assigned_args
    
    # Create GPU-assigned worker arguments
    gpu_worker_args = create_gpu_assigned_worker_args(worker_args, gpu_ids, processes_per_gpu)
    
    # Run evaluation in parallel with proper GPU assignment
    results = []
    
    # Real-time statistics tracking
    stats = {
        'thinking': {'count': 0, 'correct': 0, 'tokens': []},
        'non_thinking': {'count': 0, 'correct': 0, 'tokens': []}, 
        'error': {'count': 0}
    }
    
    
    with Pool(
        processes=total_processes,
        initializer=None,  # No shared initialization - each worker will init its own GPU
        initargs=None
    ) as pool:
        # Use imap_unordered for real-time result processing
        # Create progress bar
        pbar = tqdm(total=len(gpu_worker_args), desc="Evaluating...")
        
        # Process results as they complete
        for result in pool.imap_unordered(gpu_evaluate_sample_worker, gpu_worker_args):
            results.append(result)
            pbar.update(1)
            
            # Update statistics based on result
            chain_type = result.get('chain_type', 'error')
            token_count = result.get('token_count', 0)
            
            if chain_type == 'thinking':
                stats['thinking']['count'] += 1
                if result.get('acc', False):
                    stats['thinking']['correct'] += 1
                if token_count > 0:
                    stats['thinking']['tokens'].append(token_count)
            elif chain_type == 'non-thinking':
                stats['non_thinking']['count'] += 1
                if result.get('acc', False):
                    stats['non_thinking']['correct'] += 1
                if token_count > 0:
                    stats['non_thinking']['tokens'].append(token_count)
            else:
                stats['error']['count'] += 1
            
            # Calculate and display statistics
            think_acc = (stats['thinking']['correct'] / stats['thinking']['count'] * 100) if stats['thinking']['count'] > 0 else 0
            non_think_acc = (stats['non_thinking']['correct'] / stats['non_thinking']['count'] * 100) if stats['non_thinking']['count'] > 0 else 0
            overall_correct = stats['thinking']['correct'] + stats['non_thinking']['correct']
            overall_total = stats['thinking']['count'] + stats['non_thinking']['count']
            overall_acc = (overall_correct / overall_total * 100) if overall_total > 0 else 0
            
            think_avg_tokens = np.mean(stats['thinking']['tokens']) if stats['thinking']['tokens'] else 0
            non_think_avg_tokens = np.mean(stats['non_thinking']['tokens']) if stats['non_thinking']['tokens'] else 0
            
            # Display progress information
            if len(results) > 0:
                desc = (f"T: {stats['thinking']['count']}({think_acc:.1f}%,{think_avg_tokens:.0f}tok) | "
                       f"NT: {stats['non_thinking']['count']}({non_think_acc:.1f}%,{non_think_avg_tokens:.0f}tok) | "
                       f"Acc: {overall_acc:.1f}%")
                if stats['error']['count'] > 0:
                    desc += f" | Err: {stats['error']['count']}"
            else:
                desc = "Starting evaluation..."
            
            pbar.set_description(desc)
        
        pbar.close()
    
    # Sort results by index
    results.sort(key=lambda x: x.get('_idx', 0))
    
    # Calculate statistics (using 'acc' field like evaluate_parallel.py)
    correct = sum(1 for r in results if r.get('acc', False))
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Chain usage statistics
    thinking_count = sum(1 for r in results if r.get('chain_type') == 'thinking')
    non_thinking_count = sum(1 for r in results if r.get('chain_type') == 'non-thinking')
    error_count = sum(1 for r in results if r.get('chain_type') == 'error')
    
    # Token statistics (using 'token_count' field like evaluate_parallel.py)
    valid_samples = [r for r in results if r.get('token_count', 0) > 0]
    total_tokens = sum(r.get('token_count', 0) for r in valid_samples)
    avg_tokens = total_tokens / len(valid_samples) if valid_samples else 0
    
    thinking_tokens = [r.get('token_count', 0) for r in results 
                      if r.get('chain_type') == 'thinking' and r.get('token_count', 0) > 0]
    non_thinking_tokens = [r.get('token_count', 0) for r in results 
                          if r.get('chain_type') == 'non-thinking' and r.get('token_count', 0) > 0]
    
    avg_thinking_tokens = np.mean(thinking_tokens) if thinking_tokens else 0
    avg_non_thinking_tokens = np.mean(non_thinking_tokens) if non_thinking_tokens else 0
    
    # Accuracy by chain type (using 'acc' field)
    thinking_correct = sum(1 for r in results if r.get('chain_type') == 'thinking' and r.get('acc', False))
    non_thinking_correct = sum(1 for r in results if r.get('chain_type') == 'non-thinking' and r.get('acc', False))
    
    # Routing score statistics
    delta_accs = [r.get('routing_scores', {}).get('delta_acc', 0.0) for r in results if 'routing_scores' in r]
    margins = [r.get('routing_scores', {}).get('margin', 0.0) for r in results if 'routing_scores' in r]
    avg_delta_acc = np.mean(delta_accs) if delta_accs else 0
    avg_margin = np.mean(margins) if margins else 0
    
    # Print statistics
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Overall accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"Chain usage - Thinking: {thinking_count} ({100*thinking_count/total:.1f}%), Non-thinking: {non_thinking_count} ({100*non_thinking_count/total:.1f}%)")
    if thinking_count > 0:
        print(f"Thinking accuracy: {thinking_correct/thinking_count:.1%} ({thinking_correct}/{thinking_count})")
    if non_thinking_count > 0:
        print(f"Non-thinking accuracy: {non_thinking_correct/non_thinking_count:.1%} ({non_thinking_correct}/{non_thinking_count})")
    print(f"Avg tokens: {avg_tokens:.1f} (thinking: {avg_thinking_tokens:.1f}, non-thinking: {avg_non_thinking_tokens:.1f})")
    
    # Save results
    with jsonlines.open(args.output, 'w') as writer:
        for result in results:
            # Remove _idx field before saving (internal sorting field)
            result_copy = {k: v for k, v in result.items() if k != '_idx'}
            writer.write(result_copy)
    
    # Append summary statistics (following evaluate_parallel.py format)
    summary = {
        'accuracy': round(accuracy, 2),  # Round to 2 decimal places
        'average_tokens': round(avg_tokens, 1),
        'total_tokens': total_tokens,
        'chain_usage': {
            'thinking': thinking_count,
            'non_thinking': non_thinking_count,
            'errors': error_count
        },
        'token_stats': {
            'thinking_avg': round(avg_thinking_tokens, 1),
            'non_thinking_avg': round(avg_non_thinking_tokens, 1)
        },
        'routing_stats': {
            'avg_delta_acc': round(avg_delta_acc, 4),
            'avg_margin': round(avg_margin, 4)
        },
        'chain_accuracy': {
            'thinking': round(thinking_correct/thinking_count, 3) if thinking_count > 0 else 0,
            'non_thinking': round(non_thinking_correct/non_thinking_count, 3) if non_thinking_count > 0 else 0
        }
    }

    with open(args.output, 'a') as f:
        f.write(json.dumps(summary) + '\n')

    print(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate with router-based chain selection")
    
    # Input/output arguments
    parser.add_argument("-f", "--file", type=str, required=True,
                        help="Path to test data (JSONL)")
    parser.add_argument("-o", "--output", type=str, default="routed_results.jsonl",
                        help="Output path for results")
    
    # Model arguments
    parser.add_argument("--router", type=str, required=True,
                        help="Path to trained router model")
    parser.add_argument("--p_non", type=str, required=False, default=None,
                        help="Path to non-thinking prototypes (not used in RouterLite)")
    parser.add_argument("--codebook_ckpt", type=str, required=True,
                        help="Path to CodebookAdapter checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base model name")
    
    # Removed eps_len argument as tie-breaking mechanism is no longer used
    
    # Task arguments
    parser.add_argument("-t", "--task", type=str, default="programming",
                        choices=["math", "programming", "math_reasoning", "olympiad"],
                        help="Task type")
    parser.add_argument("--max_new_tokens", type=int, default=30720,
                        help="Maximum new tokens for non-thinking chain (default: 30720)")
    parser.add_argument("--max_new_tokens_thinking", type=int, default=30720,
                        help="Maximum new tokens for thinking chain (default: 30720)")
    
    # Parallelization arguments
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated GPU IDs to use")
    parser.add_argument("--processes_per_gpu", type=int, default=1,
                        help="Number of processes per GPU")
    
    # Other arguments
    parser.add_argument("-dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Data type for model")
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    evaluate_with_router(args)


if __name__ == "__main__":
    main()