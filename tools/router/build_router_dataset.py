"""Build router training dataset by merging evaluation results from both chains."""

import json
import jsonlines
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    """Load JSONL file."""
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            # Skip statistics line if present
            if 'statistics' in obj or 'summary' in obj:
                continue
            data.append(obj)
    return data


def merge_evaluations(nonthink_path, think_path, output_path):
    """Merge evaluation results from non-thinking and thinking chains.
    
    Expected input format for each file:
    - Each line contains: question, instruction, acc/is_correct, token_count/new_tokens
    
    Output format:
    - Each line contains: instruction, question, a_non, a_think, ell_non, ell_think
    """
    # Load both evaluation results
    print(f"Loading non-thinking results from: {nonthink_path}")
    nonthink_data = load_jsonl(nonthink_path)
    
    print(f"Loading thinking results from: {think_path}")
    think_data = load_jsonl(think_path)
    
    # Build index for matching questions
    # Use (instruction, question) as key for matching
    nonthink_dict = {}
    for item in nonthink_data:
        # Handle different field names
        question = item.get('question', item.get('query', ''))
        instruction = item.get('instruction', '')
        
        # Get accuracy (handle different field names)
        if 'acc' in item:
            acc = 1 if item['acc'] else 0  # Handle both boolean and numeric
        elif 'is_correct' in item:
            acc = 1 if item['is_correct'] else 0
        elif 'correct' in item:
            acc = 1 if item['correct'] else 0
        else:
            print(f"Warning: No accuracy field found in non-thinking item: {item.keys()}")
            acc = 0
            
        # Get token count (handle different field names)
        if 'token_count' in item:
            tokens = item['token_count']
        elif 'new_tokens' in item:
            tokens = item['new_tokens']
        elif 'tokens' in item:
            tokens = item['tokens']
        elif 'len' in item:
            tokens = item['len']
        else:
            print(f"Warning: No token count field found in non-thinking item: {item.keys()}")
            tokens = 0
            
        key = (instruction.strip(), question.strip())
        nonthink_dict[key] = {
            'acc': acc,
            'tokens': tokens,
            'full_item': item
        }
    
    think_dict = {}
    for item in think_data:
        # Handle different field names
        question = item.get('question', item.get('query', ''))
        instruction = item.get('instruction', '')
        
        # Get accuracy
        if 'acc' in item:
            acc = 1 if item['acc'] else 0  # Handle both boolean and numeric
        elif 'is_correct' in item:
            acc = 1 if item['is_correct'] else 0
        elif 'correct' in item:
            acc = 1 if item['correct'] else 0
        else:
            print(f"Warning: No accuracy field found in thinking item: {item.keys()}")
            acc = 0
            
        # Get token count
        if 'token_count' in item:
            tokens = item['token_count']
        elif 'new_tokens' in item:
            tokens = item['new_tokens']
        elif 'tokens' in item:
            tokens = item['tokens']
        elif 'len' in item:
            tokens = item['len']
        else:
            print(f"Warning: No token count field found in thinking item: {item.keys()}")
            tokens = 0
            
        key = (instruction.strip(), question.strip())
        think_dict[key] = {
            'acc': acc,
            'tokens': tokens,
            'full_item': item
        }
    
    # Find matching questions
    matched_keys = set(nonthink_dict.keys()) & set(think_dict.keys())
    print(f"Found {len(matched_keys)} matching questions")
    print(f"Non-thinking only: {len(set(nonthink_dict.keys()) - matched_keys)}")
    print(f"Thinking only: {len(set(think_dict.keys()) - matched_keys)}")
    
    # Build merged dataset
    merged_data = []
    for key in sorted(matched_keys):
        instruction, question = key
        non_item = nonthink_dict[key]
        think_item = think_dict[key]
        
        merged_item = {
            'instruction': instruction,
            'question': question,
            'a_non': non_item['acc'],  # accuracy for non-thinking
            'a_think': think_item['acc'],  # accuracy for thinking
            'ell_non': non_item['tokens'],  # token count for non-thinking
            'ell_think': think_item['tokens'],  # token count for thinking
        }
        
        # Add optional metadata
        if 'id' in non_item['full_item']:
            merged_item['id'] = non_item['full_item']['id']
        
        merged_data.append(merged_item)
    
    # Save merged dataset
    print(f"Writing {len(merged_data)} samples to: {output_path}")
    with jsonlines.open(output_path, 'w') as writer:
        for item in merged_data:
            writer.write(item)
    
    # Print statistics
    print("\nDataset Statistics:")
    total = len(merged_data)
    both_correct = sum(1 for d in merged_data if d['a_non'] == 1 and d['a_think'] == 1)
    only_non = sum(1 for d in merged_data if d['a_non'] == 1 and d['a_think'] == 0)
    only_think = sum(1 for d in merged_data if d['a_non'] == 0 and d['a_think'] == 1)
    both_wrong = sum(1 for d in merged_data if d['a_non'] == 0 and d['a_think'] == 0)
    
    avg_tokens_non = sum(d['ell_non'] for d in merged_data) / total if total > 0 else 0
    avg_tokens_think = sum(d['ell_think'] for d in merged_data) / total if total > 0 else 0
    
    print(f"Both correct: {both_correct}/{total} ({100*both_correct/total:.1f}%)")
    print(f"Only non-thinking correct: {only_non}/{total} ({100*only_non/total:.1f}%)")
    print(f"Only thinking correct: {only_think}/{total} ({100*only_think/total:.1f}%)")
    print(f"Both wrong: {both_wrong}/{total} ({100*both_wrong/total:.1f}%)")
    print(f"Average tokens non-thinking: {avg_tokens_non:.1f}")
    print(f"Average tokens thinking: {avg_tokens_think:.1f}")
    print(f"Token ratio (think/non): {avg_tokens_think/avg_tokens_non:.2f}x")
    
    # Accuracy comparison
    acc_non = sum(d['a_non'] for d in merged_data) / total if total > 0 else 0
    acc_think = sum(d['a_think'] for d in merged_data) / total if total > 0 else 0
    print(f"\nAccuracy non-thinking: {acc_non:.3f}")
    print(f"Accuracy thinking: {acc_think:.3f}")
    print(f"Accuracy delta: {acc_think - acc_non:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Build router training dataset from evaluation results")
    parser.add_argument("--nonthink", type=str, required=True,
                        help="Path to non-thinking evaluation results (JSONL)")
    parser.add_argument("--think", type=str, required=True,
                        help="Path to thinking evaluation results (JSONL)")
    parser.add_argument("--out", type=str, default="router_ds.jsonl",
                        help="Output path for merged dataset")
    
    args = parser.parse_args()
    
    merge_evaluations(
        nonthink_path=args.nonthink,
        think_path=args.think,
        output_path=args.out
    )


if __name__ == "__main__":
    main()