from generators.instruction import *
from generators.model import Message
from typing import Union, List, Optional, Callable, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from .model import generate_chat
import re

def generate_action(
    question: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    strategy: str,
    task: str,
    self_thinking=None,
    max_tokens: int = 2048,
    temperature: float = 0,
    top_p: float = 0,
    top_k: int = 1,
    do_sample: bool = False
) -> Union[str, List[str]]:
    if strategy != "thinking" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `thinking` or `simple`")
    if strategy == "thinking" and self_thinking is None:
        raise ValueError(
            f"Invalid arguments: given `strategy=thinking` but `self_thinking` is None")

    if task == 'math':
        INSTRUCTION = MATH_SIMPLE_ACTION_INSTRUCTION
    elif task == 'programming':
        INSTRUCTION = PY_SIMPLE_ACTION_INSTRUCTION


    if strategy == "thinking":
        system_prompt = INSTRUCTION
        question_input = f"Here are the question and correponding thinking from past trials:\n[Question]: {question}\[Thinking]: {self_thinking}"
    else:
        system_prompt = INSTRUCTION
        question_input = f"[Question]: {question}"
        
    messages = [
            Message(
                role="system",
                content=system_prompt,
            ),
            Message(
                role="user",
                content=question_input
            )
        ]

    response = generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample
    )
    
    return response


def generate_self_thinking(
        question: str,
        feedbacks: Tuple[bool, str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_tokens: int = 2048,
        temperature: float = 0,
        top_p: float = 0,
        top_k: int = 1,
        do_sample: bool = False
) -> str:
    if task == 'math':
        system_prompt = SELF_MATH_INSTRUCTION
    elif task == 'programming':
        system_prompt = SELF_PROGRAMMING_INSTRUCTION
    else:
        system_prompt = SELF_MATH_INSTRUCTION  # Default fallback
    
    
    feedback = feedbacks[1]

    thinking_input = f"[Question]: {question}\n[Trials and Feedback]: {feedback}\n"

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content=thinking_input
        )
    ]

    response = generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample
    )

    return response  



def parse_code_block(string: str) -> Optional[str]:
    """
    Extract code body for programming task.
    """
    # First try to find python code blocks
    code_pattern = r"```python\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)

    if match:
        return match.group(1).strip()

    # Try generic code blocks
    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)

    if match:
        return match.group(1).strip()

    # If no code blocks found, treat the entire string as code
    # This handles cases where model outputs raw code without markdown formatting
    cleaned_string = string.strip()
    
    # Remove common prefixes that might interfere with code execution
    prefixes_to_remove = [
        "Here's the solution:",
        "Here's the code:",
        "Solution:",
        "Code:",
        "Answer:",
        "The function is:",
        "def:",
    ]
    
    for prefix in prefixes_to_remove:
        if cleaned_string.lower().startswith(prefix.lower()):
            cleaned_string = cleaned_string[len(prefix):].strip()
            break
    
    return cleaned_string