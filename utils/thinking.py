from generators.generator_utils import generate_action, generate_self_thinking
from generators.instruction import *
from typing import List
import os
import json
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from envs import programming_get_feedback, math_reasoning_get_feedback, olympiad_get_feedback

def run_thinking(
    dataset: List[dict],
    max_iters: int,
    output_file: str,
    actor_model: PreTrainedModel,
    actor_tokenizer: PreTrainedTokenizer,
    thinker_model: PreTrainedModel,
    thinker_tokenizer: PreTrainedTokenizer,
    task: str
) -> None:
    f_o = open(output_file, 'w')

    for idx, item in tqdm(enumerate(dataset)):
        is_solved = False
        cur_iter = 1
        thoughts = []
        responses = []
        feedbacks = []
        try:
            question = item.get('question') or item.get('problem', '')
            if not question:
                return [], [], []

            if task == "programming":
                question += f"Your code should pass these tests: {item['test_code']}"

            # first attempt
            cur_response = generate_action(
                question=question,
                model=actor_model,
                tokenizer=actor_tokenizer,
                strategy="simple",
                task=task
            )

            responses.append(cur_response)
            
            judge, feedback = get_feedback(
                response=cur_response,
                doc=item,
                task=task
            )
            is_solved = False
            feedbacks.append(feedback)
            
            cur_feedbacks = (judge, feedback)

            # use self-thinking to iteratively improve
            while not is_solved and cur_iter <= max_iters:
                # get thinking
                thought = generate_self_thinking(
                    question=question,
                    feedbacks=cur_feedbacks,
                    model=thinker_model,
                    tokenizer=thinker_tokenizer,
                    task=task
                )

                thoughts += [thought]

                # apply self-thinking in the next attempt
                cur_response = generate_action(
                    question=question,
                    self_thinking=thought,
                    model=actor_model,
                    tokenizer=actor_tokenizer,
                    strategy="thinking",
                    task=task
                )
                responses.append(cur_response)

                judge, feedback = get_feedback(
                    response=cur_response,
                    doc=item,
                    task=task
                )
                is_solved = judge
                feedbacks.append(feedback)
                cur_feedbacks = (judge, feedback)

                cur_iter += 1

        except Exception as e:
            print(f"Case {idx}:  {e}")
            continue
        
        # store successful QTA (Question-Thinking-Answer) triplets that resulted in correct answers
        if is_solved:
            if task == 'math_reasoning':
                instruction = MATH_SIMPLE_ACTION_INSTRUCTION
            elif task == 'programming':
                instruction = PY_SIMPLE_ACTION_INSTRUCTION
            elif task == 'olympiad':
                instruction = OLYMPIAD_SIMPLE_ACTION_INSTRUCTION

            save_dict = {
                'thinking': thoughts[-1],
                'question': question,
                'answer': responses[-1],
                'instruction': instruction
            }

            f_o.write(json.dumps(save_dict, ensure_ascii=False) + "\n")
            f_o.flush()

    f_o.close()


def get_feedback(response, doc, task):
    if task == 'math_reasoning':
        return math_reasoning_get_feedback(response, doc['answer'])
    elif task == 'programming':
        from generators.generator_utils import parse_code_block
        func = parse_code_block(response)
        return programming_get_feedback(func, doc["test_list"])
    elif task == 'olympiad':
        return olympiad_get_feedback(response, doc['answer'])
    else:
        raise NotImplementedError()