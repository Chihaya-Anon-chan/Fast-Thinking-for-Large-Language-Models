import re
import math
import json
import os
from typing import Union, List, Dict, Any
from pathlib import Path

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def last_boxed_only_string(string):
    """
    Extract the last boxed answer from a string (official MATH dataset implementation).
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    """
    Remove the \boxed{} wrapper from an answer.
    """
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def is_equiv(str1, str2, verbose=False):
    """
    Check if two mathematical expressions are equivalent using the MATH dataset's equivalence checker.
    """
    if str1 is None and str2 is None:
        if verbose:
            print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(f"Stripped strings: '{ss1}' vs '{ss2}'")
        return ss1 == ss2
    except Exception as e:
        if verbose:
            print(f"Error in equivalence check: {e}")
        return str1 == str2

def load_bmath_dataset(data_path):
    """
    Load MATH dataset from various formats (directory, JSON, or JSONL).
    
    Args:
        data_path: Path to the data (directory, JSON file, or JSONL file)
        
    Returns:
        List of dictionaries in the format compatible with the evaluation system
    """
    dataset = []
    path_obj = Path(data_path)
    
    if not path_obj.exists():
        print(f"Path not found: {data_path}")
        return dataset
    
    # Case 1: JSONL file (merged format)
    if path_obj.is_file() and path_obj.suffix == '.jsonl':
        print(f"Loading from JSONL file: {data_path}")
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Convert to our format
                        item = {
                            'problem': data.get('problem', data.get('question', '')),
                            'solution': data.get('solution', ''),
                            'answer': data.get('answer', remove_boxed(last_boxed_only_string(data.get('solution', '')))),
                            'level': data.get('level', ''),
                            'type': data.get('type', data.get('category', 'unknown')),
                            'category': data.get('category', data.get('type', 'unknown')),
                            'id': data.get('id', f'item_{line_num}'),
                            'original_file': Path(data_path).name
                        }
                        
                        dataset.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
                        
            print(f"Loaded {len(dataset)} items from JSONL file")
        except Exception as e:
            print(f"Error loading JSONL file: {e}")
        return dataset
    
    # Case 2: JSON file (single merged file)
    if path_obj.is_file() and path_obj.suffix == '.json':
        print(f"Loading from JSON file: {data_path}")
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    # Array of items
                    for i, item_data in enumerate(data):
                        item = {
                            'problem': item_data.get('problem', item_data.get('question', '')),
                            'solution': item_data.get('solution', ''),
                            'answer': item_data.get('answer', remove_boxed(last_boxed_only_string(item_data.get('solution', '')))),
                            'level': item_data.get('level', ''),
                            'type': item_data.get('type', item_data.get('category', 'unknown')),
                            'category': item_data.get('category', item_data.get('type', 'unknown')),
                            'id': item_data.get('id', f'item_{i}'),
                            'original_file': Path(data_path).name
                        }
                        dataset.append(item)
                else:
                    # Single item
                    item = {
                        'problem': data.get('problem', data.get('question', '')),
                        'solution': data.get('solution', ''),
                        'answer': data.get('answer', remove_boxed(last_boxed_only_string(data.get('solution', '')))),
                        'level': data.get('level', ''),
                        'type': data.get('type', data.get('category', 'unknown')),
                        'category': data.get('category', data.get('type', 'unknown')),
                        'id': data.get('id', 'item_0'),
                        'original_file': Path(data_path).name
                    }
                    dataset.append(item)
                    
            print(f"Loaded {len(dataset)} items from JSON file")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
        return dataset
    
    # Case 3: Directory structure (original MATH format)
    if path_obj.is_dir():
        print(f"Loading from directory: {data_path}")
        
        # Determine if we're loading train or test
        if path_obj.name in ['train', 'test']:
            base_path = path_obj.parent
            split = path_obj.name
        else:
            base_path = path_obj
            # Try to determine split by checking what exists
            if (path_obj / 'train').exists():
                split = 'train'
            elif (path_obj / 'test').exists():
                split = 'test'
            else:
                # Assume the current directory is the split directory
                split = path_obj.name
                base_path = path_obj.parent
        
        split_path = base_path / split if (base_path / split).exists() else path_obj
        
        if not split_path.exists():
            print(f"Split directory {split_path} does not exist")
            return dataset
        
        # Load all categories
        categories = [d for d in split_path.iterdir() if d.is_dir()]
        
        if not categories:
            # No subdirectories, look for JSON files directly
            json_files = list(split_path.glob('*.json'))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Convert to our format
                    item = {
                        'problem': data.get('problem', ''),
                        'solution': data.get('solution', ''),
                        'answer': data.get('answer', remove_boxed(last_boxed_only_string(data.get('solution', '')))),
                        'level': data.get('level', ''),
                        'type': data.get('type', 'unknown'),
                        'category': 'unknown',
                        'id': json_file.stem,
                        'original_file': str(json_file)
                    }
                    
                    dataset.append(item)
                    
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
                    continue
        else:
            # Original format: directory with category subdirectories
            for category in categories:
                category_name = category.name
                json_files = list(category.glob('*.json'))
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Convert to our format
                        item = {
                            'problem': data.get('problem', ''),
                            'solution': data.get('solution', ''),
                            'answer': data.get('answer', remove_boxed(last_boxed_only_string(data.get('solution', '')))),
                            'level': data.get('level', ''),
                            'type': data.get('type', category_name),
                            'category': category_name,
                            'id': json_file.stem,
                            'original_file': str(json_file)
                        }
                        
                        dataset.append(item)
                        
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
                        continue
        
        print(f"Loaded {len(dataset)} items from directory")
    
    return dataset

def get_feedback(predict, answer) -> Union[bool, str]:
    """
    Get feedback for a prediction.
    """
    if is_correct(predict, answer):
        return True, f"Student answer: {predict}\n"
    
    feedback_str = f"Student answer: {predict}\nIt's incorrect.\n"
    return False, feedback_str

def is_correct(completion, answer):
    """
    Check if the completion is correct using official MATH dataset evaluation.
    """
    if answer is None:
        return False
    
    # Extract predicted answer from completion using official method
    predicted_boxed = last_boxed_only_string(completion)
    if predicted_boxed is None:
        return False
    
    predicted = remove_boxed(predicted_boxed)
    if predicted is None:
        return False
    
    # Use official equivalence checker
    return is_equiv(predicted, answer, verbose=False)