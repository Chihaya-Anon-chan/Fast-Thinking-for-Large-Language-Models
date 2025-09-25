import re
import json
import sympy as sp
from sympy import simplify, Eq, sympify, Pow
from sympy.parsing.latex import parse_latex
import math
from typing import Union, List, Dict, Any
import sys
import os

# Add the OlympiadBench auto scoring judge to the path
current_dir = os.path.dirname(__file__)
olympiad_bench_path = os.path.join(current_dir, 'OlympiadBench-main', 'eval')
if olympiad_bench_path not in sys.path:
    sys.path.append(olympiad_bench_path)

try:
    from auto_scoring_judge import AutoScoringJudge
except ImportError:
    print("Warning: Could not import AutoScoringJudge from OlympiadBench. Using fallback implementation.")
    AutoScoringJudge = None
except Exception as e:
    print(f"Warning: Error initializing AutoScoringJudge: {e}. Using fallback implementation.")
    AutoScoringJudge = None


class OlympiadEvaluator:
    """
    Olympiad problem evaluator that uses the official OlympiadBench scoring logic
    """
    def __init__(self):
        # Try to initialize AutoScoringJudge with error handling
        self.scorer = None
        if AutoScoringJudge:
            try:
                self.scorer = AutoScoringJudge()
            except Exception as e:
                print(f"Warning: Failed to initialize AutoScoringJudge: {e}")
                print("Using fallback evaluation methods.")
                self.scorer = None
        
        self.precision = 1e-8  # Default precision
        
        # Answer type mapping for different problem types
        self.answer_type_patterns = {
            'Numerical': [r'([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)'],
            'Expression': [r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\\sqrt\{([^}]+)\}', r'([a-zA-Z_]\w*)', r'\\pi', r'\\theta'],
            'Equation': [r'.*=.*'],
            'Interval': [r'[\[\(].*,.*[\]\)]', r'.*\\cup.*'],
            'Tuple': [r'\([^)]*,[^)]*\)', r'[\[\(].*,.*[\]\)]'],
        }

    def extract_boxed_answer(self, text):
        """
        Extract answer using official OlympiadBench extract_boxed_content logic
        """
        if not text:
            return None
            
        # Use exact official OlympiadBench logic
        boxed_matches = re.finditer(r'\\boxed\{', text)
        results = ""
        
        for match in boxed_matches:
            start_index = match.end()
            end_index = start_index
            stack = 1
            
            while stack > 0 and end_index < len(text):
                if text[end_index] == '{':
                    stack += 1
                elif text[end_index] == '}':
                    stack -= 1
                end_index += 1
            
            if stack == 0:
                content = text[start_index:end_index - 1]
                results += content + ","
            else:
                raise ValueError("Mismatched braces in LaTeX string.")
        
        if results == "":
            # Fallback to $...$ pattern in last line (official logic)
            last_line_ans = text.strip().split("\n")[-1]
            dollar_pattern = r"\$(.*?)\$"
            answers = re.findall(dollar_pattern, last_line_ans)
            
            if answers:
                for ans in answers:
                    results += ans + ","
            else:
                results = text
        
        return results.rstrip(',') if results.endswith(',') else results

    def preprocess_answer(self, answer_text):
        """
        Preprocess answer text to clean format for comparison
        """
        if not answer_text:
            return ""
            
        # Remove dollar signs and extra whitespace
        answer_text = re.sub(r'\$', '', answer_text)
        answer_text = answer_text.strip()
        
        # Remove common prefixes and suffixes
        answer_text = re.sub(r'^(Answer:\s*|The answer is\s*)', '', answer_text, flags=re.IGNORECASE)
        answer_text = re.sub(r'[\.\u3002]+$', '', answer_text)
        
        return answer_text.strip()

    def is_correct_with_official_scorer(self, prediction, ground_truth, precision=None):
        """
        Use the official OlympiadBench auto scoring judge for evaluation
        Note: Official judge expects (ground_truth, prediction) order
        """
        if not self.scorer:
            return False
            
        if precision is None:
            precision = self.precision
            
        try:
            # Official scorer: judge(expression1=ground_truth, expression2=prediction)
            return self.scorer.judge(ground_truth, prediction, precision)
        except Exception as e:
            # If official scorer fails, try fallback for some cases
            return False

    def fallback_comparison(self, prediction, ground_truth):
        """
        Enhanced fallback comparison when official scorer is not available
        """
        # Simple string comparison after normalization
        pred_clean = self.preprocess_answer(prediction)
        gt_clean = self.preprocess_answer(ground_truth)
        
        if pred_clean == gt_clean:
            return True
            
        # Remove common LaTeX formatting and try again
        pred_stripped = re.sub(r'[\$\\{}]', '', pred_clean)
        gt_stripped = re.sub(r'[\$\\{}]', '', gt_clean)
        
        if pred_stripped == gt_stripped:
            return True
            
        # Try numerical comparison for simple numbers
        try:
            pred_nums = re.findall(r'[-+]?\d*\.?\d+', pred_stripped)
            gt_nums = re.findall(r'[-+]?\d*\.?\d+', gt_stripped)
            
            if pred_nums and gt_nums and len(pred_nums) == len(gt_nums):
                for p_num, g_num in zip(pred_nums, gt_nums):
                    if abs(float(p_num) - float(g_num)) > self.precision:
                        return False
                return True
        except (ValueError, TypeError):
            pass
            
        # Try fuzzy string matching for similar expressions
        pred_normalized = re.sub(r'\s+', '', pred_stripped.lower())
        gt_normalized = re.sub(r'\s+', '', gt_stripped.lower())
        
        return pred_normalized == gt_normalized

    def evaluate_single_problem(self, prediction, ground_truth, answer_type=None, unit=None, precision=None):
        """
        Evaluate a single olympiad problem using official OlympiadBench logic
        
        Args:
            prediction: Model's predicted answer
            ground_truth: Correct answer  
            answer_type: Type of answer (Numerical, Expression, etc.)
            unit: Expected unit if any
            precision: Precision for numerical comparison
            
        Returns:
            bool: Whether the prediction is correct
        """
        # Skip evaluation for proof problems that need human evaluation
        if answer_type == "Need_human_evaluate":
            # For validation purposes, assume ground truth equals itself
            return prediction.strip() == ground_truth.strip()
            
        # Skip empty ground truth
        if not ground_truth or not ground_truth.strip():
            return False
            
        # Use official scorer directly with raw inputs
        if self.scorer:
            try:
                # Official scorer handles extraction and preprocessing internally
                result = self.scorer.judge(ground_truth, prediction, precision or self.precision)
                return result
            except Exception as e:
                # If official scorer fails completely, try simple string comparison
                return prediction.strip() == ground_truth.strip()
        else:
            # Fallback when scorer not available
            extracted_pred = self.extract_boxed_answer(prediction)
            if extracted_pred is None:
                extracted_pred = prediction
                
            pred_processed = self.preprocess_answer(extracted_pred)
            gt_processed = self.preprocess_answer(ground_truth)
            
            return self.fallback_comparison(pred_processed, gt_processed)


# Global evaluator instance
_evaluator = OlympiadEvaluator()


def get_feedback(predict, answer, **kwargs) -> Union[bool, str]:
    """
    Get feedback for olympiad problem evaluation
    
    Args:
        predict: Model prediction text
        answer: Ground truth answer
        **kwargs: Additional parameters (answer_type, unit, precision, etc.)
        
    Returns:
        Tuple of (is_correct: bool, feedback: str)
    """
    # Extract additional parameters
    answer_type = kwargs.get('answer_type', None)
    unit = kwargs.get('unit', None) 
    precision = kwargs.get('precision', None)
    
    # Evaluate the prediction
    is_correct = _evaluator.evaluate_single_problem(
        predict, answer, answer_type, unit, precision
    )
    
    if is_correct:
        return True, f"Student answer: {predict}\n"
    else:
        feedback_str = f"Student answer: {predict}\nIt's incorrect.\n"
        return False, feedback_str


def is_correct(prediction, answer, **kwargs) -> bool:
    """
    Check if prediction is correct for olympiad problems
    
    Args:
        prediction: Model prediction text
        answer: Ground truth answer  
        **kwargs: Additional parameters (answer_type, unit, precision, etc.)
        
    Returns:
        bool: Whether the prediction is correct
    """
    answer_type = kwargs.get('answer_type', None)
    unit = kwargs.get('unit', None)
    precision = kwargs.get('precision', None)
    
    return _evaluator.evaluate_single_problem(
        prediction, answer, answer_type, unit, precision
    )


def extract_answer(text):
    """
    Extract answer from text using OlympiadBench logic
    
    Args:
        text: Input text containing answer
        
    Returns:
        str: Extracted answer or None
    """
    return _evaluator.extract_boxed_answer(text)


def load_olympiad_dataset(data_path):
    """
    Load olympiad dataset from JSON file
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        List of problem dictionaries
    """
    if not os.path.exists(data_path):
        print(f"Dataset file not found: {data_path}")
        return []
        
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        else:
            return [data]  # Single item
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def evaluate_dataset(predictions, dataset, **kwargs):
    """
    Evaluate predictions on olympiad dataset
    
    Args:
        predictions: List of prediction texts
        dataset: List of problem dictionaries
        **kwargs: Additional evaluation parameters
        
    Returns:
        Dict with evaluation results
    """
    if len(predictions) != len(dataset):
        raise ValueError(f"Prediction count ({len(predictions)}) doesn't match dataset size ({len(dataset)})")
    
    results = []
    correct_count = 0
    
    for i, (pred, problem) in enumerate(zip(predictions, dataset)):
        # Extract ground truth answer
        gt_answer = problem.get('final_answer', [''])[0] if isinstance(problem.get('final_answer'), list) else problem.get('final_answer', '')
        
        # Get problem metadata
        answer_type = problem.get('answer_type', None)
        unit = problem.get('unit', None) 
        is_multiple = problem.get('is_multiple_answer', False)
        
        # Evaluate
        is_correct_result = is_correct(
            pred, gt_answer, 
            answer_type=answer_type, 
            unit=unit,
            **kwargs
        )
        
        if is_correct_result:
            correct_count += 1
            
        results.append({
            'id': problem.get('id', i),
            'prediction': pred,
            'ground_truth': gt_answer,
            'correct': is_correct_result,
            'answer_type': answer_type,
            'subfield': problem.get('subfield', ''),
            'classification': problem.get('classification', '')
        })
    
    accuracy = correct_count / len(results) if results else 0.0
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': len(results),
        'results': results
    }