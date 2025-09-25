#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error type analyzer
Identifies specific error types based on feedback information, providing targeted guidance for CoT generation
"""

import re
from typing import Dict, List, Tuple, Optional

class ErrorAnalyzer:
    """Error type analyzer"""
    
    def __init__(self):
        # Math error type patterns
        self.math_error_patterns = {
            'calculation': [
                r'wrong.*calculation', r'arithmetic.*error', r'computational.*mistake',
                r'incorrect.*result', r'wrong.*answer', r'calculation.*error'
            ],
            'conceptual': [
                r'misunderstand.*concept', r'wrong.*approach', r'incorrect.*method',
                r'conceptual.*error', r'fundamental.*mistake', r'wrong.*interpretation'
            ],
            'unit_conversion': [
                r'unit.*error', r'conversion.*mistake', r'dimensional.*error',
                r'wrong.*unit', r'measurement.*error', r'scale.*mistake'
            ],
            'logic': [
                r'logical.*error', r'reasoning.*mistake', r'wrong.*logic',
                r'invalid.*assumption', r'flawed.*reasoning', r'logic.*flaw'
            ],
            'boundary': [
                r'edge.*case', r'boundary.*error', r'constraint.*violation',
                r'limit.*exceeded', r'range.*error', r'boundary.*condition'
            ]
        }

        # Programming error type patterns
        self.programming_error_patterns = {
            'logic': [
                r'logic.*error', r'wrong.*algorithm', r'incorrect.*logic',
                r'algorithmic.*mistake', r'control.*flow.*error', r'condition.*error'
            ],
            'boundary': [
                r'index.*out.*of.*bounds', r'array.*index.*error', r'boundary.*error',
                r'off.*by.*one', r'edge.*case.*failed', r'bounds.*check.*failed'
            ],
            'data_structure': [
                r'data.*structure.*error', r'wrong.*data.*type', r'structure.*mistake',
                r'access.*error', r'manipulation.*error', r'data.*handling.*error'
            ],
            'complexity': [
                r'time.*limit.*exceeded', r'timeout', r'performance.*issue',
                r'efficiency.*problem', r'complexity.*error', r'optimization.*needed'
            ],
            'type_null': [
                r'null.*pointer', r'type.*error', r'none.*type.*error',
                r'null.*reference', r'type.*mismatch', r'validation.*failed'
            ]
        }
    
    def analyze_error_type(self, task: str, feedback: str, student_answer: str = "", expected_answer: str = "") -> str:
        """
        Analyze error type

        Args:
            task: Task type ('math', 'programming')
            feedback: Feedback information
            student_answer: Student answer
            expected_answer: Expected answer

        Returns:
            Error type string
        """
        if task == 'math':
            return self._analyze_math_error(feedback, student_answer, expected_answer)
        elif task == 'programming':
            return self._analyze_programming_error(feedback, student_answer, expected_answer)
        else:
            return 'unknown'
    
    def _analyze_math_error(self, feedback: str, student_answer: str, expected_answer: str) -> str:
        """Analyze math error type"""
        feedback_lower = feedback.lower()
        
        # Check various error patterns
        for error_type, patterns in self.math_error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, feedback_lower):
                    return error_type
        
        # Heuristic judgment based on answer differences
        if student_answer and expected_answer:
            # Check if it's a unit error
            if self._has_unit_difference(student_answer, expected_answer):
                return 'unit_conversion'
            
            # Check if it's a calculation error (numbers close but different)
            if self._has_calculation_difference(student_answer, expected_answer):
                return 'calculation'
        
        # Default return conceptual error
        return 'conceptual'
    
    def _analyze_programming_error(self, feedback: str, student_answer: str, expected_answer: str) -> str:
        """Analyze programming error type"""
        feedback_lower = feedback.lower()
        
        # Check various error patterns
        for error_type, patterns in self.programming_error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, feedback_lower):
                    return error_type
        
        # Judge based on test failure information
        if 'test' in feedback_lower:
            if 'timeout' in feedback_lower or 'time limit' in feedback_lower:
                return 'complexity'
            elif 'index' in feedback_lower or 'bounds' in feedback_lower:
                return 'boundary'
            elif 'null' in feedback_lower or 'none' in feedback_lower:
                return 'type_null'
        
        # Default return logic error
        return 'logic'
    
    def _has_unit_difference(self, student_answer: str, expected_answer: str) -> bool:
        """Check if there are unit differences"""
        # Simple unit check
        units = ['m', 'cm', 'mm', 'km', 'g', 'kg', 'mg', 's', 'min', 'h', 'day']
        student_units = [unit for unit in units if unit in student_answer.lower()]
        expected_units = [unit for unit in units if unit in expected_answer.lower()]
        return student_units != expected_units
    
    def _has_calculation_difference(self, student_answer: str, expected_answer: str) -> bool:
        """Check if it's a calculation error (extract and compare numerical values)"""
        try:
            # Extract numerical values
            student_nums = re.findall(r'-?\d+\.?\d*', student_answer)
            expected_nums = re.findall(r'-?\d+\.?\d*', expected_answer)
            
            if len(student_nums) == len(expected_nums) == 1:
                student_val = float(student_nums[0])
                expected_val = float(expected_nums[0])
                # If values are relatively close but not equal, it might be a calculation error
                if abs(student_val - expected_val) / max(abs(expected_val), 1) < 0.5:
                    return True
        except:
            pass
        return False
    
    def get_error_specific_context(self, task: str, error_type: str, feedback: str) -> str:
        """
        Generate specific context information based on error type

        Args:
            task: Task type
            error_type: Error type
            feedback: Feedback information

        Returns:
            Error-specific context information
        """
        context_templates = {
            'math': {
                'calculation': f"The error appears to be computational. Focus on step-by-step verification. Error details: {feedback}",
                'conceptual': f"The error seems conceptual. Emphasize fundamental understanding. Error details: {feedback}",
                'unit_conversion': f"The error involves units or conversions. Stress dimensional analysis. Error details: {feedback}",
                'logic': f"The error is in reasoning logic. Focus on assumption validation. Error details: {feedback}",
                'boundary': f"The error involves boundary conditions. Emphasize constraint checking. Error details: {feedback}"
            },
            'programming': {
                'logic': f"The error is algorithmic/logical. Focus on control flow validation. Error details: {feedback}",
                'boundary': f"The error involves boundary conditions. Emphasize bounds checking. Error details: {feedback}",
                'data_structure': f"The error is in data handling. Focus on proper data manipulation. Error details: {feedback}",
                'complexity': f"The error is performance-related. Emphasize efficiency considerations. Error details: {feedback}",
                'type_null': f"The error involves type/null issues. Focus on input validation. Error details: {feedback}"
            },
        }
        
        return context_templates.get(task, {}).get(error_type, f"Error analysis: {feedback}")

# Global error analyzer instance
error_analyzer = ErrorAnalyzer() 