import io
import sys
import contextlib

def execute_code_with_timeout(code_string, timeout=None):

    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        exec_globals = {
            '__builtins__': __builtins__,
            'typing': __import__('typing'),
            'collections': __import__('collections'), 
            'math': __import__('math'),
        }
        
        exec(code_string, exec_globals, {})
        
        output = captured_output.getvalue()
        return output
        
    except Exception as e:
        raise Exception(f"Code execution failed: {str(e)}")
    
    finally:
        sys.stdout = old_stdout