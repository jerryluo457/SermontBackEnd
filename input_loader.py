import json
import ast

def load_simulation_inputs(file_path='input.json'):
    """
    Parses a JSON where values are Python-formatted strings
    and returns a dictionary of actual Python objects.
    """
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    
    processed_inputs = {}
    
    for key, value in raw_data.items():
        try:
            # ast.literal_eval converts "[(a, 0.1)]" into a real list of tuples
            # Note: highGrowthStock must be a string 'highGrowthStock' in the JSON string
            processed_inputs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback for simple strings or malformed inputs
            processed_inputs[key] = value
            
    return processed_inputs

# Usage for main.py integration
if __name__ == "__main__":
    inputs = load_simulation_inputs()
    # Print out to verify
    for k, v in inputs.items():
        print(f"Loaded {k}: {v} (Type: {type(v)})")