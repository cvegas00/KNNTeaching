import streamlit as st
import psutil
import os
import concurrent.futures
import time

# Safe built-in functions and restricted imports
safe_builtins = {
    'range': range,
    'len': len,
    'sorted': sorted,
    'enumerate': enumerate,
}

# Function to execute user code with timeout and CPU usage check
def euclidean_distance(user_code, training_data, new_point, shared_result):
    # Prepare the environment for safe execution
    safe_locals = {}
    safe_globals = {
        '__builtins__': safe_builtins,  # Explicitly restrict built-ins
        'training_data': training_data,
        'new_point': new_point,
    }

    try:
        # Execute the user code using exec
        exec(user_code, safe_globals, safe_locals)

        # Retrieve the result from the executed code
        result = safe_locals.get('result', None)
        shared_result['result'] = result  # Store the result in the shared dictionary
    except Exception as e:
        shared_result['result'] = f"Error: {str(e)}"  # Store any error that occurs
        st.error(f"Error: {e}")


def sort_distances(user_code, distances, shared_result):
    try:
        # Initialize locals dictionary for execution environment
        safe_locals = {}

        # Prepare a strict environment with only allowed built-ins and no other imports
        safe_globals = {
            '__builtins__': safe_builtins,  # Explicitly restrict built-ins
            'distances': distances,
        }

        # Execute user code in the controlled environment
        exec(user_code, safe_globals, safe_locals)

        # Retrieve specific values or variables from the executed code
        result = safe_locals.get('result', None)  # Looking for a 'result' variable
        shared_result['result'] = result  # Store the result in the shared dictionary
    except Exception as e:
        shared_result['result'] = f"Error: {str(e)}"  # Store any error that occurs
        st.error(f"Error: {e}")

def k_instances(user_code, training_data, sort_distances_index, k, shared_result):
    try:
        # Initialize locals dictionary for execution environment
        safe_locals = {}

        # Prepare a strict environment with only allowed built-ins and no other imports
        safe_globals = {
            '__builtins__': safe_builtins,  # Explicitly restrict built-ins
            'training_data': training_data,
            'sorted_indices': sort_distances_index,
            'k': k,
        }

        # Execute user code in the controlled environment
        exec(user_code, safe_globals, safe_locals)

        # Retrieve specific values or variables from the executed code
        result = safe_locals.get('result', None)  # Looking for a 'result' variable
        shared_result['result'] = result  # Store the result in the shared dictionary
    except Exception as e:
        shared_result['result'] = f"Error: {str(e)}"  # Store any error that occurs
        st.error(f"Error: {e}")

def class_prediction(user_code, training_k_instances, shared_result):
    try:
        # Initialize locals dictionary for execution environment
        safe_locals = {}

        # Prepare a strict environment with only allowed built-ins and no other imports
        safe_globals = {
            '__builtins__': safe_builtins,  # Explicitly restrict built-ins
            'k_instances': training_k_instances,
        }

        # Execute user code in the controlled environment
        exec(user_code, safe_globals, safe_locals)

        # Retrieve specific values or variables from the executed code
        result = safe_locals.get('result', None)  # Looking for a 'result' variable

        shared_result['result'] = result  # Store the result in the shared dictionary
    except Exception as e:
        shared_result['result'] = f"Error: {str(e)}"  # Store any error that occurs
        st.error(f"Error: {e}")
