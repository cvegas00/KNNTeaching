import multiprocessing
import streamlit as st
from utils.coding import class_prediction, euclidean_distance, k_instances, sort_distances

def exec_with_timeout(function, *args, timeout=5):
    # Create a process to run the exec code with parameters
    if function == 'euclidean_distance':
        process = multiprocessing.Process(target=euclidean_distance, args=args)
    elif function == 'sort_distances':
        process = multiprocessing.Process(target=sort_distances, args=args)
    elif function == 'k_instances':
        process = multiprocessing.Process(target=k_instances, args=args)
    elif function == 'class_prediction':
        process = multiprocessing.Process(target=class_prediction, args=args)

    process.start()
    
    # Wait for the process to finish within the timeout
    process.join(timeout)
    
    # If the process is still alive, terminate it and report timeout
    if process.is_alive():
        st.error(f"Execution timed out after {timeout} seconds")
        process.terminate()
        process.join()  # Ensure the process is cleaned up


def exec_with_timeout(timeout, function, *args):
    # Create a manager to handle shared data
    with multiprocessing.Manager() as manager:
        # Create a shared dictionary to store the result
        shared_result = manager.dict()
        args = args + (shared_result,)  # Add the shared dictionary to the arguments

        # Create a process to run the exec code with parameters
        if function == 'euclidean_distance':
            process = multiprocessing.Process(target=euclidean_distance, args=args)
        elif function == 'sort_distances':
            process = multiprocessing.Process(target=sort_distances, args=args)
        elif function == 'k_instances':
            process = multiprocessing.Process(target=k_instances, args=args)
        elif function == 'class_prediction':
            process = multiprocessing.Process(target=class_prediction, args=args)
        
        process.start()
        
        # Wait for the process to finish within the timeout
        process.join(timeout)
        
        # If the process is still alive, terminate it and report timeout
        if process.is_alive():
            st.error('Execution Time Limit Exceeded')
            process.terminate()
            process.join()  # Ensure the process is cleaned up
            return None  # No result if it times out
        else:
            # Get the result from the shared dictionary
            return shared_result.get('result')
