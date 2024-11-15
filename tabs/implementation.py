import streamlit as st
import numpy as np
import pandas as pd
from scipy.spatial import distance
from streamlit_ace import st_ace
from collections import Counter

from utils.coding import class_prediction, euclidean_distance, k_instances, sort_distances
from utils.time_cpu_manager import exec_with_timeout

def implementation():
    st.header("Implementation")

    time_out = 5

    training_data = [
            (72, 55, 'Apple'),
            (75, 60, 'Apple'),
            (73, 59, 'Apple'),
            (150, 67, 'Banana'),
            (155, 73, 'Banana'),
            (160, 70, 'Banana'),
    ]

    new_point = (74, 58)

    distances = []
        
    def truncate_decimal(value, decimal_places):
        factor = 10 ** decimal_places
        return int(value * factor) / factor

    for i in range(len(training_data)):
        dist = distance.euclidean(training_data[i][:-1], new_point)
        distances.append(truncate_decimal(dist, 2))

    sort_distances_index = np.argsort(distances)

    k = 3

    training_k_instances = [training_data[i] for i in sort_distances_index[:k]]

    predicted_class = Counter([instance[-1] for instance in training_k_instances]).most_common(1)[0][0]

    st.write('Now that we know how the *k*-NN algorithm works, let\'s implement it in Python. To do so, we will implement each of the steps in the algorithm.')

    st.info("""
        **IMPORTANT INFORMATION**
               
        To make sure the code can be executed, the following restrictions are applied:
        - **DO NOT use any external libraries**, such as numpy, pandas and matplotlib.
        - Only the following build-in functions are allowed: **range**, **len**, **sorted**, **enumerate**.
        - **NO dataset needs to be created**. The training data and new point are already defined. Please use the variables **training_data** and **new_point** instead when needed.
        - The result of the function should be stored in a variable called **result**
        """)
    
    st.write("**Format of the data**")

    st.write("*Training data:*")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.write("The training data is a list of tuples, where each tuple contains the following information:")
        st.write("- The first element is the weight of the fruit in grams.")
        st.write("- The second element is the size of the fruit in centimeters.")
        st.write("- The third element is the class of the fruit.")

    with col2:
        st.write("See an example of training data below:")

        data = {
            'Weight [grams]': [72, 75, 73, 150, 155, 160],
            'Diameter [millimeters]': [55, 60, 59, 67, 73, 70],
            'Class': ['Apple', 'Apple', 'Apple', 'Banana', 'Banana', 'Banana']
        }

        df_data = pd.DataFrame(data)
        st.dataframe(df_data)

    st.write("*Training data:*")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.write("The new point is a tuple containing the following information:")
        st.write("- The first element is the weight of the fruit in grams.")
        st.write("- The second element is the size of the fruit in centimeters.")

    with col2:    
        st.write("See an example of a new below:")

        test_data = {
                'Weight [grams]': [74],
                'Diameter [millimeters]': [58],
            }

        df_test_data = pd.DataFrame(test_data)
        st.dataframe(df_test_data)

    st.write("")
    st.write("**Step 1: Calculate the distance between the new instance and all instances in the training set**")

    # Title of the app
    st.write('In this first step, create a script to calculate the Euclidean distance between the new instance and all instances in the training set. As seen in the Introduction tab, the Euclidean distance is calculated as follows:')

    st.latex(r'd(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}')

    st.write("Remember that no external libraries are allowed, and the result should be stored in a variable called **result**. The training data and new point are already defined. Please use the variables **training_data** and **new_point**.")
    

    # Create a text area where the user can input code
    user_code = st_ace(language="python", theme="chrome", height=300, key="code_1")
    #user_code = st.text_area("Enter your Python code here:", height=300, key="calculate_distances")

    
    # Initialize session state for Step 1 completion
    if "step_1_completed" not in st.session_state:
        st.session_state.step_1_completed = False

    # Initialize session state for Step 2 completion
    if "step_2_completed" not in st.session_state:
        st.session_state.step_2_completed = False

    # Initialize session state for Step 2 completion
    if "step_3_completed" not in st.session_state:
        st.session_state.step_3_completed = False

    # Initialize session state for Step 2 completion
    if "step_4_completed" not in st.session_state:
        st.session_state.step_4_completed = False

    # Button to execute the code
    if st.button("Run Code", key="calculate_distances_button"):
        #result = euclidean_distance(user_code, training_data, new_point)
        result = exec_with_timeout(time_out, 'euclidean_distance', user_code, training_data, new_point)

        # Display the result or a message if no result was found
        if result is not None:
            try:
                truncated_results = [truncate_decimal(r, 2) for r in result]
            except:
                st.error("The result variable should be a list of distances.")
                st.session_state.step_1_completed = False
                st.session_state.step_2_completed = False
                st.session_state.step_3_completed = False
                st.session_state.step_4_completed = False

            try:
                if truncated_results == distances:
                    st.session_state.step_1_completed = True
            except:
                pass
                
            else:
                st.session_state.step_1_completed = False
                st.session_state.step_2_completed = False
                st.session_state.step_3_completed = False
                st.session_state.step_4_completed = False

                st.error("The distances have **NOT** been calculated correctly. Please check your code and try again.")
        else:
            st.session_state.step_1_completed = False
            st.session_state.step_2_completed = False
            st.session_state.step_3_completed = False
            st.session_state.step_4_completed = False
            st.error("No result variable found.")

    if st.session_state.step_1_completed:    
        st.success("**Congratulations!** The distances have been calculated correctly. Now you can move on to Step 2!")

        st.write("**Step 2: Sort the distances in ascending order**")

        st.write("In this step, create a script to sort the distances calculated in the previous step in ascending order. Remember that no external libraries are allowed, and the result should be stored in a variable called **result**.")

        st.info("""
                **IMPORTANT INFORMATION**
                            
                - The output **should NOT** be a list of distances in an ascending order but the indices of the distances in the original list. For example, if the original list of distances is [3.2, 1.5, 2.7], the output should be [1, 2, 0].
                - You do not have to calculate the distances again. Use the variable **distance** instead.
                """)
                    
        # Create a text area where the user can input code
        #user_code_sort = st.text_area("Enter your Python code here:", height=300, key="sort_distances")
        user_code_sort = st_ace(language="python", theme="chrome", height=300, key="code_2")

        # Button to execute the code
        if st.button("Run Code", key="calculate_sort_button"):
            #result = sort_distances(user_code_sort, distances)
            result = exec_with_timeout(time_out, 'sort_distances', user_code_sort, distances)

            # Display the result or a message if no result was found
            if result is not None:
                try:
                    if list(result) == list(sort_distances_index):
                        st.session_state.step_2_completed = True
                        
                    else:
                        st.session_state.step_2_completed = False
                        st.session_state.step_3_completed = False
                        st.session_state.step_4_completed = False
                        st.error("Wrong result! Please check your code and try again.")
                except:
                    st.session_state.step_2_completed = False
                    st.session_state.step_3_completed = False
                    st.session_state.step_4_completed = False
                    st.error("The result variable should be a list of indices.")
            else:
                st.session_state.step_2_completed = False
                st.session_state.step_3_completed = False
                st.session_state.step_4_completed = False
                st.error("No result variable found.")

    if st.session_state.step_2_completed:
        # Initialize session state for Step 2 completion
        if "step_3_completed" not in st.session_state:
            st.session_state.step_3_completed = False
        
        st.success("**Congratulations!** The distances have been sorted correctly. Now you can move on to Step 3!")

        st.write("**Step 3: Select the k instances with the smallest distances**")

        st.write("In this step, create a script to select the **k** instances with the smallest distances. Remember that no external libraries are allowed, and the result should be stored in a variable called **result**.")

        st.info("""
                **IMPORTANT INFORMATION**
                            
                - The result should be a subset of the training data based on *k* value.
                - The following variables are availabe: **training_data**, **sorted_indices** and **k**.
                """)
        
        user_code_k_instances = st_ace(language="python", theme="chrome", height=300, key="code_3")

        # Button to execute the code
        if st.button("Run Code", key="k_instances_button"):
            #result = k_instances(user_code_k_instances, training_data, sort_distances_index, k)
            result = exec_with_timeout(time_out, 'k_instances', user_code_k_instances, training_data, sort_distances_index, k)

            # Display the result or a message if no result was found
            if result is not None:
                try:
                    if training_k_instances == result:
                        st.session_state.step_3_completed = True
                        
                    else:
                        st.session_state.step_3_completed = False
                        st.session_state.step_4_completed = False
                        st.error("The selected *k* instances are not correct. Please check your code and try again.")
                except:
                    st.session_state.step_3_completed = False
                    st.session_state.step_4_completed = False
                    st.error("The result variable should be a list of indices.")
            else:
                st.session_state.step_3_completed = False
                st.session_state.step_4_completed = False
                st.error("No result variable found.")

    if st.session_state.step_3_completed:
        # Initialize session state for Step 4 completion
        if "step_4_completed" not in st.session_state:
            st.session_state.step_4_completed = False
        
        st.success("**Congratulations!** The instances have been selected correctly. Now you can move on to Step 4!")

        st.write("**Step 4: Assign the new instance to the class that appears most frequently among the k instances**")

        st.write("In this step, create a script to assign the new instance to the class that appears most frequently among the k instances. Remember that no external libraries are allowed, and the result should be stored in a variable called **result**.")

        st.info("""
                **IMPORTANT INFORMATION**
                            
                - Only the k_instances variable is available in this step.
                - The result should be the class that appears most frequently among the *k* instances.""")
        
        user_class_prediction = st_ace(language="python", theme="chrome", height=300, key="code_4")

        # Button to execute the code
        if st.button("Run Code", key="class_prediction"):
            #result = class_prediction(user_class_prediction, training_k_instances)
            result = exec_with_timeout(time_out, 'class_prediction', user_class_prediction, training_k_instances)

            # Display the result or a message if no result was found
            if result is not None:
                try:
                    if predicted_class == result:
                        st.session_state.step_4_completed = True
                        
                    else:
                        st.session_state.step_4_completed = False
                        st.error("The predicted class is not correct. Please check your code and try again.")
                except:
                    st.session_state.step_4_completed = False
                    st.error("The result variable should be a list of indices.")
            else:
                st.session_state.step_4_completed = False
                st.error("No result variable found.")


    if st.session_state.step_4_completed:
        st.success("Amazing work! You have successfully implemented the main steps of the KNN algorithm!")
        
        