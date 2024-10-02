import streamlit as st
import random
from streamlit_sortables import sort_items

def algorithm():
    st.header("Algorithm")
    st.write("Now that we know how the *k-NN* algorithm works, let's define the algorithm. To do so, " +
             "place the following elements in the correct order:")
    
    algorithm_elements = [
        "Select the k instances with the smallest distances",
        "Calculate the distance between the new instance and all instances in the training set",
        "For each new instance in the test set:",
        "Return the classes assigned to the new instances",
        "Assign the new instance to the class that appears most frequently among the k instances",
        "Sort the distances in ascending order",
        
    ]

    sorted_items = sort_items(algorithm_elements)

    if (
        sorted_items[0] == "For each new instance in the test set:" and
        sorted_items[1] == "Calculate the distance between the new instance and all instances in the training set" and
        sorted_items[2] == "Sort the distances in ascending order" and
        sorted_items[3] == "Select the k instances with the smallest distances" and
        sorted_items[4] == "Assign the new instance to the class that appears most frequently among the k instances" and
        sorted_items[5] == "Return the classes assigned to the new instances"
        ):
        
        st.success("Congratulations! You have successfully defined the *k*-NN algorithm. As you have" +
                   " indicated, the *k*-NN algorithm is as follows:")
        
        st.write("1. For each new instance in the test set:")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;i. Calculate the distance between the new instance and all instances in the training set")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ii. Sort the distances in ascending order")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;iii. Select the *k* instances with the smallest distances")
        st.write("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;iv. Assign the new instance to the class that appears most frequently among the *k* instances")                            
        st.write("2. Return the classes assigned to the new instances")

        st.info("Now that we have defined the *k*-NN algorithm, let's see how we can implement it in Python." +
                " To do so, go to the *Step-by-step implementation* tab.")

        