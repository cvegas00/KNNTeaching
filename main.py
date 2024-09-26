import streamlit as st

from tabs.introduction import introduction
from tabs.algorithm import algorithm
from tabs.implementation import implementation
from tabs.practise import practise

# Set the title of the app
st.title("How does *k*-Nearest Neighbors algorithm (*k*-NN) work?")

# Add some content (optional)
st.write("The main aim of this page is to provide to the student a more interactive way to learn about the *k*-NN algorithm.")

tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Algorithm", "Step-by-step Implementation", "Practise!"])

with tab1:
    introduction()

with tab2:
    algorithm()

with tab3:
    implementation()

with tab4:
    practise()