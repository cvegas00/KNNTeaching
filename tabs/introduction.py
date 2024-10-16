import streamlit as st
import pandas as pd

from streamlit_sortables import sort_items
from scipy.spatial import distance

import matplotlib.pyplot as plt

def introduction():
    
    if 'distance_1' not in st.session_state:
        st.session_state['distance_1'] = 0.0
    
    if 'is_disabled_1' not in st.session_state:
        st.session_state['is_disabled_1'] = False

    if 'distance_2' not in st.session_state:
        st.session_state['distance_2'] = 0.0
    
    if 'is_disabled_2' not in st.session_state:
        st.session_state['is_disabled_2'] = False

    if 'distance_3' not in st.session_state:
        st.session_state['distance_3'] = 0.0
    
    if 'is_disabled_3' not in st.session_state:
        st.session_state['is_disabled_3'] = False

    if 'distance_4' not in st.session_state:
        st.session_state['distance_4'] = 0.0
        
    if 'is_disabled_4' not in st.session_state:
        st.session_state['is_disabled_4'] = False

    if 'distance_5' not in st.session_state:
        st.session_state['distance_5'] = 0.0
        
    if 'is_disabled_5' not in st.session_state:
        st.session_state['is_disabled_5'] = False

    if 'distance_6' not in st.session_state:
        st.session_state['distance_6'] = 0.0
        
    if 'is_disabled_6' not in st.session_state:
        st.session_state['is_disabled_6'] = False

    st.header("Introduction")
    st.write("The *k*-NN is a simple and effective algorithm utilised in machine learning for " +
             "classification and regression tasks. For this case, we want to identify which " +
             " class a new instance belongs to. Thus, are we going to perform classification or " +
             " a regression analysis?")
    
    options = ['', 'Regression Analysis', 'Classification Analysis']
    selected_options = st.selectbox('Select answer:', options)
    
    if selected_options == 'Regression Analysis':
        st.error("Regression analysis is performed when we want to predict a **continuous output** " +
                 "based on the input features. In this case, we want to predict the class of a new instance. ")
    elif selected_options == 'Classification Analysis':
        st.success("Correct! We are going to perform a classification analysis.")

    st.write("Before defining the *k*-NN algorithm, we need to understand what " +
             "classification is. In machine learning, classification is a type of **supervised learning** " +
             "where the main objective is to predict the categorical class labels of new instances based on past " +
             "observations. For instance, based on the following past observations:")
    
    st.image("./imgs/introduction_1.svg", use_column_width=True, caption="Figure 1: Past observations.")

    st.write("Which would be the predicted class of the new instance?")

    st.image("./imgs/introduction_2.svg", caption="Figure 2: New instance.")

    options = ['', 'Banana', 'Apple']
    selected_options = st.selectbox('Select answer:', options)

    
    if selected_options == 'Banana':
        st.error("Sorry :( It seems someone needs to eat more fruit and vegetables.")
    
    # Check if the selection is correct
    elif selected_options == 'Apple':
        st.success("Bravo! The correct answer is Apple.")
        st.write("You have just demonstrated your cognitive capabilities" +
                    " by adequately predicting the class of the new instance. This is not a surprise," +
                    " as you have been classifying objects since you were a child. " +
                    " We are very smart, we know, but how" +
                    " is this process done by a machine?")
        
        st.write("To answer this question, we need a dataset that we can use to perform the classification analysis." +
                 " Let's use the following dataset:")
        
        data = {
            'Weight [grams]': [72, 75, 73, 150, 155, 160],
            'Diameter [millimeters]': [55, 60, 59, 67, 73, 70],
            'Class': ['Apple', 'Apple', 'Apple', 'Banana', 'Banana', 'Banana']
        }

        df_data = pd.DataFrame(data)
        st.dataframe(df_data)

        st.write("We will use the above dataset to predict the classes of the following instances:")

        test_data = {
            'Weight [grams]': [74, 152],
            'Diameter [millimeters]': [58, 72],
            'Class': ['????', '????']
        }

        df_test_data = pd.DataFrame(test_data)
        st.dataframe(df_test_data)

        st.write("We will name the first dataset as **training dataset**, as"
                 " we are going to use it to train the *k*-NN model. By contrast, we will name the " +
                  "second dataset as **test dataset**, as we will use it to test the " +
                   "predicting capabilities of the *k*-NN model." )
        
        ## Add button if the learning is supervised, unsuperivsed or reinforcement
        st.write("A quick catch-up! Do you remember whether *k*-NN applies supervised or unsupervised learning?")
        options = ['', 'Supervised Learning', 'Unsupervised Learning']
        selected_options = st.selectbox('Select answer:', options)
        
        if selected_options == 'Unsupervised Learning':
            st.error("Unsupervised learning is performed when we want to identify patterns in the data without " +
                     "having access to the class labels. In this case, we have access to the class labels. Therefore, " +
                     "is it supervised or unsupervised learning?")
        
        elif selected_options == 'Supervised Learning':
            st.success("Correct! *k*-NN applies supervised learning as we have access to the class labels.")
        
        
        st.write("Now, that we know what **supervised learning**, and **classification analysis** are " +
                 " and we have introduced the dataset required to solve the problem defined, we are ready dig into " +
                  "the *k*-NN algorithm!" )
        
        st.write("The *k*-NN algorithm aims to predict the class of a new instance by considering the class " +
                 "of its *k* nearest neighbours. We will define the degree of neibourghness between pair of instances " +
                  "based on the utilisation of a distance metric.")
        
        st.write("Let's start with its implementation!")
        
        st.write("The **first step** is to determine the instance, the class of which we want to predict. In this example, " +
                 "we will consider the following instance:")
        
        instance = {
            'Weight [grams]': [74],
            'Diameter [millimeters]': [58],
            'Class': ['????']
        }

        df_instance = pd.DataFrame(instance)
        st.dataframe(df_instance)

        st.write("The **second step** is to calculate the distance between the instance we want to predict and the instances " +
                 "in the training dataset. This will allow us to determine which are the most similar training instances " +
                  "to the one we are analysing, and thus identifying the training nearest neighbours of the instance we want to predict.")

        st.write("The most common distance metric used in the *k*-NN algorithm is the **Euclidean distance**. " +
                 "The Euclidean distance between two instances is calculated as follows:")
        
        st.latex(r'd(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}')

        st.write('According the k-NN algorithm, now we have to apply this equation to calculate the distance' +
                 ' between the instance the class of which we want to predict and each of the training instances.' +
                 ' In order to provide an example of how to apply this equation, we will calculate the distance' +
                 ' between the instance we want to predict and the first instance in the training dataset ' +
                 '(please see the mentioned instances in the table below).')
        

        data_first = {
            'Weight [grams]': [72, 74],
            'Diameter [millimeters]': [55, 58],
            'Class': ['Apple', '????']
        }

        df_data_first = pd.DataFrame(data_first)

        def colours(data_first):
            if data_first.name == 'Weight [grams]':
                return ['color: magenta']*len(data_first)
            elif data_first.name == 'Diameter [millimeters]':
                return ['color: green']*len(data_first)
            
        df_style = df_data_first.style.apply(colours, axis=0)

        st.write(df_style.to_html(), unsafe_allow_html=True)

        st.write("Now, let's calculate the Euclidean distance:")

        st.latex(r'd(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} = \sqrt{(\textcolor{magenta}{72} - \textcolor{magenta}{74})^2 + (\textcolor{green}{55} - \textcolor{green}{58})^2} = \sqrt{4 + 9} = \sqrt{13} \approx 3.60')

        st.write("Now that we already know how to calculate the Euclidean distance between two instances, we can calculate the distance " +
                 "between the instance we want to predict and all the instances in the training dataset. So, now it is your turn to practise!")

        st.info("Please note that for the result to be accepted the number needs to be round to the second decimal " +
                   "by applying truncation. For instance, 3.609 should be rounded to 3.60.")
        
        distances = []
        
        def truncate_decimal(value, decimal_places):
            factor = 10 ** decimal_places
            return int(value * factor) / factor

        for i in range(df_data.shape[0]):
            dist = distance.euclidean(df_data.iloc[i, :-1], df_instance.iloc[0, :-1])
            distances.append(truncate_decimal(dist, 2))

        def check_distance_1():
            try:
                if float(str(st.session_state.distance_1).replace(",", ".")) == distances[0]:
                    st.session_state.is_disabled_1 = True
                else:
                    st.session_state.is_disabled_1 = False
            except:
                pass

        def check_distance_2():
            try:
                if float(str(st.session_state.distance_2).replace(",", ".")) == distances[1]:
                    st.session_state.is_disabled_2 = True
                else:
                    st.session_state.is_disabled_2 = False
            except:
                pass

        def check_distance_3():
            try:
                if float(str(st.session_state.distance_3).replace(",", ".")) == distances[2]:
                    st.session_state.is_disabled_3 = True
                else:
                    st.session_state.is_disabled_3 = False
            except:
                pass

        def check_distance_4():
            try:
                if float(str(st.session_state.distance_4).replace(",", ".")) == distances[3]:
                    st.session_state.is_disabled_4 = True
                else:
                    st.session_state.is_disabled_4 = False
            except:
                pass

        def check_distance_5():
            try:
                if float(str(st.session_state.distance_5).replace(",", ".")) == distances[4]:
                    st.session_state.is_disabled_5 = True
                else:
                    st.session_state.is_disabled_5 = False
            except:
                pass

        def check_distance_6():
            try:
                if float(str(st.session_state.distance_6).replace(",", ".")) == distances[5]:
                    st.session_state.is_disabled_6 = True
                else:
                    st.session_state.is_disabled_6 = False
            except:
                pass

        col1, col2 = st.columns([1.3, 1])

        with col1:
            st.write("**Training set (X)**")
            st.dataframe(df_data)

            st.write("**Test set (T)**")
            st.dataframe(df_instance)

        with col2:
            # Create a number input with two decimal places
            distance_1 = st.text_input("Enter the Euclidean distance between X0 and T0:", value="0.0", autocomplete=None, key="distance_1", on_change=check_distance_1(), disabled=st.session_state.is_disabled_1)
            distance_2 = st.text_input("Enter the Euclidean distance between X1 and T0:", value="0.0", autocomplete=None, key="distance_2", on_change=check_distance_2(), disabled=st.session_state.is_disabled_2)
            distance_3 = st.text_input("Enter the Euclidean distance between X2 and T0:", value="0.0", autocomplete=None, key="distance_3", on_change=check_distance_3(), disabled=st.session_state.is_disabled_3)
            distance_4 = st.text_input("Enter the Euclidean distance between X3 and T0:", value="0.0", autocomplete=None, key="distance_4", on_change=check_distance_4(), disabled=st.session_state.is_disabled_4)
            distance_5 = st.text_input("Enter the Euclidean distance between X4 and T0:", value="0.0", autocomplete=None, key="distance_5", on_change=check_distance_5(), disabled=st.session_state.is_disabled_5)
            distance_6 = st.text_input("Enter the Euclidean distance between X5 and T0:", value="0.0", autocomplete=None, key="distance_6", on_change=check_distance_6(), disabled=st.session_state.is_disabled_6)

        #try:
        if (float(str(distance_1).replace(",", ".")) == distances[0] and
                float(str(distance_2).replace(",", ".")) == distances[1] and
                float(str(distance_3).replace(",", ".")) == distances[2] and
                float(str(distance_4).replace(",", ".")) == distances[3] and
                float(str(distance_5).replace(",", ".")) == distances[4] and
                float(str(distance_6).replace(",", ".")) == distances[5]):
                st.success("Well done! You have calculated the Euclidean distances correctly.")

                st.write("Now that we have calculated the Euclidean distances between the instance we want to predict and all the instances in the training dataset, " +
                        "we can assess if how similar is the training instances to the instance the class of which we want to predict. If the resulting distance is large" +
                        " it means that the instances are dissimilar, and thus it is probable that the class of the instance we want to predict is different from the class of the training instance.")
                
                st.write("By contrast, if the resulting distance is small, it means that the instances are similar, and thus it is probable that the class of the instance we want to predict is the same as the class of the training instance." +
                        " From the resulting distances, it seems that the instance we want to predict (T0) is very similar to the training instance X0. However, it is very dissimilar to the training instance X5." +
                        " To better highlight this, we can create a table with the resulting distances ordered from the smallest to the largest distance. Can you do it for me?")
            
                data = ["d(X0, T0) = " + str(distances[0]),
                        "d(X1, T0) = " + str(distances[1]),
                        "d(X2, T0) = " + str(distances[2]),
                        "d(X3, T0) = " + str(distances[3]),
                        "d(X4, T0) = " + str(distances[4]),
                        "d(X5, T0) = " + str(distances[5]),
                ]

                sorted_items = sort_items(data)

                if (sorted_items[0] == "d(X2, T0) = " + str(distances[2]) and
                    sorted_items[1] == "d(X1, T0) = " + str(distances[1]) and
                    sorted_items[2] == "d(X0, T0) = " + str(distances[0]) and
                    sorted_items[3] == "d(X3, T0) = " + str(distances[3]) and
                    sorted_items[4] == "d(X4, T0) = " + str(distances[4]) and
                    sorted_items[5] == "d(X5, T0) = " + str(distances[5])):
                    st.success("Well done! You have ordered the distances correctly. The resulting table is shown below:")

                    final_distances = {
                        'Distance': ['d(X0, T0)', 'd(X1, T0)', 'd(X2, T0)', 'd(X3, T0)', 'd(X4, T0)', 'd(X5, T0)'],
                        'Value': distances,
                        'Class': ['Apple', 'Apple', 'Apple', 'Banana', 'Banana', 'Banana']
                    }

                    st.write("Now that we have ordered the distances based on the similarity between the instances, it is time to consider the hyperparameter *k*." +
                             " *k* represents the number of nearest neighbors to include in the prediction. Let's see a couple of examples for better understanding:")

                    st.image("./imgs/k_1.svg", caption="Figure 3: Predicted class when k is equal to 3.")       

                    st.image("./imgs/k_2.svg", caption="Figure 4: Predicted class when k is equal to 5.")

                    st.write("Once understood the importance of *k*, we can proceed classification of the test instance (T0)." +      
                        " For this, we will need to define the value of *k*, which is the number of nearest neighbours we will consider to predict the class of the test instance." +
                        " For this example, we will consider *k* = 3. Thus, we will consider the classes of the three nearest neighbours to predict the class of the test." +
                        " Can you do it for me?")
                    
                    df_final_distances = pd.DataFrame(final_distances)
                    df_final_distances = df_final_distances.sort_values(by='Value')
                    st.write(df_final_distances)

                    options = ['', 'Banana', 'Apple']
                    selected_options = st.selectbox('Predicted class for T0:', options)

                    
                    if selected_options == 'Banana':
                        st.error("Incorrect. Try again!")
                    
                    # Check if the selection is correct
                    elif selected_options == 'Apple':
                        st.success("Well done! You have just predicted the class of the test instance (T0) through the " +
                                " *k*-NN algorithm. It was easy, wasn't it?")
                        
                        st.write("In this case, you know that **the test instance (T0) belongs to the class Apple**, because you have identified that T0 is very similar to the training instances X0, X1, and X2. Also, because I said so." +
                                   " However, to adequately check whether the prediction is correct, we would need to compare the predicted class with the actual class of the test instance.")
                        
                        st.write("That said, if we do not have the actual class of the test instance, we can graphical represent the training instances and the test instance in a scatter plot (as represented below)")
                        
                        fig, ax = plt.subplots()

                        df_apple = df_data[df_data['Class'] == 'Apple']
                        df_banana = df_data[df_data['Class'] == 'Banana']

                        ax.scatter(df_apple['Weight [grams]'], df_apple['Diameter [millimeters]'], color='red', label='Apple')
                        ax.scatter(df_banana['Weight [grams]'], df_banana['Diameter [millimeters]'], color='#C49102', label='Banana')

                        ax.scatter(df_instance['Weight [grams]'], df_instance['Diameter [millimeters]'], color='blue', label='????')

                        ax.set_xlabel('Weight [grams]')
                        ax.set_ylabel('Diameter [millimeters]')

                        ax.legend(frameon=False, loc='upper left')

                        # Display the plot in the app
                        st.pyplot(fig)

                        st.write("It can be observed that the test instance with class unknown (represented in blue) is very similar to the training instances with class Apple (represented in red)." + 
                                 "Thus, we can be confident that the test instance belongs to the class Apple.")
                        
                        st.write("Now that you have understood how the *k*-NN algorithm works, you can proceed to the next tab " +
                                "to define the algorithm of the *k*-NN algorithm. Before that, here you have some tips when " +
                                " implementing the *k*-NN algorithm:")
                        
                        st.markdown("""
                                        - **Selection of *k***. The selection of the value of *k* is crucial for the performance of the *k*-NN algorithm. Even though we have define it at a later stage, the selection of the value of *k* is usually defined in the initial stages.
                                        - **Distance metric**. The selection of the distance metric is also crucial for the performance of the *k*-NN algorithm. The Euclidean distance is the most common distance metric used in the *k*-NN algorithm. However, other distance metrics can be used depending on the problem at hand.
                                        - **Feature scaling**. The *k*-NN algorithm is sensitive to the scale of the features. Thus, it is recommended to scale the features before applying the *k*-NN algorithm.
                                        - **Curse of dimensionality**. The *k*-NN algorithm is sensitive to the curse of dimensionality. Thus, it is recommended to reduce the dimensionality of the dataset before applying the *k*-NN algorithm.
                                        - **Computational cost**. The *k*-NN algorithm is computationally expensive, as it requires to calculate the distance between the instance we want to predict and all the instances in the training dataset. Thus, it is recommended to use the *k*-NN algorithm for small datasets.
                                    """)
                        
        #except Exception as e:
        #    st.error("Please, make sure that you have entered valid numbers.")
        #    print(e)