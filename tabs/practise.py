import streamlit as st
import pandas as pd
from scipy.spatial import distance
import numpy as np

from sklearn.datasets import make_classification
import streamlit.components.v1 as components

def practise():
    if 'submit' not in st.session_state:
        st.session_state['submit'] = False

    if 'p_distance_1' not in st.session_state:
        st.session_state['p_distance_1'] = 0.0
    
    if 'p_is_disabled_1' not in st.session_state:
        st.session_state['p_is_disabled_1'] = False

    if 'p_distance_2' not in st.session_state:
        st.session_state['p_distance_2'] = 0.0
    
    if 'p_is_disabled_2' not in st.session_state:
        st.session_state['p_is_disabled_2'] = False

    if 'p_distance_3' not in st.session_state:
        st.session_state['p_distance_3'] = 0.0
    
    if 'p_is_disabled_3' not in st.session_state:
        st.session_state['p_is_disabled_3'] = False

    if 'p_distance_4' not in st.session_state:
        st.session_state['p_distance_4'] = 0.0
        
    if 'p_is_disabled_4' not in st.session_state:
        st.session_state['p_is_disabled_4'] = False

    if 'p_distance_5' not in st.session_state:
        st.session_state['p_distance_5'] = 0.0
        
    if 'p_is_disabled_5' not in st.session_state:
        st.session_state['p_is_disabled_5'] = False

    if 'p_distance_6' not in st.session_state:
        st.session_state['p_distance_6'] = 0.0
        
    if 'p_is_disabled_6' not in st.session_state:
        st.session_state['p_is_disabled_6'] = False

    if 'p_distance_7' not in st.session_state:
        st.session_state['p_distance_7'] = 0.0
        
    if 'p_is_disabled_7' not in st.session_state:
        st.session_state['p_is_disabled_7'] = False

    if 'p_distance_8' not in st.session_state:
        st.session_state['p_distance_8'] = 0.0
        
    if 'p_is_disabled_8' not in st.session_state:
        st.session_state['p_is_disabled_8'] = False

    if 'p_distance_9' not in st.session_state:
        st.session_state['p_distance_9'] = 0.0
        
    if 'p_is_disabled_9' not in st.session_state:
        st.session_state['p_is_disabled_9'] = False

    if 'p_distance_10' not in st.session_state:
        st.session_state['p_distance_10'] = 0.0
        
    if 'p_is_disabled_10' not in st.session_state:
        st.session_state['p_is_disabled_10'] = False

    def submitted():
        st.session_state.submit = True

    st.header("Practise!")
    
    if 'form_disabled' not in st.session_state:
        st.session_state['form_disabled'] = False

    if 'df_data' not in st.session_state:
        st.session_state['df_data'] = pd.DataFrame()
    
    if 'df_instance' not in st.session_state:
        st.session_state['df_instance'] = pd.DataFrame()

    if 'y' not in st.session_state:
        st.session_state['y'] = -1

    if 'n_features' not in st.session_state:
        st.session_state['n_features'] = 0

    if 'n_classes' not in st.session_state:
        st.session_state['n_classes'] = 0


    def submit_form(n_features, n_classes):

        try:
            st.session_state['n_features'] = n_features
            st.session_state['n_classes'] = n_classes
            n_samples = 11

            X, y = make_classification(n_features=int(n_features), n_classes=int(n_classes),
                                        n_informative=int(n_features), n_redundant=0, n_repeated=0, n_samples=n_samples)

            df_data = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(int(n_features))])
            df_data = pd.concat([df_data, pd.Series(y, name="Target")], axis=1)
            df_data = df_data.round(2)
            
            letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

            for i in range(df_data.shape[0]):
                df_data.iloc[i, -1] = letters[df_data.iloc[i, -1]]

            df_instance = pd.DataFrame(df_data.iloc[-1, :].values.reshape(1, -1), columns=df_data.columns)
            y = df_data.iloc[-1, -1]
            df_instance.iloc[-1, -1] = '????'
            df_data = df_data.drop(df_data.tail(1).index)

            st.session_state.df_data = df_data
            st.session_state.df_instance = df_instance
            st.session_state.y = y

            st.success("Dataset generated successfully! **Please press enter to continue.**")
            st.session_state['form_disabled'] = True
        
        except Exception as e:
            print(e)
            st.error("An error occurred while generating the dataset. Please try again.")
            st.error("**Reminder**. For adequate performance of the *k*-NN algorithm, it is recommended that the number of classes is smaller or equal than the number of features.")


    if not st.session_state.form_disabled:
        st.write("**Step 1. Generate the dataset**")

        with st.form(key="generate_dataset"):
            n_features = st.number_input("Number of features:", min_value=2, value='min', max_value=6, step=1, key="columns")
            n_classes = st.number_input("Number of classes:", min_value=2, value='min', max_value=10, step=1, key="classes")
            submit_button = st.form_submit_button("Generate dataset")

            if submit_button:
                submit_form(n_features, n_classes)

    else:
        st.write("**Step 1. Generate the dataset**")

        with st.form(key="disabled_generate_dataset"):
            n_features = st.number_input("Number of features:", value=st.session_state.n_features, key="disabled_columns", disabled=True)
            n_classes = st.number_input("Number of classes:", value=st.session_state.n_classes, key="disabled_classes", disabled=True)
            submit_button = st.form_submit_button("Generate dataset", disabled=True)

        st.write("**Step 2. Calculate the distance between the new instance and all instances in the training set**")

        distances = []
        df_data = st.session_state.df_data
        df_instance = st.session_state.df_instance
        y = st.session_state.y
        
        def truncate_decimal(value, decimal_places):
            factor = 10 ** decimal_places
            return int(value * factor) / factor

        for i in range(df_data.shape[0]):
            dist = distance.euclidean(df_data.iloc[i, :-1], df_instance.iloc[0, :-1])
            distances.append(truncate_decimal(dist, 2))

        def p_check_distance_1():
            try:
                if float(str(st.session_state.p_distance_1).replace(",", ".")) == distances[0]:
                    st.session_state.p_is_disabled_1 = True
                else:
                    st.session_state.p_is_disabled_1 = False
            except:
                pass

        def check_distance_2():
            try:
                if float(str(st.session_state.p_distance_2).replace(",", ".")) == distances[1]:
                    st.session_state.p_is_disabled_2 = True
                else:
                    st.session_state.p_is_disabled_2 = False
            except:
                pass

        def check_distance_3():
            try:
                if float(str(st.session_state.p_distance_3).replace(",", ".")) == distances[2]:
                    st.session_state.p_is_disabled_3 = True
                else:
                    st.session_state.p_is_disabled_3 = False
            except:
                pass

        def check_distance_4():
            try:
                if float(str(st.session_state.p_distance_4).replace(",", ".")) == distances[3]:
                    st.session_state.p_is_disabled_4 = True
                else:
                    st.session_state.p_is_disabled_4 = False
            except:
                pass

        def check_distance_5():
            try:
                if float(str(st.session_state.p_distance_5).replace(",", ".")) == distances[4]:
                    st.session_state.p_is_disabled_5 = True
                else:
                    st.session_state.p_is_disabled_5 = False
            except:
                pass

        def check_distance_6():
            try:
                if float(str(st.session_state.p_distance_6).replace(",", ".")) == distances[5]:
                    st.session_state.p_is_disabled_6 = True
                else:
                    st.session_state.p_is_disabled_6 = False
            except:
                pass

        def check_distance_7():
            try:
                if float(str(st.session_state.p_distance_7).replace(",", ".")) == distances[6]:
                    st.session_state.p_is_disabled_7 = True
                else:
                    st.session_state.p_is_disabled_7 = False
            except:
                pass

        def check_distance_8():
            try:
                if float(str(st.session_state.p_distance_8).replace(",", ".")) == distances[7]:
                    st.session_state.p_is_disabled_8 = True
                else:
                    st.session_state.p_is_disabled_8 = False
            except:
                pass

        def check_distance_9():
            try:
                if float(str(st.session_state.p_distance_9).replace(",", ".")) == distances[8]:
                    st.session_state.p_is_disabled_9 = True
                else:
                    st.session_state.p_is_disabled_9 = False
            except:
                pass
        def check_distance_10():
            try:
                if float(str(st.session_state.p_distance_10).replace(",", ".")) == distances[9]:
                    st.session_state.p_is_disabled_10 = True
                else:
                    st.session_state.p_is_disabled_10 = False
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
            distance_1 = st.text_input("Enter the Euclidean distance between X0 and T0:", value="0.0", autocomplete=None, key="p_distance_1", on_change=p_check_distance_1(), disabled=st.session_state.p_is_disabled_1)
            distance_2 = st.text_input("Enter the Euclidean distance between X1 and T0:", value="0.0", autocomplete=None, key="p_distance_2", on_change=check_distance_2(), disabled=st.session_state.p_is_disabled_2)
            distance_3 = st.text_input("Enter the Euclidean distance between X2 and T0:", value="0.0", autocomplete=None, key="p_distance_3", on_change=check_distance_3(), disabled=st.session_state.p_is_disabled_3)
            distance_4 = st.text_input("Enter the Euclidean distance between X3 and T0:", value="0.0", autocomplete=None, key="p_distance_4", on_change=check_distance_4(), disabled=st.session_state.p_is_disabled_4)
            distance_5 = st.text_input("Enter the Euclidean distance between X4 and T0:", value="0.0", autocomplete=None, key="p_distance_5", on_change=check_distance_5(), disabled=st.session_state.p_is_disabled_5)
            distance_6 = st.text_input("Enter the Euclidean distance between X5 and T0:", value="0.0", autocomplete=None, key="p_distance_6", on_change=check_distance_6(), disabled=st.session_state.p_is_disabled_6)
            distance_7 = st.text_input("Enter the Euclidean distance between X6 and T0:", value="0.0", autocomplete=None, key="p_distance_7", on_change=check_distance_7(), disabled=st.session_state.p_is_disabled_7)
            distance_8 = st.text_input("Enter the Euclidean distance between X7 and T0:", value="0.0", autocomplete=None, key="p_distance_8", on_change=check_distance_8(), disabled=st.session_state.p_is_disabled_8)
            distance_9 = st.text_input("Enter the Euclidean distance between X8 and T0:", value="0.0", autocomplete=None, key="p_distance_9", on_change=check_distance_9(), disabled=st.session_state.p_is_disabled_9)
            distance_10 = st.text_input("Enter the Euclidean distance between X9 and T0:", value="0.0", autocomplete=None, key="p_distance_10", on_change=check_distance_10(), disabled=st.session_state.p_is_disabled_10)

        try:
            if (float(str(distance_1).replace(",", ".")) == distances[0] and
                float(str(distance_2).replace(",", ".")) == distances[1] and
                float(str(distance_3).replace(",", ".")) == distances[2] and
                float(str(distance_4).replace(",", ".")) == distances[3] and
                float(str(distance_5).replace(",", ".")) == distances[4] and
                float(str(distance_6).replace(",", ".")) == distances[5] and
                float(str(distance_7).replace(",", ".")) == distances[6] and
                float(str(distance_8).replace(",", ".")) == distances[7] and
                float(str(distance_9).replace(",", ".")) == distances[8] and
                float(str(distance_10).replace(",", ".")) == distances[9]):
                
                st.success("Well done! You have calculated the Euclidean distances correctly.")

                st.write("**Step 3. Sort the distances in ascending order**")

                distance_data = [
                    {"Distance": "Distance 1 (X0, T0)", "Value": distances[0]},
                    {"Distance": "Distance 2 (X1, T0)", "Value": distances[1]},
                    {"Distance": "Distance 3 (X2, T0)", "Value": distances[2]},
                    {"Distance": "Distance 4 (X3, T0)", "Value": distances[3]},
                    {"Distance": "Distance 5 (X4, T0)", "Value": distances[4]},
                    {"Distance": "Distance 6 (X5, T0)", "Value": distances[5]},
                    {"Distance": "Distance 7 (X6, T0)", "Value": distances[6]},
                    {"Distance": "Distance 8 (X7, T0)", "Value": distances[7]},
                    {"Distance": "Distance 9 (X8, T0)", "Value": distances[8]},
                    {"Distance": "Distance 10 (X9, T0)", "Value": distances[9]},
                ]

                user_inputs = []

                col1, col2 = st.columns(2)
                    
                with col1:
                    st.write(pd.DataFrame(distance_data))
                                        
                with col2:
                    for i, row in enumerate(distance_data):
                        try:
                            user_input = st.text_input(f"Order Index for {row['Distance']}", max_chars=1, value=str(i), key=row["Distance"])
                            user_inputs.append(int(user_input))
                        except:
                            st.error("Please, make sure that you have entered valid order index.")

                sort_distances = np.sort(distances)

                st.write(sort_distances)
                st.write(distances)

                try:

                    if (distances[user_inputs.index(0)] == sort_distances[0] and
                        distances[user_inputs.index(1)] == sort_distances[1] and
                        distances[user_inputs.index(2)] == sort_distances[2] and
                        distances[user_inputs.index(3)] == sort_distances[3] and
                        distances[user_inputs.index(4)] == sort_distances[4] and
                        distances[user_inputs.index(5)] == sort_distances[5] and
                        distances[user_inputs.index(6)] == sort_distances[6] and
                        distances[user_inputs.index(7)] == sort_distances[7] and
                        distances[user_inputs.index(8)] == sort_distances[8] and
                        distances[user_inputs.index(9)] == sort_distances[9]):

                        st.success("Well done! You have sorted the distances correctly.")

                        st.write("**Step 4. Select the *k* instances with the smallest distances**")

                        distance_data = pd.DataFrame(distance_data)
                        distance_data["Class"] = df_data["Target"]
                        distance_data = distance_data.sort_values(by="Value")

                        st.write(distance_data)

                        k = st.number_input("Enter the value of *k*:", min_value=1, max_value=10, step=1, key="k")

                        st.write("**Step 5. Assign the new instance to the class that appears most frequently among the *k* instances**")

                        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

                        k_options = letters[:n_classes]
                        k_options = [''] + k_options

                        k_instances = distance_data.head(k).iloc[:, -1].values
                        predicted_class = pd.DataFrame(k_instances).value_counts().idxmax()

                        selected_options = st.selectbox('Select answer:', k_options)
                        
                        if selected_options == predicted_class[0]:
                            st.success("Well done! You have adequately implemented the *k*-NN algorithm!")
                            st.balloons()

                        elif selected_options != '':
                            st.error("Incorrect answer. Please, try again.")
                            
                except Exception as e:
                    print(e)
                    pass

        except Exception as e:
            st.error("Please, make sure that you have entered valid numbers.")
            print(e)