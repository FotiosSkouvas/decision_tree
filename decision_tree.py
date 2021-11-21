#import essential libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

#Statement of our multi-element containers
header = st.container()
dataset = st.container()
data_overview = st.container()
data_visualization = st.container()
data_quering = st.container()
data_modeling = st.container()

#Introduction
with header:
    st.title('Predictive Maintenance')
    st.markdown('**Sooner or Later, all machines run to a failure!**')
    st.markdown('Predictive maintenance refers to the use of data-driven, proactive maintenance methods that are designed to analyze the condition of equipment and help predict when specific conditions are met and before the equipment breaks down.')
#Insert of dataset
with dataset:
    st.header('Dataset')
    st.markdown('*Our application allows you to explore your dataset, visualize selected features, perform simple searches through queries and predict possible failures.*')
    uploaded_file = st.file_uploader("First of all, choose a CSV file:")
    if uploaded_file is not None:
        pred_data = pd.read_csv(uploaded_file)
        st.write('Below are presented the first five rows of your dataset, in order to ensure that you have provided the right dataset: ')
        st.write(pred_data.head(5))

        #Dataset Overview
        with data_overview:
            st.header('Dataset Overview')
            st.markdown('First, you will have the oportunity to see the structure of your dataset:')
            st.markdown('*Your dataset contains the following columns:*')
            st.write(pred_data.columns)
            st.markdown('Now, you will have the opportunity to check if your dataset contains outliers.')
            st.markdown('*The table below provides a description of your dataset:*')
            st.write(pred_data.describe())
            st.markdown('The non NULL analysis provided below, will help you check if there are missing values in your dataset.')
            st.markdown('*Non NULL values analysis:*')
            st.write(pred_data.count())
            st.markdown('*Here, we should provide a description of the values of the dataset, in order to check the dataset for format issues, but there is a problem with dtypes function at streamlit (Unable to convert numpy.dtype to pyarrow.DataType. This is likely due to a bug in Arrow).*')
            st.header('Data Cleaning')
            clean = st.radio(
            'Is your data clean?',
            ('Yes', 'No'))
            if clean == 'Yes':
                st.success('You can proceed with the analysis')

                #Data visualization options
                with data_visualization:
                    st.header('Data Visualization')
                    st.markdown('In this section you will have the opportunity to visualize your dataset as a scatter plot or as a pie chart. The scatter plot visualizes the comparison of two selected values. The pie chart provides visualization of the statistics for one selected value of your dataset.')
                    opt_vis = st.radio(
                    'Please select your preferred plot: ',
                    ('Scatter Plot', 'Pie Chart'))
                    if opt_vis == 'Scatter Plot':
                        x_axis = st.selectbox('Please select x axis:',
                        pred_data.columns)
                        y_axis = st.selectbox('Please select y axis:',
                        pred_data.columns)
                        st.markdown('*Scatter Plot*')
                        fig_1 = px.scatter(pred_data, x = pred_data[x_axis], y = pred_data[y_axis])
                        st.plotly_chart(fig_1)
                    else:
                        st.markdown('*Pie chart*')
                        val_1 = st.selectbox('Please select the value:',
                        pred_data.columns)
                        fig_2, ax = plt.subplots()
                        ax.pie(pred_data[val_1].value_counts(), labels = pred_data[val_1].unique(), autopct='%1.1f%%')
                        ax.set_title(val_1)
                        st.pyplot(fig_2)

                #Data quering
                with data_quering:
                    st.header('Possible Cause')
                    st.markdown('You will now have the opportunity to compare a system failure with one possible cause:')
                    root_cause = st.selectbox('Please select a failure category of your system:',
                    pred_data.columns)
                    pred_data_root = pred_data[pred_data[root_cause]==1]
                    st.write('Total number of this category failures in your database: ', len(pred_data_root))
                    percentage = len(pred_data_root)/len(pred_data)*100
                    st.write(percentage, '% of observations failed due to this failure category')
                    st.markdown('Which features contribution to selected failure category would you like to observe for the selected root cause?')
                    feat = st.selectbox('Please select a feature of your system:',
                    pred_data.columns)
                    pred_data_feat = pred_data_root[feat].value_counts()
                    st.write(pred_data_feat)

                #Data modeling
                with data_modeling:
                    st.header('Model Training')
                    st.markdown('Our application will now try to make predictions for your selected failure. Our application uses the decision tree method.')
                    st.markdown('*The following table shows the actual and the predicted values of the selected failure: *')    
                    X = pred_data.drop(['Product ID', 'Type', root_cause], axis=1)
                    y = pred_data[root_cause]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                    regressor = DecisionTreeRegressor()
                    regressor.fit(X_train, y_train)
                    DecisionTreeRegressor()
                    y_pred = regressor.predict(X_test)
                    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                    st.write(df)
                    #Evaluation of model
                    st.markdown('**Evaluation of the model**')
                    st.markdown('*The metrics below are used to evaluate the predictions of our model:*')
                    st.write('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
                    st.write('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
                    st.write('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
                    st.write('Accuracy:', np.sqrt(metrics.accuracy_score(y_test, y_pred)))



            else:
                st.error('Please clean your data and upload the updated CSV file to continue')
    else:
        st.error('Please upload a CSV file to continue')
