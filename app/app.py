import streamlit as st
import pandas as pd
from io import StringIO
from src.utils.helper import (
    load_config,
    update_log,
    load_weights,
)
from predict import Prediction

df = pd.DataFrame()

def model_predict(df):
    config = load_config()
    prediction = Prediction(config).live_predict(df)
    return prediction


tab1, tab2 = st.tabs(["Real Time Prediction", "Batch Prediction"])

with tab1:
    st.header("Real Time Prediction")
    st.header("  Features ")

    feature1 = st.number_input(label = 'tGravityAcc-min()-X')
    feature2 = st.number_input(label ='tGravityAcc-energy()-X')
    feature3 = st.number_input(label ='angle(X,gravityMean)')
    feature4 = st.number_input(label ='tGravityAcc-min()-Y')
    feature5 = st.number_input(label ='tGravityAcc-mean()-X')
    feature6 = st.number_input(label ='tGravityAcc-max()-Y')
    feature7 = st.number_input(label ='tGravityAcc-max()-X')
    feature8 = st.number_input(label ='angle(Y,gravityMean)')
    feature9 = st.number_input(label ='tGravityAcc-mean()-Y')
    feature10 = st.number_input(label ='fBodyAccJerk-entropy()-X')

    data_dict = {       'tGravityAcc-min()-X':      feature1,
                        'tGravityAcc-energy()-X':   feature2,
                       'angle(X,gravityMean)':      feature3,
                       'tGravityAcc-min()-Y':       feature4,
                       'tGravityAcc-mean()-X':      feature5,
                       'tGravityAcc-max()-Y':       feature6,
                       'tGravityAcc-max()-X':       feature7,
                       'angle(Y,gravityMean)':     feature8,
                       'tGravityAcc-mean()-Y':     feature9,
                       'fBodyAccJerk-entropy()-X':        feature10
                 }
    df = pd.DataFrame(data_dict, index=[0])

    if st.button('Predict'):
        st.write('Model Prediction')
        prediction = model_predict(df)
        st.write(prediction.T)

with tab2:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Choose a file")

    #option = st.selectbox('Choose Model for Predictions',('Logistic Regression', 'Decision Tree', 'Random Forest'))

    #st.write('You selected:', option)



    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        #st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        #st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
        df = dataframe
        st.write("Data uploaded successfully")
        #st.write(dataframe)

    if st.button('Batch Predict'):
        st.write('Model Prediction')
        prediction = model_predict(df)
        print(prediction.head()['prediction_label'])
        pred_df = pd.DataFrame(prediction['prediction_label'].value_counts())
        pred_df.columns = ['minutes']
        st.bar_chart(pred_df)

        st.write(pred_df)