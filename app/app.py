import streamlit as st
import random
import pandas as pd
from io import StringIO
from PIL import Image
from src.utils.helper import (
    load_config,
    update_log,
    load_weights,
)
from predict import Prediction

df = pd.DataFrame()
config = load_config()

def model_predict(df):
    config = load_config()
    prediction = Prediction(config).model_predict(df.reshape(1, -1),0.5)
    le = load_weights(config["encoder_weights"])
    prediction = le.inverse_transform(prediction)
    return prediction

def model_batch_predict(df):
    config = load_config()
    prediction = Prediction(config).live_predict(df)
    return prediction

tab1, tab2 = st.tabs(["Real Time Prediction", "Batch Prediction"])
cols = ['tGravityAcc-min()-X','tGravityAcc-energy()-X','angle(X,gravityMean)','tGravityAcc-min()-Y','tGravityAcc-mean()-X',
           'tGravityAcc-max()-Y', 'tGravityAcc-max()-X','angle(Y,gravityMean)','tGravityAcc-mean()-Y','tGravityAcc-energy()-Y']

df = pd.read_csv(config['testdata_file'])
processor = load_weights(config["feature_pipeline"])
df_transformed = processor.transform(df)
df = df_transformed#[cols]
print(df)

with tab1:
    st.header("Real Time Prediction Watch")
    st.header("  Generate Sendor Value ")
    global val
    val = [0,0,0,0,0,0,0,0,0,0]
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open('./app/phone.jpeg')

        st.image(image, caption='Smart Phone Activity Tracker')

        random_int = st.slider('Select a range of Random Sensor value',0, len(df))
        val = df[random_int]#df.values[int(random_int)]

    with col2:

        feature1 = st.number_input(label = 'tGravityAcc-min()-X' ,value = val[0])
        feature2 = st.number_input(label ='tGravityAcc-energy()-X',value = val[1])
        feature3 = st.number_input(label ='angle(X,gravityMean)',value = val[2])
        feature4 = st.number_input(label ='tGravityAcc-min()-Y',value = val[3])
        feature5 = st.number_input(label ='tGravityAcc-mean()-X',value = val[4])
        feature6 = st.number_input(label ='tGravityAcc-max()-Y',value = val[5])
        feature7 = st.number_input(label ='tGravityAcc-max()-X',value = val[6])
        feature8 = st.number_input(label ='angle(Y,gravityMean)',value = val[7])
        feature9 = st.number_input(label ='tGravityAcc-mean()-Y',value = val[8])
        feature10 = st.number_input(label='tGravityAcc-energy()-Y', value=val[9])
        feature10 = st.number_input(label ='fBodyAccJerk-entropy()-X',value = val[9])

        data_dict = {       'tGravityAcc-min()-X':      feature1,
                            'tGravityAcc-energy()-X':   feature2,
                           'angle(X,gravityMean)':      feature3,
                           'tGravityAcc-min()-Y':       feature4,
                           'tGravityAcc-mean()-X':      feature5,
                           'tGravityAcc-max()-Y':       feature6,
                           'tGravityAcc-max()-X':       feature7,
                           'angle(Y,gravityMean)':     feature8,
                           'tGravityAcc-mean()-Y':     feature9,
                           'tGravityAcc-energy()-Y':        feature10
                     }
        #test_df = pd.DataFrame(data_dict, index=[0])
        #print(test_df)
        test_df = val#pd.DataFrame(df.iloc[random_int,:])
        #if st.button('Predict'):
        prediction = model_predict(test_df)
        st.write('Model Prediction is : ', prediction)
            #st.write(prediction.T)

with tab2:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Choose a file")

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
        prediction = model_batch_predict(df)
        print(prediction.head()['prediction_label'])
        pred_df = pd.DataFrame(prediction['prediction_label'].value_counts())
        pred_df.columns = ['minutes']
        st.bar_chart(pred_df)

        st.write(pred_df)