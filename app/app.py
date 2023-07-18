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

    activity = st.radio(
        "Select Activity to get sensors value",
        ('STANDING',  'SITTING','LAYING','WALKING', 'WALKING_UPSTAIRS','WALKING_DOWNSTAIRS'))

    if activity == 'STANDING':
        val = [ 0.97657704,  0.88534827, -0.81188833, -0.13432725,  0.95804439,
       -0.17985003,  0.8843957 ,  0.19233134, -0.15910602, -0.96162393]
    elif activity == 'SITTING':
        val = [ 0.98520393,  0.90727763, -0.89278795,  0.11383441,  0.96635272,
        0.06377597,  0.8925701 ,  0.02086838,  0.09212252, -0.97656826]
    elif activity == 'LAYING':
        val = [-0.35485178, -0.99486021,  0.53911539,  0.97368142, -0.40004035,
        0.91000875, -0.45805357, -0.84447893,  0.96236333,  0.8544026]
    elif activity == 'WALKING':
        val = [ 0.91899885,  0.74466009, -0.66407709, -0.24594059,  0.90354823,
       -0.28391389,  0.83465449,  0.26795461, -0.26845704, -0.87903034]
    elif activity == 'WALKING_UPSTAIRS':
        val = [ 0.95453392,  0.92233673, -0.7945455 , -0.21143753,  0.97189159,
       -0.17007847,  0.92337829,  0.21196459, -0.18991835, -0.94123471]
    elif activity == 'WALKING_DOWNSTAIRS':
        val = [ 0.90818192,  0.75996929, -0.71123519, -0.17506794,  0.90954198,
       -0.20938135,  0.84816272,  0.2213512 , -0.19942022, -0.93653733]
    else:
        val = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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
    #feature10 = st.number_input(label ='fBodyAccJerk-entropy()-X',value = val[9])

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
    df = pd.DataFrame(data_dict, index=[0])

    if st.button('Predict'):
        prediction = model_predict(df)
        st.write('Model Prediction is : ', str(prediction['prediction_label'].values))
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