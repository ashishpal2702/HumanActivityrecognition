import streamlit as st
import random
import pandas as pd
from io import StringIO
from PIL import Image
import joblib


train_features = joblib.load("./model_features/train_features.joblib")

model = joblib.load("./model_weights/my_random_forest.joblib")

data = pd.read_csv('/Users/apal/Documents/PathtoAI/AnalyticsVidhya/Mlops/data/test_data.csv')

le = joblib.load("./model_features/encoder_weights.joblib")

def model_predict(df,model,le):
    #new_data_features = df[train_features]
    print("shape = ", df.shape)
    print(df)
    prediction = model.predict(df)
    prediction = le.inverse_transform(prediction)
    return prediction

def model_batch_predict(df,train_features,model,le):
    new_data_features = df[train_features]
    prediction = model.predict(new_data_features)
    prediction = le.inverse_transform(prediction)
    df['Prediction_label'] = prediction
    return prediction

tab1, tab2 = st.tabs(["Real Time Prediction", "Batch Prediction"])
df_transformed = data[train_features]
with tab1:
    st.header("Real Time Prediction Watch")
    st.header("  Generate Sendor Value ")
    global val
    val = [0,0,0,0,0,0,0,0,0,0]
    col1, col2 = st.columns(2)
    with col1:
        image = Image.open('../app/phone.jpeg')

        st.image(image, caption='Smart Phone Activity Tracker')

        random_int = st.slider('Select a range of Random Sensor value',0, len(df_transformed))
        val = df_transformed.iloc[random_int].values
    
        
        

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
                           'tGravityAcc-energy()-Y':     feature10
                     }

        test_df = pd.DataFrame(val.reshape(1, -1))
        print(test_df.shape)
        #if st.button('Predict'):
        print(test_df)
        prediction = model_predict(test_df,model,le)
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
        prediction = model_batch_predict(df,train_features,model,le)
        print(prediction.head()['prediction_label'])
        pred_df = pd.DataFrame(prediction['prediction_label'].value_counts())
        pred_df.columns = ['minutes']
        st.bar_chart(pred_df)

        st.write(pred_df)
        
        