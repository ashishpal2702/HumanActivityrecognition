import gradio as gr
import plotly.express as px
from io import StringIO
from zipfile import ZipFile
import pandas as pd
from src.utils.helper import (
    load_config,
    update_log,
    load_weights,
)
from predict import Prediction



def plotly_plot(df):
    # prepare some data
    x = df['prediction_label'].unique()
    y = df['prediction_label'].value_counts().values
    data = pd.DataFrame()
    data['Activity'] = x
    data['count'] = y
    # create a new plot
    p = px.bar(data, x='Activity', y='count')

    return p


def model_predict(df):
    config = load_config()
    prediction = Prediction(config).live_predict(df)
    return prediction

def upload_file(file):
    print(file)
    #df = pd.read_csv('/Users/apal/Documents/PathtoAI/AnalyticsVidhya/Mlops/data/Human_Activity_Recognition_Using_Smartphones_TestData.csv') #('./data/tmp.csv')
    #df = pd.read_csv(data)
    df = pd.read_csv(file.name, encoding='utf-8')

    prediction = model_predict(df)
    plot = plotly_plot(df)

    return prediction.head() , plot

demo = gr.Interface(
    upload_file,
    gr.File(file_count="single", file_types=[".csv"]),
    [gr.Dataframe(),gr.Plot()],
    cache_examples=True
)

if __name__ == "__main__":
    demo.launch(share=True)