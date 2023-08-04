# HumanActivityrecognition


<!-- Execution -->
<h2 id="execution"> 🍴 Execution</h2>

<!--This project is written in Python programming language. <br>-->
The open source packages used in this project is available in [requirements.txt](requirements.txt)

### Setup

Clone this repo to your desktop :

To install all the dependencies.

Run

`pip install -e .`

`pip install -r requirements.txt`

### Usage

Download Data from
https://www.kaggle.com/datasets/pathtoai/human-activity-recognition-using-smartphone

Update the training data path in [config.toml](./config/config.toml)


For Model Training: 

`python train.py`

For Batch Prediction: 

`python predict.py`

For Model Real time Prediction: 

`python inference.py <model_weight> <test_data> <output_data>`


For Model Inference Application : 

`python app/app.py`
