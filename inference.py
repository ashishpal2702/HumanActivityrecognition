import pandas as pd
import numpy as np
import datetime
import sys
from src.utils.helper import (
    load_config,
    update_log,
    load_weights,
)
from src.data.make_dataset import Dataset
from src.features.build_features import BuildFeatures
from src.models.predict_model import Predict

class Inference:
    def __init__(self, config):
        self.config = config
        self.md = Dataset(self.config)
        self.feat = BuildFeatures(self.config)

    def build_test_feature(self, df):
        df = self.feat.preprocess(df)
        #cols = load_weights(self.config["feat_col"])
        #print(cols)
        #df = df[cols]
        df_transformed = self.feat.test_data_preprocessing_pipeline(df)
        print(df_transformed.shape)
        return df_transformed

    def model_predict(self, df, model_weights_path, threshold):
        p = Predict(threshold)
        model = p.load_model(model_weights_path)
        prediction = p.model_predict(model, df)
        return prediction

    def live_predict(self, df,model_weights_path):
        threshold = 0.5
        x_test_std = self.build_test_feature(df)
        df['prediction'] = self.model_predict(x_test_std, model_weights_path, threshold)
        le = load_weights(self.config["encoder_weights"])
        df['prediction_label'] = le.inverse_transform(df['prediction'])
        return df

if __name__ == "__main__":

    update_log("Predict Activity")
    config = load_config()
    model_weights_path = sys.argv[1]
    test_df = pd.read_csv(sys.argv[2])
    output_path = sys.argv[3]
    print(model_weights_path)
    print(test_df)
    print(output_path)
    prediction_table = Inference(config).live_predict(test_df, model_weights_path)
    print(prediction_table.head())

    prediction_table.to_csv(output_path)