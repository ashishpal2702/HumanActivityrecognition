import logging
import pandas as pd
import numpy as np
import datetime
from src.utils.helper import (
    load_config,
    load_credentials,
    update_log,
    save_data,
    load_weights,
)
from src.connector.connections import Connections
from src.data.make_dataset import MakeDataset
from src.features.build_features import BuildFeatures
from src.models.predict_model import Predict


class Prediction:
    def __init__(self, config):
        self.config = config
        self.md = MakeDataset(self.config)
        self.feat = BuildFeatures(self.config)

    def build_test_feature(self, df):
        df = self.feat.preprocess(df)
        cols = load_weights(self.config["feat_col"])
        print(cols)
        df = df[cols]
        df_transformed = self.feat.test_data_preprocessing_pipeline(df)
        print(df_transformed.shape)
        return df_transformed

    def model_predict(self, df, threshold):
        p = Predict(threshold)
        model = p.load_model(self.config["model_weights"])
        prediction = p.model_predict(model, df)
        return prediction

    def bacth_predict(self):
        """Load data from for Model training and train model and save the weights."""
        threshold = 0.5
        update_log("making feature data set from raw data")
        df = self.md.read_data(self.config["feature_file"])
        test_df = self.md.make_test_dataset(df)
        x_test_std = self.build_test_feature(test_df)
        prediction = self.model_predict(x_test_std, threshold)
        df["Churn_prediction"] = prediction
        save_data(df, self.config["model_prediction"], update=True)
        # Save daily model prediction
        dt_now = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        daily_model_prediction_file = (
            self.config["daily_prediction_dir"] + str(dt_now) + ".csv"
        )
        save_data(df, daily_model_prediction_file, update=False)

    def live_predict(self, df):
        threshold = 0.5
        x_test_std = self.build_test_feature(df)
        prediction = self.model_predict(x_test_std, threshold)
        return prediction


if __name__ == "__main__":
    update_log("Prediction customers who are likely to churn")
    config = load_config()
    Prediction(config).bacth_predict()