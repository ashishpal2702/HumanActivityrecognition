import logging
import pandas as pd
import numpy as np
from src.utils.helper import (
    load_config,
    update_log,
    save_weights,
)
from src.data.make_dataset import Dataset
from src.features.build_features import BuildFeatures
from src.models.train_model import Train

class Training:
    def __init__(self, config, k):
        self.config = config
        self.k = k
        self.top_k_cols = []
        self.feat = BuildFeatures(self.config)
    ## Select Top K features from the data
    def get_best_features(self, X, Y):
        X = self.feat.preprocess(X)
        numerical_features, categorical_features = self.feat.get_features_col(X)
        X_encode = self.feat.encode_data(X.copy(), categorical_features)
        self.top_k_cols = self.feat.get_top_k_features(X_encode, Y, self.k)
        print(self.top_k_cols)
        return self.top_k_cols
    ## Transform data for train and test dataset
    def build_feature(self, X, mode):
        if mode == "train":
            numerical_features, categorical_features = self.feat.get_features_col(X)
            df_transformed = self.feat.train_data_preprocessing_pipeline(
                X, numerical_features, categorical_features
            )
            return df_transformed
        if mode == "test":
            df_transformed = self.feat.test_data_preprocessing_pipeline(X)
            return df_transformed
    ## Main training function
    def train(self):
        """Load data from for Model training and train model and save the weights."""
        update_log("making feature data set from raw data")
        md = Dataset(self.config)
        train = Train()
        # Read Training Dataset
        df = md.read_data(self.config["training_data_file"])
        ## EDA
        md.data_analysis(df)
        ## Preprocess Data and remove Date time column
        df = md.preprocess(df, 'date_time')
        # Make Data set for Training
        X, Y = md.make_train_dataset(df,'Activity')
        ## Select top 10 features from the data set
        top_k_features = self.get_best_features(X, Y)
        X = X[top_k_features]
        ## Generate Training and Test Data set
        x_train, x_test, y_train, y_test = md.data_split(X, Y)
        ## Standardised Train and test dataset for Linear models
        x_train_std = self.build_feature(x_train, mode="train")
        x_test_std = self.build_feature(x_test, mode="test")
        models = []
        accuracy_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        business_profit = []
        ## Run Training for Different Models and get model metrics
        for model in train.get_classification_models():
            score, f1, precision, recall = train.train_and_predict(
                model, x_train_std, y_train, x_test_std, y_test
            )
            print(model)
            print(score, f1, precision, recall)
            models.append(type(model).__name__)
            accuracy_scores.append(score)
            #roc_auc_scores.append(roc)
            f1_scores.append(f1)
            precision_scores.append(f1)
            recall_scores.append(f1)

        Model_comarison = pd.DataFrame(
            {
                "Model": models,
                "Accuracy": accuracy_scores,
                "F1": f1_scores,
                "Precision": precision_scores,
                "Recall": recall_scores
            }
        )
        print(Model_comarison)


if __name__ == "__main__":
    update_log("Start Model training on latest processed data")
    k = 10
    config = load_config()
    Training(config,k).train()
