import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
from src.utils.helper import save_weights, load_weights

class Dataset:
    def __init__(self, config):
        self.config = config

    def read_data(self, file_path):
        df = pd.read_parquet(file_path)
        return df

    def data_analysis(self, df):
        print(df.describe(include="all"))
        print(df.info())
        print(df.isna().sum())

    def make_train_dataset(self, df, y_col):
        le = LabelEncoder()
        df[y_col] = le.fit_transform(df[y_col])
        X = df.drop([y_col], axis=1)
        Y = df[y_col]
        save_weights(le, self.config["encoder_weights"])
        return X, Y

    def make_test_dataset(self, df):
        X = df
        return X

    def data_split(self, X, Y):
        x_train, x_test, y_train, y_test = train_test_split(X, Y)
        return x_train, x_test, y_train, y_test
