import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from src.utils.helper import save_weights, load_weights

class BuildFeatures:
    def __init__(self, config):
        self.config = config

    def preprocess(self, df):
        return df

    def encode_data(self, df, categorical_features):
        for col in categorical_features:
            lbe = LabelEncoder()

            df[col] = lbe.fit_transform(df[col])
        return df

    def get_top_k_features(self, X, Y, k):
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, Y)
        feature_df = pd.DataFrame(
            data=(X.columns, clf.feature_importances_)
        ).T.sort_values(by=1, ascending=False)
        cols = feature_df.head(k)[0].values
        return cols

    def get_features_col(self, df):
        numerical_features = df.select_dtypes(include="number").columns.tolist()
        print(f"There are {len(numerical_features)} numerical features:", "\n")
        categorical_features = df.select_dtypes(exclude="number").columns.tolist()
        print(f"There are {len(categorical_features)} categorical features:", "\n")

        return numerical_features, categorical_features

    def train_data_preprocessing_pipeline(
        self, df, numerical_features, categorical_features
    ):

        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )
        full_processor = ColumnTransformer(
            transformers=[
                ("number", numeric_pipeline, numerical_features),
                ("category", categorical_pipeline, categorical_features),
            ]
        )
        df_transformed = full_processor.fit_transform(df)
        save_weights(full_processor, self.config["feature_pipeline"])
        return df_transformed

    def test_data_preprocessing_pipeline(self, df):
        processor = load_weights(self.config["feature_pipeline"])
        df_transformed = processor.transform(df)
        return df_transformed
