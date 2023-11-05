"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
        'tGravityAcc-energy()-X', 
        'tGravityAcc-mean()-X',
       'angle(X,gravityMean)', 
        'tGravityAcc-min()-X',
       'tGravityAcc-min()-Y', 
        'tGravityAcc-max()-Y',
       'tGravityAcc-max()-X', 
        'tGravityAcc-mean()-Y',
       'angle(Y,gravityMean)',
        'tBodyAccJerk-entropy()-X',
       'tBodyAcc-max()-X',
      'fBodyAccMag-mad()'
]
label_column = "Activity"

feature_columns_dtype = {
    "tGravityAcc-energy()-X": np.float64,
    "tGravityAcc-mean()-X": np.float64,
    "angle(X,gravityMean)": np.float64,
    "tGravityAcc-min()-X": np.float64,
    "tGravityAcc-min()-Y": np.float64,
    "tGravityAcc-max()-Y": np.float64,
    "tGravityAcc-max()-X": np.float64,
    "tGravityAcc-mean()-Y": np.float64,
    "angle(Y,gravityMean)": np.float64,
    "tBodyAccJerk-entropy()-X": np.float64,
    "tBodyAcc-max()-X": np.float64,
    "fBodyAccMag-mad()": np.float64,
    
}
label_column_dtype = {"Activity": 'str'}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/sample_train.csv"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    #logger.debug("Reading downloaded data.")
    logger.info("Reading downloaded data.")
    #df = pd.read_csv(
    #    fn,
    #    header=None,
    #    names=feature_columns_names + [label_column],
    #    dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    #)
    df = pd.read_csv(fn)
    os.unlink(fn)

    logger.debug("Defining transformers.")

    logger.info("Applying transforms.")
    df.dropna(inplace = True)
    
    cat_enc = {'WALKING_DOWNSTAIRS': 0, 'LAYING':1, 'WALKING_UPSTAIRS':2, 'SITTING':3,
       'STANDING':4, 'WALKING':5}
    df['Activity'] = df['Activity'].map(cat_enc)
    
    y = df["Activity"]
    X_pre= df.drop('Activity',axis = 1)
    
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
