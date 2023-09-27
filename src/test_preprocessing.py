
import pandas as pd
import numpy as np
import argparse
import os
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
def _parse_args():

    parser = argparse.ArgumentParser()
    
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/test/')
    parser.add_argument('--filename', type=str, default='test_data.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')

    return parser.parse_known_args()


def get_train_features(X):
    feature_path = "s3://sagemaker-studio-009676737623-l4vs7j0o0ib/mlops-level1-data/feature/feature.joblib"
    final_cols = load(feature_path)
    return final_cols

if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    # Load data
    df = pd.read_csv(os.path.join(args.filepath,args.filename))
    #df.drop(['date_time'],axis =1 ,inplace = True)
    ##
    
    X = df.drop(['Activity'], axis =1)
    Y = df['Activity']
    k =12
    final_cols = get_train_features(X)
    test_data = test_data[final_cols]

    test_data[['Activity']].to_csv(os.path.join(args.outputpath, 'test/test_y.csv'), index=False, header=False)
    test_data.drop(['Activity'], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_x.csv'), index=False, header=False)
    
    ## Save Features columns
    print("## Processing complete for Test Data, s Exiting.")
