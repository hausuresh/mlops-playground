import os
import warnings
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
MODEL_DIR = Path(os.getcwd())


mlflow.set_tracking_uri(("file://"+os.path.abspath(os.path.join(MODEL_DIR, 'mlruns'))).replace("\\","/"))
#mlflow.set_tracking_uri("sqlite:///mlruns.db")

# get args
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
data_month = int(sys.argv[3]) if len(sys.argv) > 3 else 5

# Define training data source
# fix window/linux sep
TRAIN_FILE_PATH = os.path.join(MODEL_DIR,'model','data','train-data','training_data_month_{}.zip'.format(data_month))

# Set global random state for stable training
np.random.seed(40)
rand = 40

def get_training_data(url):
    try:
        train_data = pd.read_csv(url)
    except Exception as e:
        logger.exception(
            "Unable to read data from {}. Error: {}".format(url,e)
        )
    return train_data 

def split_data(data, rand):
    '''
    Custom split to avoid data leakage
    Returns
    -------        
        Pandas dataframes: X_train, X_test, y_train, y_test
    '''
    
    df_train_avl = data[data['label']>0]
    df_train_notavl = data[data['label']==0]
    y_aval = df_train_avl.pop('label')
    X_aval = df_train_avl
    y_notaval = df_train_notavl.pop('label')
    X_notaval = df_train_notavl

    X_train_avl, X_test_avl, y_train_avl, y_test_avl = train_test_split(X_aval, y_aval, test_size=0.3, random_state=rand)
    X_train_notavl, X_test_notavl, y_train_notavl, y_test_notavl = train_test_split(X_notaval, y_notaval, test_size=0.3, random_state=rand)

    X_train = pd.concat([X_train_avl,X_train_notavl])
    y_train = pd.concat([y_train_avl,y_train_notavl])
    X_test = pd.concat([X_test_avl,X_test_notavl])
    y_test = pd.concat([y_test_avl,y_test_notavl])

    # Free resources
    del df_train_avl
    del df_train_notavl
    del y_aval
    del X_aval
    del y_notaval
    del X_notaval

    return X_train, X_test, y_train, y_test


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data = get_training_data(url=TRAIN_FILE_PATH)
    
    # Split data
    train_x, test_x,train_y,test_y = split_data(data=data,rand=rand)
    
    with mlflow.start_run(run_name='toy-model-1') as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=rand)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Toy model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("data_month", data_month)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # check info
        print("Run id: {}".format(run.info.run_id))

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticNet")
        else:
            mlflow.sklearn.log_model(lr, "model")
