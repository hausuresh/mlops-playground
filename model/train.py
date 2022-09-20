
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RandomForestRegressor
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# define datasource
ORDER_DATA_URL = 'data\source-data\data_order.csv'
STORE_DATA_URL = 'data\source-data\data_metadata_store.csv'
PRODUCT_DATA_URL = 'data\source-data\data_metadata_product.csv'

# get args
max_depth = float(sys.argv[1]) if len(sys.argv) > 1 else 3
n_estimators = float(sys.argv[2]) if len(sys.argv) > 2 else 50
data_month = float(sys.argv[3]) if len(sys.argv) > 3 else 4

def get_order_data(order_data_url):
    df_order = pd.read_csv(order_data_url)
    df_order.columns= [x.lower() for x in list(df_order.columns)]
    # transform pickle or some thing....

def get_store_data(store_data_url):
    df_store = pd.read_csv(store_data_url)
    df_store.columns= [x.lower() for x in list(df_store.columns)]
    # transform pickle or some thing....

def get_product_data(product_data_url):
    df_product= pd.read_csv(product_data_url)
    df_product.columns= [x.lower() for x in list(df_product.columns)]
    # transform pickle or some thing....

def get_product_vocab(order_data, store_data):
    # get vocab here
    pass

def preprocess_data():
    # Merge & preprocess data here
    train_data_file_path = 'data\train-data\train_data_month_{}.csv'.format(data_month)
    if os.path(train_data_file_path):
        train_data = pd.read_csv(train_data_file_path)
        return train_data
    else:
        df_order = get_order_data(ORDER_DATA_URL)
        df_store = get_store_data(STORE_DATA_URL)
        df_product = get_product_data(PRODUCT_DATA_URL)
        df_matrix = get_product_vocab(df_order, df_store)
        # pickle file or join some thing here
        # return train_data

        # write train-data file
        pass

def split_data(data, rand):
    '''
        Custom split to avoid data leakage
        return: X_train, X_test, y_train, y_test
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

    return X_train, X_test, y_train, y_test

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    rand = 40

    data = preprocess_data()

    train_x , test_x ,train_y,test_y = split_data(data)

    with mlflow.start_run():
        lr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("RandomForestRegressor model (max_depth=%f, n_estimators=%f):" % (max_depth, n_estimators))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="RandomForestRegressorModel_month_{}".format(data_month))
            # mlflow.set_tracking_uri("http://localhost:5000")
        else:
            mlflow.sklearn.log_model(lr, "model")
            # mlflow.set_tracking_uri("http://localhost:5000")
