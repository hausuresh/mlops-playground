import yaml
import os
import logging
import requests
import numpy as np
import pandas as pd

from mlflow import pyfunc
from seldon_core import Storage
from seldon_core.user_model import SeldonComponent, SeldonNotImplementedError
from typing import Dict, List, Union

logger = logging.getLogger()

MLFLOW_SERVER = "model"
DATA_SOURCE = "https://github.com/hausuresh/pf-model/raw/main/model/data/train-data/vocab_data_month_5.zip"
MODEL_URI = 'gs://pod-seldon-model/pod-toy-model/model'

class MLFlowServerCustom(SeldonComponent):
    def __init__(self, model_uri: str, xtype: str = "ndarray"):
        super().__init__()
        logger.info(f"Creating MLFLow server with URI {model_uri}")
        logger.info(f"xtype: {xtype}")
        self.model_uri = model_uri
        self.xtype = xtype
        self.ready = False
        self.column_names = None

    def load(self):
        logger.info(f"Downloading model from {self.model_uri}")
        model_folder = Storage.download(self.model_uri)
        self._model = pyfunc.load_model(model_folder)
        logger.info(f"Get data from {DATA_SOURCE}")
        #Cache data source
        self.df = pd.read_csv(DATA_SOURCE)
        self.ready = True

    def predict(
        self, X: np.ndarray, feature_names: List[str] = [], meta: Dict = None
    ) -> Union[np.ndarray, List, Dict, str, bytes]:
        logger.debug(f"Requesting prediction with: {X}")

        if not self.ready:
            raise requests.HTTPError("Model not loaded yet")
        else: 
            X = self.pre_process(X)
            print(X)

        if self.xtype == "ndarray":
            result = self._model.predict(X)
        else:
            if feature_names is not None and len(feature_names) > 0:
                df = pd.DataFrame(data=X, columns=feature_names)
            else:
                df = pd.DataFrame(data=X)
            result = self._model.predict(df)

        if isinstance(result, pd.DataFrame):
            if self.column_names is None:
                self.column_names = result.columns.to_list()
            result = result.to_numpy()

        logger.debug(f"Prediction result: {result}")
        return result

    def init_metadata(self):
        file_path = os.path.join(self.model_uri, "metadata.yaml")

        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f.read())
        except FileNotFoundError:
            logger.debug(f"metadata file {file_path} does not exist")
            return {}
        except yaml.YAMLError:
            logger.error(
                f"metadata file {file_path} present but does not contain valid yaml"
            )
            return {}

    def class_names(self):
        if self.column_names is not None:
            return self.column_names

        raise SeldonNotImplementedError("prediction result is not a dataframe")

    def pre_process(self, X: np.ndarray):
        df = self.df
        # Process dataframe
        X = pd.DataFrame(X,columns=['store_id','product_id'])
        # Get store_id & product_id metadata from vocab
        X = X.merge(df, how = 'left', on = ['store_id','product_id'] ).fillna(0).drop_duplicates()
        X = X.drop(['store_id','product_id'],axis=1)

        logger.debug(np.array(X, dtype=float))

        return np.array(X, dtype=float)