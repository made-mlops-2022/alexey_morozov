from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import dataclasses
import pickle
import logging


from source.params.model_params import ModelParams

model_type = Union[LogisticRegression, RandomForestClassifier]

def get_model(model_params: ModelParams) -> model_type:
    if model_params.model == 'lr':
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()
    # fields = dataclasses.fields(model_params.params)

    model.set_params(**dict(model_params.params.__dict__.items()))

    return model

def load_model(model_path: str) -> model_type:
    logging.info('Moadel loading: ' + model_path)
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def save_model(model: model_type, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info('Model saved: ' + model_path)

# def predict(model: model_type, preprocessed_x):
#     logging.info()
#     return model.predict(preprocessed_x)