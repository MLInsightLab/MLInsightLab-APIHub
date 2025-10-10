from global_variables import ALLOWED_MODEL_FLAVORS, PYFUNC_FLAVOR, SKLEARN_FLAVOR, TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR, ALLOWED_PREDICT_FUNCTIONS
from templates import PredictRequest
from io import BytesIO
import datetime as dt
import numpy as np
import subprocess
import requests
import mlflow
import boto3
import json
import os

# S3 storage client
s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    endpoint_url=os.environ['S3_ENDPOINT_URL']
)

# Save prediction


def save_prediction(
    model_name: str,
    model_flavor: str,
    model_version_or_alias: str | int,
    body: PredictRequest,
    prediction: list,
    username: str
):
    prediction_file_path = os.path.join(
        model_name, model_flavor, str(model_version_or_alias), str(dt.datetime.now()))
    prediction_data = {
        'input': body.model_dump(),
        'prediction': prediction,
        'username': username
    }

    s3.put_object(
        Body=json.dumps(prediction_data),
        Bucket='predictions',
        Key=prediction_file_path
    )

    return True

# List models with predictions in the predictions directory


def list_models_with_predictions():
    all_objects = s3.list_objects(Bucket='predictions')['Contents']
    model_dirs = ['/'.join(obj['Key'].split('/')[:3]) for obj in all_objects]
    print(model_dirs)
    models = list(set(model_dirs))
    models = [model.split('/') for model in models]
    return models

# Get saved predictions for a specific model


def get_predictions(
        model_name: str,
        model_flavor: str,
        model_version_or_alias: str
):
    # Return all files for the specific model name, flavor, version combination
    all_objects = s3.list_objects(Bucket='predictions')['Contents']
    all_files = [f['Key'] for f in all_objects if f['Key'].startswith(
        f'{model_name}/{model_flavor}/{model_version_or_alias}')]

    # Return None if no feedback comes back
    if len(all_files) == 0:
        return None

    # Get the predictions back
    predictions = {}
    for filename in all_files:
        timestamp = filename.split('/')[-1]
        file_obj = BytesIO()
        s3.download_fileobj('predictions', filename, file_obj)
        file_obj.seek(0)
        content = json.load(file_obj)
        predictions[timestamp] = content

    return predictions

# Predict_model function that runs prediction


def predict_model(
    model: mlflow.models.Model | dict,
    to_predict: np.ndarray | list | str | dict,
    model_flavor: str,
    predict_function: str,
    params: dict,
    dtype: str = None
):
    f'''
    Make predictions with a model

    Parameters
    ----------
    model : mlflow.models.Model or dictionary
        The model to run prediction on
    to_predict : np.ndarray or array-like
        The data to predict on
    model_flavor : str
        The flavor of the model, must be one of {ALLOWED_MODEL_FLAVORS}
    predict_function : str
        The predict function to run, must be one of {ALLOWED_PREDICT_FUNCTIONS}
    params : dict
        Parameters to run prediction with
    '''

    if isinstance(model, dict):
        container_name = model['container_name']
        container_port = '8888'
        convert_to_numpy = False

        if isinstance(to_predict, np.ndarray):
            to_predict = to_predict.tolist()
            convert_to_numpy = True

        with requests.Session() as sess:
            resp = sess.post(
                f'http://{container_name}:{container_port}/predict',
                json={
                    'inputs': to_predict,
                    'predict_function': predict_function,
                    'convert_to_numpy': convert_to_numpy,
                    'params': params,
                    'dtype': dtype
                }
            )
        if not resp.ok:
            raise ValueError(
                f'There was an error running predict on the model: {resp.json()}')
        else:
            return resp.json()

    if predict_function == 'predict':
        try:
            if model_flavor == PYFUNC_FLAVOR:
                results = model.predict(to_predict, params=params)
            elif model_flavor in [TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR]:
                if params:
                    results = model(to_predict, **params)
                else:
                    results = model(to_predict)
            elif model_flavor == SKLEARN_FLAVOR:
                results = model.predict(to_predict)
        except Exception:
            try:
                if isinstance(to_predict, np.ndarray):
                    to_predict = to_predict.reshape(-1, 1)
                if model_flavor == PYFUNC_FLAVOR:
                    results = model.predict(to_predict, params=params)
                elif model_flavor in [TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR]:
                    if params:
                        results = model(to_predict, **params)
                    else:
                        results = model(to_predict)
                elif model_flavor == SKLEARN_FLAVOR:
                    results = model.predict(to_predict)
            except Exception as e:
                raise ValueError(
                    f'There was an issue running `predict`: {str(e)}')

    elif predict_function == 'predict_proba':
        try:
            results = model.predict_proba(to_predict)
        except Exception:
            try:
                results = model.predict_proba(to_predict.reshape(-1, 1))
            except Exception:
                raise ValueError('There was an issue running `predict_proba`')

    elif predict_function == 'transform':
        try:
            results = model.transform(to_predict)
        except Exception:
            try:
                results = model.transform(to_predict.reshape(-1, 1))
            except Exception:
                raise ValueError('There was an issue running `transform`')

    else:
        raise ValueError(
            'Only `predict`, `predict_proba`, and `transform` are supported predict functions')

    if isinstance(results, np.ndarray):
        results = results.tolist()

    return {
        'prediction': results
    }


# Function to load the model cache file


def load_models_from_cache():
    '''
    Load models from the cache directory
    '''

    try:
        file_obj = BytesIO()
        s3.download_fileobj('model-cache', 'models.json', file_obj)
        file_obj.seek(0)
        models = json.load(file_obj)
        return models

    except Exception as e:
        return None
