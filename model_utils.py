from global_variables import PREDICTIONS_DIR, ALLOWED_MODEL_FLAVORS, PYFUNC_FLAVOR, SKLEARN_FLAVOR, TRANSFORMERS_FLAVOR, HUGGINGFACE_FLAVOR, ALLOWED_PREDICT_FUNCTIONS, SERVED_MODEL_CACHE_FILE
from transformers import pipeline, BitsAndBytesConfig
from templates import PredictRequest
from glob import glob
import datetime as dt
import numpy as np
import subprocess
import mlflow
import json
import os


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
        PREDICTIONS_DIR, model_name, model_flavor, model_version_or_alias, str(dt.datetime.now()))
    prediction_data = {
        'input': body.model_dump(),
        'prediction': prediction,
        'username': username
    }
    if not os.path.exists(os.path.dirname(prediction_file_path)):
        os.makedirs(os.path.dirname(prediction_file_path))

    with open(prediction_file_path, 'w') as f:
        json.dump(prediction_data, f)

    return True

# List models in the predictions directory


def list_models_with_predictions():
    model_dirs = glob(f'{PREDICTIONS_DIR}/*/*/*')
    return [d.split('/')[2:] for d in model_dirs]

# Get predictions


def get_predictions(
        model_name: str,
        model_flavor: str,
        model_version_or_alias: str
):
    prediction_directory = os.path.join(
        PREDICTIONS_DIR, model_name, model_flavor, model_version_or_alias)
    if not os.path.exists(prediction_directory):
        return None

    files = os.listdir(prediction_directory)

    predictions = {}
    for file in files:
        with open(os.path.join(prediction_directory, file), 'r') as f:
            predictions[file] = json.load(f)

    return predictions

# Predict_model function that runs prediction


def predict_model(
    model: mlflow.models.Model,
    to_predict: np.ndarray,
    model_flavor: str,
    predict_function: str,
    params: dict
):
    f"""
    Make predictions with a model

    Parameters
    ----------
    model : mlflow.models.Model
        The model to run prediction on
    to_predict : np.ndarray or array-like
        The data to predict on
    model_flavor : str
        The flavor of the model, must be one of {ALLOWED_MODEL_FLAVORS}
    predict_function : str
        The predict function to run, must be one of {ALLOWED_PREDICT_FUNCTIONS}
    params : dict
        Parameters to run prediction with
    """
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

    else:
        raise ValueError(
            'Only `predict` and `predict_proba` are supported predict functions')

    if isinstance(results, np.ndarray):
        results = results.tolist()

    return {
        'prediction': results
    }

# Load_model function that allows to load model from either alias or version


def fload_model(
    model_name: str,
    model_flavor: str,
    model_version: str | int | None = None,
    model_alias: str | None = None,
    requirements: str | None = None,
    quantization_kwargs: dict | None = None,
    **kwargs
):
    f"""
    Load a model from the MLFlow server

    Parameters
    ----------
    model_name : str
        The name of the model
    model_flavor : str
        The flavor of the model, must be one of {ALLOWED_MODEL_FLAVORS}
    model_version : int or None (default None)
        The version of the model
    model_alias : str or None (default None)
        The alias of the model, without the `@` character
    requirements : str or None (default None)
        Any pip requirements for loading the model
    quantization_kwargs : dict or None (default None)
        Quantization keyword arguments. NOTE: Only applies for hfhub models
    **kwargs : additional keyword arguments
        Additional keyword arguments. NOTE: Only applies to hfhub models

    Notes
    -----
    - One of either `model_version` or `model_alias` must be provided

    Returns
    -------
    model : mlflow Model
        The model, in the flavor specified

    Raises
    ------
    - MlflowException, when the model cannot be loaded
    """

    if not (model_version or model_alias) and model_flavor != HUGGINGFACE_FLAVOR:
        raise ValueError('Model version or model alias must be provided')

    if model_flavor not in ALLOWED_MODEL_FLAVORS:
        raise ValueError(
            f'Only "pyfunc", "sklearn", "transformers", and "hfhub" model flavors supported, got {model_flavor}')

    try:

        # If the model is not a huggingface model, then format the model uri
        if model_flavor != HUGGINGFACE_FLAVOR:
            if model_version:
                model_uri = f'models:/{model_name}/{model_version}'
            elif model_alias:
                model_uri = f'models:/{model_name}@{model_alias}'

            # Install dependencies for the model from mlflow
            subprocess.run(
                [
                    'pip',
                    'install',
                    '-r',
                    mlflow.pyfunc.get_model_dependencies(model_uri)
                ]
            )

        # Install requirements for the model if it's a huggingface model
        else:
            if requirements:
                subprocess.run(
                    [
                        'pip',
                        'install',
                        requirements
                    ]
                )

        # Load the model if it is requested to be a pyfunc model
        if model_flavor == PYFUNC_FLAVOR:
            model = mlflow.pyfunc.load_model(model_uri)

        # Load the model if it is requested to be a sklearn model
        elif model_flavor == SKLEARN_FLAVOR:
            model = mlflow.sklearn.load_model(model_uri)

        # Load the model if it is requested to be a transformers model
        elif model_flavor == TRANSFORMERS_FLAVOR:
            if mlflow.transformers.is_gpu_available():
                # NOTE: This loads the model to GPU automatically
                # TODO: Change this so that it can be done more intelligently
                model = mlflow.transformers.load_model(
                    model_uri,
                    kwargs={
                        'device_map': 'auto'
                    }
                )
            else:
                model = mlflow.transformers.load_model(model_uri)

        # Load the model if it is a huggingface model
        elif model_flavor == HUGGINGFACE_FLAVOR:
            if quantization_kwargs:
                bnb_config = BitsAndBytesConfig(**quantization_kwargs)
                if not kwargs.get('model_kwargs'):
                    kwargs['model_kwargs'] = {}
                kwargs['model_kwargs']['quantization_config'] = bnb_config

            model = pipeline(**kwargs)

        return model

    except Exception:
        raise mlflow.MlflowException('Could not load model')


# Function to load models from cache


def load_models_from_cache():
    """
    Load models from the cache directory
    """
    try:
        with open(SERVED_MODEL_CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return None
