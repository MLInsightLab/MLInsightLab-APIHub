import argon2
import os

# Global variables for model flavors
ALLOWED_MODEL_FLAVORS = [
    'pyfunc',
    'sklearn',
    'transformers',
    'hfhub'
]
PYFUNC_FLAVOR = ALLOWED_MODEL_FLAVORS[0]
SKLEARN_FLAVOR = ALLOWED_MODEL_FLAVORS[1]
TRANSFORMERS_FLAVOR = ALLOWED_MODEL_FLAVORS[2]
HUGGINGFACE_FLAVOR = ALLOWED_MODEL_FLAVORS[3]

# Global variables for prediction functions
ALLOWED_PREDICT_FUNCTIONS = [
    'predict',
    'predict_proba'
]
PREDICT = ALLOWED_PREDICT_FUNCTIONS[0]
PREDICT_PROBA = ALLOWED_PREDICT_FUNCTIONS[1]

DATA_DIRECTORY = os.environ['DATA_DIRECTORY']

# Location to store predictions
PREDICTIONS_DIR = os.environ['PREDICTIONS_CACHE_DIR']

VARIABLE_STORE_DIRECTORY = os.environ['VARIABLE_STORE_DIRECTORY']
VARIABLE_STORE_FILE = os.path.join(
    VARIABLE_STORE_DIRECTORY, 'variable_store.json')

# Database location
DB_DIRECTORY = '/database'
DB_FILE = os.path.join(DB_DIRECTORY, 'permissions.db')

# Admin username, password, and key
ADMIN_USERNAME = os.environ['ADMIN_USERNAME']
ADMIN_PASSWORD = os.environ['ADMIN_PASSWORD']
ADMIN_KEY = os.environ['ADMIN_KEY']

# Hashed admin key and password
HASHED_ADMIN_KEY = argon2.PasswordHasher().hash(ADMIN_KEY)
HASHED_ADMIN_PASSWORD = argon2.PasswordHasher().hash(ADMIN_PASSWORD)

# Location to cache state of loaded models
SERVED_MODEL_CACHE_DIR = os.environ['SERVED_MODEL_CACHE_DIR']
SERVED_MODEL_CACHE_FILE = os.path.join(SERVED_MODEL_CACHE_DIR, 'models.json')

# Password requirements
MINIMUM_PASSWORD_LENGTH = 8
