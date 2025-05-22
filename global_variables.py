import argon2
import os

# Global variables for model flavors
ALLOWED_MODEL_FLAVORS = [
    'pyfunc',
    'sklearn',
    'transformers',
    'hfhub'
]

# Flavors broken out
PYFUNC_FLAVOR = ALLOWED_MODEL_FLAVORS[0]
SKLEARN_FLAVOR = ALLOWED_MODEL_FLAVORS[1]
TRANSFORMERS_FLAVOR = ALLOWED_MODEL_FLAVORS[2]
HUGGINGFACE_FLAVOR = ALLOWED_MODEL_FLAVORS[3]

# Global variables for prediction functions
ALLOWED_PREDICT_FUNCTIONS = [
    'predict',
    'predict_proba',
    'transform'
]
PREDICT = ALLOWED_PREDICT_FUNCTIONS[0]
PREDICT_PROBA = ALLOWED_PREDICT_FUNCTIONS[1]
TRANSFORM = ALLOWED_PREDICT_FUNCTIONS[2]

# Database credentials
POSTGRES_HOST = os.environ['POSTGRES_HOST']
POSTGRES_USER = os.environ['POSTGRES_USER']
POSTGRES_PASSWORD = os.environ['POSTGRES_PASSWORD']
POSTGRES_DB = os.environ['POSTGRES_DB']

DB_CONNECTION_STRING = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}'

# Admin username, password, and key
ADMIN_USERNAME = os.environ['ADMIN_USERNAME']
ADMIN_PASSWORD = os.environ['ADMIN_PASSWORD']
ADMIN_KEY = os.environ['ADMIN_KEY']

# Hashed admin key and password
HASHED_ADMIN_KEY = argon2.PasswordHasher().hash(ADMIN_KEY)
HASHED_ADMIN_PASSWORD = argon2.PasswordHasher().hash(ADMIN_PASSWORD)

# Password requirements
MINIMUM_PASSWORD_LENGTH = 8
