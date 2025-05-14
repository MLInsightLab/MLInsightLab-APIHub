from global_variables import ADMIN_USERNAME, HASHED_ADMIN_KEY, HASHED_ADMIN_PASSWORD, DB_CONNECTION_STRING
import psycopg2
import argon2
import string
import random
import os

# Function to generate an API key


def generate_api_key():
    '''
    Generate an API key
    '''
    key = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    return f'mlil-{key}'

# Function to verify if a password meets minimum requirements


def password_meets_requirements(password):
    return all(
        [
            any([letter in password for letter in string.ascii_lowercase]),
            any([letter in password for letter in string.ascii_uppercase]),
            any([number in password for number in string.digits])
        ]
    )

# Function to generate a password


def generate_password():
    '''
    Generates a password
    '''
    password = ''
    while not password_meets_requirements(password):
        password = ''.join(random.choices(
            string.ascii_letters + string.digits, k=12))
    return password

# Function to validate role


def validate_role(role):
    if role not in ['admin', 'data_scientist', 'user']:
        raise ValueError('Not a valid role')
    return True

# Set up the database


def setup_database():
    '''
    Set up the database if it doesn't already exist

    NOTE: Can be run safely even if the database has already been created
    '''

    # Create table
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS users(username TEXT UNIQUE, role TEXT, apikey TEXT, password TEXT);'
    )
    con.commit()
    cursor.close()
    con.close()

    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"INSERT INTO users (username, role, apikey, password) VALUES ('{ADMIN_USERNAME}', 'admin', '{HASHED_ADMIN_KEY}', '{HASHED_ADMIN_PASSWORD}') ON CONFLICT (username) DO NOTHING;"
    )
    con.commit()
    cursor.close()
    con.close()

    return True

# Validate user's key


def validate_user_key(username, key):
    '''
    Validate a username, key combination

    If successful, returns the user's role

    If unsuccessful, raises an appropriate Exception
    '''

    # Execute the query on the database
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"SELECT * FROM users WHERE username='{username}';"
    )
    res = cursor.fetchall()
    con.close()

    # If there is not record for the user in the database, then the user does not exist -> raise ValueError
    if len(res) == 0:
        raise ValueError('User does not exist')

    # If there is more than one record for the user in the database, then there are duplicate usernames -> raise ValueError
    if len(res) > 1:
        raise ValueError('Multiple user records exist')

    # Expand the username, role, and hashed key
    username, role, hashed_key, hashed_password = res[0]

    # Return the role of the user if the key is validated
    try:
        argon2.PasswordHasher().verify(hashed_key, key)
        return role
    except Exception:
        raise ValueError('Incorrect Key Provided')

# Validate user password


def validate_user_password(username, password):
    '''
    Validate a username, password combination

    If successful, returns the user's role

    If unsuccessful, raises an appropriate Exception
    '''

    # Execute the query on the database
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"SELECT * FROM users WHERE username='{username}';"
    )
    res = cursor.fetchall()
    con.close()

    # If there is not record for the user in the database, then the user does not exist -> raise ValueError
    if len(res) == 0:
        raise ValueError('User does not exist')

    # If there is more than one record for the user in the database, then there are duplicate usernames -> raise ValueError
    if len(res) > 1:
        raise ValueError('Multiple user records exist')

    # Expand the username, role, and hashed key
    username, role, hashed_key, hashed_password = res[0]

    # Return the role of the user if the key is validated
    try:
        argon2.PasswordHasher().verify(hashed_password, password)
        return role
    except Exception:
        raise ValueError('Incorrect Password Provided')

# Create new user


def fcreate_user(username, role, api_key=None, password=None):
    '''
    Create a new user with an assigned role and (optionally) with an API key and password

    If successful, returns the user's API key

    NOTE: If user with the specified username already exists, raises ValueError
    '''

    # Establish connection to the database and check for the username already existing
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"SELECT * FROM users WHERE username='{username}';"
    )
    res = cursor.fetchall()
    con.close()

    if len(res) > 0:
        raise ValueError('Username already exists')

    # If the API key is not already provided, generate the API key
    if api_key is None:
        api_key = generate_api_key()

    # If the password is not already provided, generate the password
    if password is None:
        password = generate_password()

    if not password_meets_requirements(password):
        raise ValueError('Password does not meet minimum requirements')

    # Validate the prospective role
    validate_role(role)

    # Hash the API key and password
    hashed_api_key = argon2.PasswordHasher().hash(api_key)
    hashed_password = argon2.PasswordHasher().hash(password)

    # Insert new user into the database
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"INSERT INTO users (username, role, apikey, password) VALUES ('{username}', '{role}', '{hashed_api_key}', '{hashed_password}');"
    )
    con.commit()
    cursor.close()
    con.close()

    return api_key, password

# Delete a user


def fdelete_user(username):
    '''
    Delete a user from the database
    '''

    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"DELETE FROM users WHERE username='{username}';"
    )
    con.commit()
    cursor.close()
    con.close()

    return True

# Issue a new API key for a user


def fissue_new_api_key(username, key=None):
    '''
    Issue a new API key for a specified user

    NOTE: Raises ValueError if zero or more than one user exists with the username
    '''

    # Connect to the database and ensure that the user already exists
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"SELECT * FROM users WHERE username='{username}';"
    )
    res = cursor.fetchall()
    con.close()

    # Validate that only one user with that username exists
    if len(res) == 0:
        raise ValueError('User does not exist')
    elif len(res) > 1:
        raise ValueError('More than one user with that username exists')

    # Generate API key if one is not provided
    if key is None:
        key = generate_api_key()

    # Hash the key
    hashed_key = argon2.PasswordHasher().hash(key)

    # Update user in the database
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"UPDATE users SET apikey='{hashed_key}' WHERE username='{username}';"
    )
    con.commit()
    cursor.close()
    con.close()

    # Return the new API key
    return key

# Issue a new password for a user


def fissue_new_password(username, password=None):
    '''
    Issue a new password for a specified user

    NOTE: Raises ValueError if zero or more than one user exists with the username
    NOTE: Raises ValueError if password does not meet minimum length requirements or does not contain at least one uppercase and one lowercase letter
    '''

    # Connect to the database and ensure that the user already exists
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"SELECT * FROM users WHERE username='{username}';"
    )
    res = cursor.fetchall()
    con.close()

    # Validate that only one user with that username exists
    if len(res) == 0:
        raise ValueError('User does not exist')
    elif len(res) > 1:
        raise ValueError('More than one user with that username exists')

    # Generate API key if one is not provided
    if password is None:
        password = generate_password()

    if not password_meets_requirements(password):
        raise ValueError('Password does not meet minimum requirements')

    # Hash the key
    hashed_password = argon2.PasswordHasher().hash(password)

    # Update user in the database
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"UPDATE users SET password='{hashed_password}' WHERE username='{username}';"
    )
    con.commit()
    cursor.close()
    con.close()

    # Return the new password
    return password

# Get a user's role


def fget_user_role(username):
    '''
    Get a user's role
    '''

    # Connect to the database and ensure the user already exists
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"SELECT * FROM users WHERE username='{username}';"
    )
    res = cursor.fetchall()
    con.close()

    # Validate thatonly oneuser with that username exists
    if len(res) == 0:
        raise ValueError('User does not exist')
    elif len(res) > 1:
        raise ValueError('More than one user with that username exists')

    username, role, hashed_key, hashed_password = res[0]

    return role

# Update a user's role


def fupdate_user_role(username, new_role):
    '''
    Change a user's role
    '''

    # Connect to the database and ensure that the user already exists
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"SELECT * FROM users WHERE username='{username}';"
    )
    res = cursor.fetchall()
    con.close()

    # Validate that only one user with that username exists
    if len(res) == 0:
        raise ValueError('User does not exist')
    elif len(res) > 1:
        raise ValueError('More than one user with that username exists')

    # Validate the new role
    validate_role(new_role)

    # Update user role in the database
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute(
        f"UPDATE users SET role='{new_role}' WHERE username='{username}';"
    )
    con.commit()
    cursor.close()
    con.close()

    return new_role

# List all users


def flist_users():
    '''
    List all of the users in the database
    '''

    # Connect to the database
    con = psycopg2.connect(DB_CONNECTION_STRING)
    cursor = con.cursor()
    cursor.execute('SELECT username, role FROM users;')
    res = cursor.fetchall()
    con.close()
    return res
