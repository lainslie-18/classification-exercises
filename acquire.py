import pandas as pd
import numpy as np
import os
from env import host, user, password


def connect_to_db(db, user=user, host=host, password=password):
    # this function uses my env file info to connect to Codeup database
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

######################## Titanic Data ######################################

def query_titanic_data():
    # this function performs a SQL query and reads the data from the Codeup db to a df
    sql_query = 'select * from passengers'
    df = pd.read_sql(sql_query, connect_to_db('titanic_db'))
    return df


def get_titanic_data():
    # this function checks for and reads a local csv file, if it exists. if not, it writes data
    # to a csv file and returns a df

    if os.path.isfile('titanic_df.csv'):
        df = pd.read_csv('titanic_df.csv', index_col=0)
    else:
        df = query_titanic_data()
        df.to_csv('titanic_df.csv')
    return df


######################## Iris Data ######################################

def query_iris_data():
    # this function performs a SQL query and reads the data from the Codeup db to a df
    sql_query = '''
                select 
                    species_id,
                    species_name,
                    sepal_length,
                    sepal_width,
                    petal_length,
                    petal_width
                from measurements
                join species using(species_id)
                '''

    df = pd.read_sql(sql_query, connect_to_db('iris_db'))
    return df


def get_iris_data():
    # this function checks for and reads a local csv file, if it exists. if not, it writes data
    # to a csv file and returns a df

    if os.path.isfile('iris_df.csv'):
        df = pd.read_csv('iris_df.csv', index_col=0)
    else:
        df = query_iris_data()
        df.to_csv('iris_df.csv')
    return df


######################## Telco Data ######################################

def query_telco_data():
    # this function performs a SQL query and reads the data from the Codeup db to a df
    sql_query = '''
                select * from customers
                join contract_types using(contract_type_id)
                join internet_service_types using(internet_service_type_id)
                join payment_types using(payment_type_id)
                '''

    df = pd.read_sql(sql_query, connect_to_db('telco_churn'))
    return df


def get_telco_data():
    # this function checks for and reads a local csv file, if it exists. if not, it writes data
    # to a csv file and returns a df

    if os.path.isfile('telco.csv'):
        df = pd.read_csv('telco.csv', index_col=0)
    else:
        df = query_telco_data()
        df.to_csv('telco.csv')
    return df