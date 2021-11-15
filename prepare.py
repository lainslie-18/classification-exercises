import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

# import our own acquire module
import acquire


####################### iris_data ###############################

def clean_iris_data(df):
    # this function takes in the iris data set and cleans it
    
    # drops unnecessary columns
    df = df.drop(columns='species_id')
    # renames columns for easier reading
    df.rename(columns={'species_name':'species'},inplace=True)
    # Creates boolean dummy variables for species rather than using the string version of name
    dummy_df = pd.get_dummies(df['species'], dummy_na=False)
    # concatenates dummy variable rows onto original dataframe
    df = pd.concat([df, dummy_df], axis=1)
    return df

def split_iris_data(df):
    # this function takes in the iris data set and splits it into train, test, and validate datasets
    train, test = train_test_split(df, test_size = .2, stratify = df.species)
    train, validate = train_test_split(train, test_size = .3, stratify = train.species)
    return train, validate, test


def prep_iris_data(df):
    # this function takes in the iris data set and cleans and preps it for exploration

    df = clean_iris_data(df)
    train, validate, test = split_iris_data(df)
    return train, validate, test


####################### titanic_data ###############################

def clean__titanic_data(df):
    # this function takes in the titanic data set and cleans it
    
    # dropping duplicate columns
    df = df.drop_duplicates()
    # dropping unnecessary columns
    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    df = df.drop(columns=cols_to_drop)
    # filling nan values under embark_town column with most common value
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    # Creating boolean dummy variables for categorical columns rather than using string name
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False,drop_first=(True,True))
    # concatenating dummy variable rows onto original dataframe
    df = pd.concat([df, dummy_df], axis=1)
    return df


def split_titanic_data(df):
    # this function takes in a dataframe and splits it into train, test, and validate sets
    
    # splitting entire dataframe 80/20 into train and test sets
    train, test = train_test_split(df, test_size = .2, random_state=123, stratify = df.survived)
    # splitting off 30% of train set created in previous step to create a validate set
    train, validate = train_test_split(train, test_size = .3, random_state=123, stratify = train.survived)
    return train, validate, test


def impute_mode_titanic(train, validate, test):
    # this function takes in the titanic train, validate, and test data sets and imputes the nans in the embark_town column with the most frequent value
    
    # specifying the strategy to use
    imputer = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
    # fitting the imputer to the columns to determine most frequent value then calling transform to fill in empty values
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.fit_transform(validate[['embark_town']])
    test[['embark_town']] = imputer.fit_transform(test[['embark_town']])
    return train, validate, test


def prep_titanic_data(df):
    # this function takes in a dataframe and applies other functions to clean and split the data
    df = clean_titanic_data(df)
    train, validate, test = split_titanic_data(df)
    return train, validate, test


########################### telco_data #################################

def clean_telco_data(df):
    # this function takes in the telco data and cleans it
    
    # changing data type for total charges from string to float
    df.total_charges = pd.to_numeric(df.total_charges, errors='coerce')
    # dropping rows where new customers have not yet had opportunity to churn
    df = df[df.total_charges.notnull()]
    # dropping duplicates
    df = df.drop_duplicates()
    # dropping columns that are unnecessary or where info is duplicated in another column
    cols_to_drop = ['customer_id', 'payment_type_id', 'internet_service_type_id', 'contract_type_id', 'phone_service']
    df = df.drop(columns=cols_to_drop)
    # replacing information included in another column to simplify encoding (creates only two values)
    df.replace('No internet service', 'No', inplace=True)
    
    # creating df of dummy variables for columns with two values, dropping first
    dummy_df1 = pd.get_dummies(df[['gender', 'partner', 'dependents', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'churn']], drop_first=True)
    # creating df of dummy variables for columns with more than two values, keeping all columns for clarity
    dummy_df2 = pd.get_dummies(df[['contract_type', 'internet_service_type', 'payment_type']])
    # concatenating dummy variable dfs onto original dataframe
    df = pd.concat([df, dummy_df1, dummy_df2], axis=1)
    
    # renaming columns for clarity
    df.rename(columns={
                'gender_Male': 'is_male',
                'partner_Yes': 'with_partner',
                'dependents_Yes': 'with_dependents',
                'multiple_lines_No phone service': 'no_phone_service',
                'multiple_lines_Yes': 'multiple_lines',
                'online_security_Yes': 'online_security',
                'online_backup_Yes': 'online_backup',
                'device_protection_Yes': 'device_protection',
                'tech_support_Yes': 'tech_support',
                'streaming_tv_Yes': 'streaming_tv',
                'streaming_movies_Yes': 'streaming_movies',
                'paperless_billing_Yes': 'paperless_billing',
                'churn_Yes': 'has_churned',
                'contract_type_Month-to-month': 'month_to_month_contract',
                'contract_type_One year': 'one_year_contract',
                'contract_type_Two year': 'two_year_contract',
                'internet_service_type_DSL': 'dsl_internet',
                'internet_service_type_Fiber optic': 'fiber_optic_internet',
                'internet_service_type_None': 'no_internet_service',
                'payment_type_Bank transfer (automatic)': 'bank_transfer_payment_automatic',
                'payment_type_Credit card (automatic)': 'credit_card_payment_automatic',
                'payment_type_Electronic check': 'electronic_check_payment',
                'payment_type_Mailed check': 'mailed_check_payment'}, inplace=True)
    return df


def split_telco_data(df):
   # this function takes in telco df with churn as stratify variable and returns train, validate, and test dfs
    train, test = train_test_split(df, test_size = .2, stratify = df.churn)
    train, validate = train_test_split(train, test_size = .3, stratify = train.churn)
    return train, validate, test


def prep_telco_data(df):
    # this function takes in telco df and cleans and splits it to prepare for exploring
    df = clean_telco_data(df)
    train, validate, test = split_telco_data(df)
    return train, validate, test