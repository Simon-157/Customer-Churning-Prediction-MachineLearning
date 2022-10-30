from sklearn.model_selection import train_test_split
import pandas as pd

def train_validate_test_split(data, target, seed = 123):
    """
    It splits the data into train, validate and test sets, where the test set is 20% of the original
    data, the validate set is 30% of the remaining data, and the train set is the remaining 50% of the
    original data.

    :param data: The dataframe you want to split
    :param target: The column name of the target variable
    :param seed: The random seed to use when splitting the data, defaults to 123 (optional)
    :return: A tuple of three dataframes
    """
    train_validate, test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data[target])

    train, validate = train_test_split(train_validate, test_size=0.3, random_state=seed,stratify=train_validate[target])
    return train, validate, test

def process_unencoded_data(data):
    """
    It takes in a dataframe, drops duplicates, removes rows where tenure is 0, removes $ and , from
    TotalCharges, converts TotalCharges to float, strips whitespace from all object columns, and returns
    a train, validate, and test dataframe

    :param data: the dataframe to be split
    :return: A tuple of 4 dataframes
    """
    data.drop_duplicates(inplace = True)
    data = data[data.tenure > 0].copy()
    data.TotalCharges = data.TotalCharges.str.strip()
    data.TotalCharges = data.TotalCharges.str.replace('[$,]','', regex = True)
    data.TotalCharges = data.TotalCharges.astype(float)

    categorical_columns = data.select_dtypes('object').columns


    for column in categorical_columns:
        data[column] = data[column].str.strip()
    data.drop(columns = ['InternetService', 'Contract', 'PaymentMethod'], inplace = True)
    return train_validate_test_split(data, 'Churn')

def binary_to_Y_N(value):
    if value == 1:
        return 'Yes'
    elif value == 0:
        return 'No'

def process_clean_data(data):
    data.drop_duplicates(inplace = True)
    data = data[data.tenure > 0].copy()
    data.TotalCharges= data.TotalCharges.str.strip()
    data.TotalCharges= data.TotalCharges.str.replace('[$,]','', regex = True)
    data.TotalCharges= data.TotalCharges.astype(float)
    categorical_columns = data.select_dtypes('object').columns[1:]


    for col in categorical_columns:
        data[col] = data[col].str.strip()
    data.drop(columns = ['customerID','InternetService', 'Contract', 'PaymentMethod'], inplace = True)
    data.SeniorCitizen = data.SeniorCitizen.astype(object)
    data.SeniorCitizen = data.SeniorCitizen.apply(binary_to_Y_N)
    categorical_columns = data.select_dtypes('object').columns

    dummy_df = pd.get_dummies(data[categorical_columns], drop_first = True)
    data = pd.concat([data, dummy_df], axis = 1)
    data = data.drop(columns = categorical_columns)
    data.rename(columns = {'Churn_Yes':'Churn'}, inplace = True)

    return train_validate_test_split(data, 'Churn')
