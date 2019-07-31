import pandas as pd


def load_and_merge(raw_data_path, split_name, nrows):
    identity = pd.read_csv('{}/{}_identity.csv'.format(raw_data_path, split_name), nrows=nrows)
    transaction = pd.read_csv('{}/{}_transaction.csv'.format(raw_data_path, split_name), nrows=nrows)
    data = pd.merge(transaction, identity, on='TransactionID', how='left')
    return data
