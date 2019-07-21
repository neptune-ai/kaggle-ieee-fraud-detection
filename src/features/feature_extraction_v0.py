import os

import neptune
from neptunecontrib.versioning.data import log_data_version
from neptunecontrib.api.utils import get_filepaths
import pandas as pd

from const import V0_CAT_COLS
from src.utils import read_config

CONFIG = read_config(config_path=os.getenv('CONFIG_PATH'))

neptune.init(project_qualified_name=CONFIG.project)

RAW_DATA_PATH = CONFIG.data.raw_data_path
FEATURES_DATA_PATH = CONFIG.data.features_data_path
FEATURE_NAME = 'v0'
NROWS = None


def load_and_merge(raw_data_path, split_name, nrows):
    identity = pd.read_csv('{}/{}_identity.csv'.format(raw_data_path, split_name), nrows=nrows)
    transaction = pd.read_csv('{}/{}_transaction.csv'.format(raw_data_path, split_name), nrows=nrows)
    data = pd.merge(transaction, identity, on='TransactionID', how='left')
    return data


def feature_engineering_v0(df):
    df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])[
        'TransactionAmt'].transform('mean')
    df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])[
        'TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / df.groupby(['card1'])[
        'TransactionAmt'].transform('std')
    df['TransactionAmt_to_std_card4'] = df['TransactionAmt'] / df.groupby(['card4'])[
        'TransactionAmt'].transform('std')

    df['id_02_to_mean_card1'] = df['id_02'] / df.groupby(['card1'])['id_02'].transform('mean')
    df['id_02_to_mean_card4'] = df['id_02'] / df.groupby(['card4'])['id_02'].transform('mean')
    df['id_02_to_std_card1'] = df['id_02'] / df.groupby(['card1'])['id_02'].transform('std')
    df['id_02_to_std_card4'] = df['id_02'] / df.groupby(['card4'])['id_02'].transform('std')

    df['D15_to_mean_card1'] = df['D15'] / df.groupby(['card1'])['D15'].transform('mean')
    df['D15_to_mean_card4'] = df['D15'] / df.groupby(['card4'])['D15'].transform('mean')
    df['D15_to_std_card1'] = df['D15'] / df.groupby(['card1'])['D15'].transform('std')
    df['D15_to_std_card4'] = df['D15'] / df.groupby(['card4'])['D15'].transform('std')

    df['dist1_to_mean_card1'] = df['dist1'] / df.groupby(['card1'])['dist1'].transform('mean')
    df['dist1_to_mean_card4'] = df['dist1'] / df.groupby(['card4'])['dist1'].transform('mean')
    df['dist1_to_std_card1'] = df['dist1'] / df.groupby(['card1'])['dist1'].transform('std')
    df['dist1_to_std_card4'] = df['dist1'] / df.groupby(['card4'])['dist1'].transform('std')

    df['D4_to_mean_card1'] = df['D4'] / df.groupby(['card1'])['D4'].transform('mean')
    df['D4_to_mean_card4'] = df['D4'] / df.groupby(['card4'])['D4'].transform('mean')
    df['D4_to_std_card1'] = df['D4'] / df.groupby(['card1'])['D4'].transform('std')
    df['D4_to_std_card4'] = df['D4'] / df.groupby(['card4'])['D4'].transform('std')

    df['card1_count'] = df.groupby(['card1'])['TransactionID'].transform('count')
    df['card2_count'] = df.groupby(['card2'])['TransactionID'].transform('count')
    df['card4_count'] = df.groupby(['card4'])['TransactionID'].transform('count')

    return df


def get_cols_to_drop(df):
    many_null_cols = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]
    big_top_value_cols = [col for col in df.columns if
                          df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
    cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))

    if 'isFraud' in cols_to_drop:
        cols_to_drop.remove('isFraud')

    return cols_to_drop


def drop_existing_cols(df, cols):
    existing_cols = [col for col in cols if col in df.columns]
    return df.drop(existing_cols, axis=1)


def main():
    print('started experimnent')
    with neptune.create_experiment(name='feature engineering',
                                   tags=['feature-extraction', FEATURE_NAME],
                                   upload_source_files=get_filepaths(),
                                   properties={'feature_version': FEATURE_NAME}):
        cols_to_drop = V0_CAT_COLS
        for split_name in ['train', 'test']:
            print('processing {}'.format(split_name))
            data = load_and_merge(RAW_DATA_PATH, split_name, NROWS)
            features = feature_engineering_v0(data)
            cols_to_drop.extend(get_cols_to_drop(features))
            features = drop_existing_cols(features, cols_to_drop)
            features_path = os.path.join(FEATURES_DATA_PATH, '{}_features_{}.csv'.format(split_name, FEATURE_NAME))
            features.to_csv(features_path, index=None)
            log_data_version(features_path, prefix='{}_features_'.format(split_name))


if __name__ == '__main__':
    main()
