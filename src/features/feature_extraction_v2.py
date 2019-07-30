from collections import OrderedDict
import os

import neptune
from neptunecontrib.versioning.data import log_data_version
from neptunecontrib.api.utils import get_filepaths
from category_encoders import OrdinalEncoder
from pandarallel import pandarallel
import pandas as pd
import numpy as np

from src.features.const import ID_COLS, V1_COLS, V1_CAT_COLS
from src.utils import read_config, check_env_vars
from src.features.utils import load_and_merge

pandarallel.initialize()
check_env_vars()
CONFIG = read_config(config_path=os.getenv('CONFIG_PATH'))

neptune.init(project_qualified_name=CONFIG.project)

RAW_DATA_PATH = CONFIG.data.raw_data_path
FEATURES_DATA_PATH = CONFIG.data.features_data_path
FEATURE_NAME = 'v2'
NROWS = 1000


def _split_email(x, colname):
    if type(x) == float and np.isnan(x):
        email_first, email_rest = None, None
    else:
        split = x.split('.')
        email_first = split[0]
        email_rest = '.'.join(split[1:])
    return pd.Series(OrderedDict([('{}_first'.format(colname), email_first),
                                  ('{}_rest'.format(colname), email_rest)]))


def clean_email(df, email_cols):
    all_new_cols = []
    for col in email_cols:
        new_cols = ['{}_first'.format(col), '{}_rest'.format(col)]
        df[new_cols] = df[col].parallel_apply(lambda x: _split_email(x, colname=col))
        df.drop(col, inplace=True, axis=1)
        all_new_cols.extend(new_cols)
    return df, all_new_cols


def main():
    print('started experimnent')
    with neptune.create_experiment(name='feature engineering',
                                   tags=['feature-extraction', FEATURE_NAME],
                                   upload_source_files=get_filepaths(),
                                   properties={'feature_version': FEATURE_NAME}):
        print('loading data')
        train = load_and_merge(RAW_DATA_PATH, 'train', NROWS)[ID_COLS + V1_COLS + ['isFraud']]
        test = load_and_merge(RAW_DATA_PATH, 'test', NROWS)[ID_COLS + V1_COLS]

        categorical_cols = set(V1_CAT_COLS)
        print('cleaning data')
        email_cols = ['P_emaildomain', 'R_emaildomain']
        train, new_email_cols = clean_email(train, email_cols)
        test, _ = clean_email(test, email_cols)

        categorical_cols.update(new_email_cols)
        for col in email_cols:
            categorical_cols.remove(col)
        categorical_cols = list(categorical_cols)
        neptune.set_property('categorical_columns', str(categorical_cols))

        print('encoding categoricals')
        encoder = OrdinalEncoder(cols=categorical_cols).fit(train[ID_COLS + categorical_cols])
        train[ID_COLS + categorical_cols] = encoder.transform(train[ID_COLS + categorical_cols])
        test[ID_COLS + categorical_cols] = encoder.transform(test[ID_COLS + categorical_cols])

        train_features_path = os.path.join(FEATURES_DATA_PATH, 'train_features_{}.csv'.format(FEATURE_NAME))
        print('saving train to {}'.format(train_features_path))
        train.to_csv(train_features_path, index=None)
        log_data_version(train_features_path, prefix='train_features_')

        test_features_path = os.path.join(FEATURES_DATA_PATH, 'test_features_{}.csv'.format(FEATURE_NAME))
        print('saving test to {}'.format(test_features_path))
        test.to_csv(test_features_path, index=None)
        log_data_version(test_features_path, prefix='test_features_')


if __name__ == '__main__':
    main()
