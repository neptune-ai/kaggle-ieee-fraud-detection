import os

import neptune
from neptunecontrib.versioning.data import log_data_version
from neptunecontrib.api.utils import get_filepaths
from category_encoders import OrdinalEncoder

from src.features.const import ID_COLS, V1_COLS, V1_CAT_COLS
from src.utils import read_config, check_env_vars
from src.features.utils import load_and_merge

check_env_vars()
CONFIG = read_config(config_path=os.getenv('CONFIG_PATH'))

neptune.init(project_qualified_name=CONFIG.project)

RAW_DATA_PATH = CONFIG.data.raw_data_path
FEATURES_DATA_PATH = CONFIG.data.features_data_path
FEATURE_NAME = 'v1'
NROWS = None


def main():
    print('started experimnent')
    with neptune.create_experiment(name='feature engineering',
                                   tags=['feature-extraction', FEATURE_NAME],
                                   upload_source_files=get_filepaths(),
                                   properties={'feature_version': FEATURE_NAME}):
        print('loading data')
        train = load_and_merge(RAW_DATA_PATH, 'train', NROWS)[ID_COLS + V1_COLS + ['isFraud']]
        test = load_and_merge(RAW_DATA_PATH, 'test', NROWS)[ID_COLS + V1_COLS]

        print('encoding categoricals')
        encoder = OrdinalEncoder(cols=V1_CAT_COLS).fit(train[ID_COLS + V1_COLS])
        train[ID_COLS + V1_COLS] = encoder.transform(train[ID_COLS + V1_COLS])
        test = encoder.transform(test)

        print('saving data')
        train_features_path = os.path.join(FEATURES_DATA_PATH, 'train_features_{}.csv'.format(FEATURE_NAME))
        train.to_csv(train_features_path, index=None)
        log_data_version(train_features_path, prefix='train_features_')

        test_features_path = os.path.join(FEATURES_DATA_PATH, 'test_features_{}.csv'.format(FEATURE_NAME))
        test.to_csv(test_features_path, index=None)
        log_data_version(test_features_path, prefix='test_features_')


if __name__ == '__main__':
    main()
