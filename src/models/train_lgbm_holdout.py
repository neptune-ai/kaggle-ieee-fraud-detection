import os

import lightgbm as lgb
import neptune
from neptunecontrib.monitoring.lightgbm import neptune_monitor
from neptunecontrib.versioning.data import log_data_version
from neptunecontrib.api.utils import get_filepaths
from neptunecontrib.monitoring.reporting import send_binary_classification_report
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils import read_config, check_env_vars
from src.models.utils import sample_negative_class

check_env_vars()
CONFIG = read_config(config_path=os.getenv('CONFIG_PATH'))

neptune.init(project_qualified_name=CONFIG.project)

FEATURES_DATA_PATH = CONFIG.data.features_data_path
PREDICTION_DATA_PATH = CONFIG.data.prediction_data_path
SAMPLE_SUBMISSION_PATH = CONFIG.data.sample_submission_path
FEATURE_NAME = 'v0'
MODEL_NAME = 'lgbm'
NROWS = 50000
SEED = 1234

VALIDATION_PARAMS = {'validation_schema': 'holdout',
                     'validation_fraction': 0.26}

MODEL_PARAMS = {'num_leaves': 256,
                'min_child_samples': 79,
                'objective': 'binary',
                'max_depth': 15,
                'learning_rate': 0.02,
                "boosting_type": "gbdt",
                "subsample_freq": 3,
                "subsample": 0.9,
                "bagging_seed": 11,
                "metric": 'auc',
                "verbosity": -1,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'colsample_bytree': 0.9,
                'seed': 1234
                }

TRAINING_PARAMS = {'nrows': NROWS,
                   'negative_sample_fraction': 0.1,
                   'negative_sample_seed': SEED,
                   'num_boosting_rounds': 5000,
                   'early_stopping_rounds': 200
                   }


def fit_predict(train, valid, test, model_params, training_params, fine_tuning=True):
    X_train = train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_train = train['isFraud']

    X_valid = valid.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y_valid = valid['isFraud']

    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_valid, y_valid)

    if fine_tuning:
        callbacks = None
    else:
        callbacks = [neptune_monitor()]
    clf = lgb.train(model_params, trn_data,
                    training_params['num_boosting_rounds'],
                    valid_sets=[trn_data, val_data],
                    early_stopping_rounds=training_params['early_stopping_rounds'],
                    callbacks=callbacks)
    valid_preds = clf.predict(X_valid, num_iteration=clf.best_iteration)

    if fine_tuning:
        return valid_preds
    else:
        train_preds = clf.predict(X_train, num_iteration=clf.best_iteration)
        X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)
        test_preds = clf.predict(X_test, num_iteration=clf.best_iteration)
        return train_preds, valid_preds, test_preds


def fmt_preds(y_pred):
    return np.concatenate((1.0 - y_pred.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)


def main():
    print('loading data')
    train_features_path = os.path.join(FEATURES_DATA_PATH, 'train_features_' + FEATURE_NAME + '.csv')
    test_features_path = os.path.join(FEATURES_DATA_PATH, 'test_features_' + FEATURE_NAME + '.csv')

    print('... train')
    train = pd.read_csv(train_features_path, nrows=TRAINING_PARAMS['nrows'])

    print('... test')
    test = pd.read_csv(test_features_path, nrows=TRAINING_PARAMS['nrows'])

    idx_split = int((1 - VALIDATION_PARAMS['validation_fraction']) * len(train))
    train, valid = train[:idx_split], train[idx_split:]

    train = sample_negative_class(train,
                                  fraction=TRAINING_PARAMS['negative_sample_fraction'],
                                  seed=TRAINING_PARAMS['negative_sample_seed'])

    hyperparams = {**MODEL_PARAMS, **TRAINING_PARAMS, **VALIDATION_PARAMS}

    print('starting experiment')
    with neptune.create_experiment(name='model training',
                                   params=hyperparams,
                                   upload_source_files=get_filepaths(),
                                   tags=[MODEL_NAME, 'features_'.format(FEATURE_NAME), 'training']):
        print('logging data version')
        log_data_version(train_features_path, prefix='train_features_')
        log_data_version(test_features_path, prefix='test_features_')

        print('training')
        train_preds, valid_preds, test_preds = fit_predict(train, valid, test, MODEL_PARAMS, TRAINING_PARAMS)

        print('logging metrics')
        train_auc = roc_auc_score(train['isFraud'], train_preds)
        valid_auc = roc_auc_score(valid['isFraud'], valid_preds)
        neptune.send_metric('train_auc', train_auc)
        neptune.send_metric('valid_auc', valid_auc)
        send_binary_classification_report(valid['isFraud'], fmt_preds(valid_preds),
                                          channel_name='valid_classification_report')

        print('postprocessing predictions')
        valid_predictions_path = os.path.join(PREDICTION_DATA_PATH,
                                              'valid_prediction_{}_{}.csv'.format(FEATURE_NAME, MODEL_NAME))
        test_predictions_path = os.path.join(PREDICTION_DATA_PATH,
                                             'test_prediction_{}_{}.csv'.format(FEATURE_NAME, MODEL_NAME))
        submission_path = os.path.join(PREDICTION_DATA_PATH,
                                       'submission_{}_{}.csv'.format(FEATURE_NAME, MODEL_NAME))
        submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

        valid = pd.concat([valid, pd.DataFrame(valid[["TransactionDT", 'TransactionID']], columns=['prediction'])],
                          axis=1)
        test = pd.concat([test[["TransactionDT", 'TransactionID']], pd.DataFrame(test_preds, columns=['prediction'])],
                         axis=1)
        submission['isFraud'] = pd.merge(submission, test, on='TransactionID')['prediction']
        valid.to_csv(valid_predictions_path, index=None)
        test.to_csv(test_predictions_path, index=None)
        submission.to_csv(submission_path, index=None)
        neptune.send_artifact(valid_predictions_path)
        neptune.send_artifact(test_predictions_path)
        neptune.send_artifact(submission_path)
        print('experiment finished')


if __name__ == '__main__':
    main()
