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
from sklearn.model_selection import KFold
from src.utils import read_config

CONFIG = read_config(config_path=os.getenv('CONFIG_PATH'))

neptune.init(project_qualified_name=CONFIG.project)

FEATURES_DATA_PATH = CONFIG.data.features_data_path
PREDICTION_DATA_PATH = CONFIG.data.prediction_data_path
SAMPLE_SUBMISSION_PATH = CONFIG.data.sample_submission_path
FEATURE_NAME = 'v0'
MODEL_NAME = 'lgbm'
NROWS = 100000

VALIDATION_PARAMS = {'validation_seed': 1234,
                     'validation_schema': 'kfold',
                     'n_splits': 5}

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
                   'num_boosting_rounds': 5000,
                   'early_stopping_rounds': 200
                   }


def fit_predict(X, y, X_test, folds, model_params, training_params):
    in_fold, out_of_fold, test_preds = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X_test))
    for fold_nr, (trn_idx, val_idx) in enumerate(folds.split(X.values, y.values)):
        print("Fold {}".format(fold_nr))

        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        trn_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_valid, y_valid)

        monitor = neptune_monitor(prefix='fold{}_'.format(fold_nr))
        clf = lgb.train(model_params, trn_data,
                        training_params['num_boosting_rounds'],
                        valid_sets=[trn_data, val_data],
                        early_stopping_rounds=training_params['early_stopping_rounds'],
                        callbacks=[monitor])
        in_fold[trn_idx] = clf.predict(X.iloc[trn_idx], num_iteration=clf.best_iteration)
        out_of_fold[val_idx] = clf.predict(X.iloc[val_idx], num_iteration=clf.best_iteration)
        test_preds += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
    return in_fold, out_of_fold, test_preds


def fmt_preds(y_pred):
    return np.concatenate((1.0 - y_pred.reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)


if __name__ == '__main__':
    print('loading data')
    train_features_path = os.path.join(FEATURES_DATA_PATH, 'train_features_' + FEATURE_NAME + '.csv')
    test_features_path = os.path.join(FEATURES_DATA_PATH, 'test_features_' + FEATURE_NAME + '.csv')

    print('... train')
    train = pd.read_csv(train_features_path, nrows=TRAINING_PARAMS['nrows'])
    X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
    y = train.sort_values('TransactionDT')['isFraud']
    train = train[["TransactionDT", 'TransactionID']]

    print('... test')
    test = pd.read_csv(test_features_path, nrows=TRAINING_PARAMS['nrows'])
    X_test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
    test = test[["TransactionDT", 'TransactionID']]

    folds = KFold(n_splits=VALIDATION_PARAMS['n_splits'], random_state=VALIDATION_PARAMS['validation_seed'])

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
        in_fold, out_of_fold, test_preds = fit_predict(X, y, X_test, folds, MODEL_PARAMS, TRAINING_PARAMS)

        print('logging metrics')
        train_auc, valid_auc = roc_auc_score(y, in_fold), roc_auc_score(y, out_of_fold)
        neptune.send_metric('train_auc', train_auc)
        neptune.send_metric('valid_auc', valid_auc)
        send_binary_classification_report(y, fmt_preds(out_of_fold), channel_name='valid_classification_report')

        print('postprocessing predictions')
        train_predictions_path = os.path.join(PREDICTION_DATA_PATH,
                                              'train_prediction_{}_{}.csv'.format(FEATURE_NAME, MODEL_NAME))
        test_predictions_path = os.path.join(PREDICTION_DATA_PATH,
                                             'test_prediction_{}_{}.csv'.format(FEATURE_NAME, MODEL_NAME))
        submission_path = os.path.join(PREDICTION_DATA_PATH,
                                       'submission_{}_{}.csv'.format(FEATURE_NAME, MODEL_NAME))
        submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

        train = pd.concat([train, pd.DataFrame(out_of_fold, columns=['prediction'])], axis=1)
        test = pd.concat([test, pd.DataFrame(test_preds, columns=['prediction'])], axis=1)
        submission['isFraud'] = pd.merge(submission, test, on='TransactionID')['prediction']
        train.to_csv(train_predictions_path, index=None)
        test.to_csv(test_predictions_path, index=None)
        submission.to_csv(submission_path, index=None)
        neptune.send_artifact(train_predictions_path)
        neptune.send_artifact(test_predictions_path)
        neptune.send_artifact(submission_path)
        print('experiment finished')
