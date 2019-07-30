import os

import neptune
from neptunecontrib.api.utils import get_filepaths
import neptunecontrib.monitoring.skopt as sk_utils
import pandas as pd
import skopt
from sklearn.metrics import roc_auc_score

from src.models.train_lgbm_holdout import FEATURES_DATA_PATH, FEATURE_NAME, TRAINING_PARAMS, VALIDATION_PARAMS, \
    fit_predict
from src.models.utils import sample_negative_class

STATIC_PARAMS = {'objective': 'binary',
                 "boosting_type": "gbdt",
                 "bagging_seed": 11,
                 "metric": 'auc',
                 "verbosity": -1,
                 'seed': 1234
                 }

HPO_PARAMS = {'n_calls': 1000,
              'n_random_starts': 30,
              'base_estimator': 'ET',
              'acq_func': 'EI',
              'xi': 0.01,
              'kappa': 1.96,
              'n_points': 10000,
              }

SPACE = [skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
         skopt.space.Integer(1, 30, name='max_depth'),
         skopt.space.Integer(2, 100, name='num_leaves'),
         skopt.space.Integer(10, 1000, name='min_data_in_leaf'),
         skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
         skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform'),
         ]


def main():
    print('loading data')
    train_features_path = os.path.join(FEATURES_DATA_PATH, 'train_features_' + FEATURE_NAME + '.csv')

    print('... train')
    train = pd.read_csv(train_features_path, nrows=TRAINING_PARAMS['nrows'])

    idx_split = int((1 - VALIDATION_PARAMS['validation_fraction']) * len(train))
    train, valid = train[:idx_split], train[idx_split:]

    train = sample_negative_class(train,
                                  fraction=TRAINING_PARAMS['negative_sample_fraction'],
                                  seed=TRAINING_PARAMS['negative_sample_seed'])

    @skopt.utils.use_named_args(SPACE)
    def objective(**params):
        model_params = {**params, **STATIC_PARAMS}
        valid_preds = fit_predict(train, valid, None, model_params, TRAINING_PARAMS, fine_tuning=True)
        valid_auc = roc_auc_score(valid['isFraud'], valid_preds)
        return -1.0 * valid_auc

    experiment_params = {**STATIC_PARAMS,
                         **TRAINING_PARAMS,
                         **HPO_PARAMS,
                         }

    with neptune.create_experiment(name='skopt forest sweep',
                                   params=experiment_params,
                                   tags=['skopt', 'forest', 'tune'],
                                   upload_source_files=get_filepaths()):
        results = skopt.forest_minimize(objective, SPACE,
                                        callback=[sk_utils.NeptuneMonitor()],
                                        **HPO_PARAMS)
        best_auc = -1.0 * results.fun
        best_params = results.x

        neptune.send_metric('valid_auc', best_auc)
        neptune.set_property('best_parameters', str(best_params))

        sk_utils.send_best_parameters(results)
        sk_utils.send_plot_convergence(results, channel_name='diagnostics_hpo')
        sk_utils.send_plot_evaluations(results, channel_name='diagnostics_hpo')
        sk_utils.send_plot_objective(results, channel_name='diagnostics_hpo')


if __name__ == '__main__':
    main()
