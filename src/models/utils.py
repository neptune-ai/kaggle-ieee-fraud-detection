import pandas as pd


def sample_negative_class(train, fraction, seed=None):
    train_pos = train[train.isFraud == 1]
    train_neg = train[train.isFraud == 0].sample(frac=fraction, random_state=seed)

    train = pd.concat([train_pos, train_neg], axis=0)
    train = train.sort_values('TransactionDT')
    return train

