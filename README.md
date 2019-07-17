# kaggle-ieee-fraud-detection
Example of a project with experiment management.

## Installation
By running:

```bash
source make_project
```

you can create a conda environment with all the packages required. 
Also it will create data folder and subdirectories for raw data, extracted features and predictions.

From now on you can activate or deactivate your conda environment by running:

```bash
conda activate ieee
```

or

```bash
conda deactivate 
```

## Running code

### Feature extraction
Go to `src/features/feature_extractions_v0.py` and change data directories and/or feature names.

Run:
```bash
python src/features/feature_extractions_v0.py
```

It will create features and store them in `data/features`

### Run training, evaluation and prediction
Go to `src/models/train_lgbm.py` and change data directories, features you want to use and model name.
You can also tweak the hyperparameters and/or validation schema or number of folds.

Run:
```bash
python src/models/train_lgbm.py
```

It will create out of fold predictions on train, test and a submission and store them in `data/predictions`

## Questions
You can ask in [kaggle discussion]() or on our [spectrum chat](https://spectrum.chat/neptune-community?tab=posts)


