# kaggle-ieee-fraud-detection
Example of a project with Neptune experiment management.
It is build following the [cookie-cutter data science project structure](https://github.com/drivendata/cookiecutter-data-science).

This project shows how you can organize your work by tracking:
- data/feature versions

![image1](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/d0c079a7076c2292d38ab78cfa0947bdfc4d35b5/kaggle_properties.png)

- hyperparameters

![image2](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/d0c079a7076c2292d38ab78cfa0947bdfc4d35b5/kaggle_parameters.png)

- metrics

![image3](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/d0c079a7076c2292d38ab78cfa0947bdfc4d35b5/kaggle_metrics.png)

- diagnostic charts like confusion matrix, roc auc curve, or prediction distributions 

![image4](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/d0c079a7076c2292d38ab78cfa0947bdfc4d35b5/kaggle_images.png)

- prediction artifacts 

![image5](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/d0c079a7076c2292d38ab78cfa0947bdfc4d35b5/kaggle_artifacts.png)

- environment

![image6](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/d0c079a7076c2292d38ab78cfa0947bdfc4d35b5/kaggle_environment.png)

- code

![image7](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/d0c079a7076c2292d38ab78cfa0947bdfc4d35b5/kaggle_code.png)

As an added bonus you can learn how to add live monitoring for lightgbm model.

![image8](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/d0c079a7076c2292d38ab78cfa0947bdfc4d35b5/kaggle_charts.png)

And the best part is that all of those experiments can be shared and discussed with everyone you want in Neptune.
Just [go to my project](https://ui.neptune.ai/jakub-czakon/ieee-fraud-detection/experiments) and see for yourself. 

## Installation
By running:

```bash
source Makefile
```

conda environment with all the packages required will be created. 
Also, it will create `data` folder and subdirectories for `raw` data, extracted `features` and `predictions`.

From now on you can activate or deactivate your conda environment by running:

```bash
conda activate ieee
```

or

```bash
conda deactivate 
```

Set your environment variables for `NEPTUNE_API_TOKEN` and `NEPTUNE_PROJECT` by running:

```bash
export NEPTUNE_API_TOKEN=go_to_neptune_app_and_get_your_token
export NEPTUNE_PROJECT=USER_NAME/PROJECT_NAME
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
python src/models/train_lgbm_holdout.py
```

It will create out of fold predictions on train, test and a submission and store them in `data/predictions`

## Questions
You can ask in [kaggle discussion]() or on our [spectrum chat](https://spectrum.chat/neptune-community?tab=posts)


