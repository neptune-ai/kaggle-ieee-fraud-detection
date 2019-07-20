import sys
import subprocess

subprocess.call(['pip', 'install', 'neptune-client', 'neptune-contrib'])

REPO_PATH = "../input/ieee-repo/repository/neptune-ml-kaggle-ieee-fraud-detection-fbedab9"
sys.path.append(REPO_PATH)

os.environ["CONFIG_PATH"] = os.path.join(REPO_PATH, 'kaggle_config.yaml')
os.environ["NEPTUNE_API_TOKEN"] = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJiNzA2YmM4Zi03NmY5LTRjMmUtOTM5ZC00YmEwMzZmOTMyZTQifQ=="

# Feature extraction
# subprocess.call(['python', os.path.join(REPO_PATH, 'src', 'features', 'feature_extraction_v0.py')])

# Model training
subprocess.call(['python', os.path.join(REPO_PATH, 'src', 'models', 'train_lgbm_holdout.py')])