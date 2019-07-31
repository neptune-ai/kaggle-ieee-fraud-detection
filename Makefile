# create env
conda env create -f environment.yml
conda activate ieee

# add ieee kernel
python -m ipykernel install --user --name ieee --display-name "ieee"

# enable neptune notebook extension
jupyter labextension install neptune-notebooks

# create data folders
mkdir data
mkdir data/raw data/features data/predictions
