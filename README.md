# Variants classifier
An xgboost-based tool that classifies SNPs data.

## Dependencies
Install `conda` with your preferred packet manager.

Create a new environment 
- `conda create -y -n xgb python==3.8`

Load it 
- `conda activate xgb`

Install the dependencies 
- `yes | pip install xgboost==2.0.0 graphviz pandas matplotlib numpy scikit-learn`

## How to run
Load the environment (if not already loaded)
- `conda activate xgb`

Launch the script 
- `python xgbclassifier.py --dataset features.csv --target mortality.csv`

See the help for advanced options
- `python xgbclassifier.py --help`

## Cite
If you use this repo in your work, please cite (paper under submission)

```
@article{ml2025phenotype,
  title={Machine Learning Methods for Phenotype Prediction from High-Dimensional, Low Population Aquaculture Data},
  author={Faldani, Giovanni and Rossignolo, Enrico and Signor, Eleonora and Longo, Alessio and Faggion, Sara and Bargelloni, Luca and Comin, Matteo and Pizzi, Cinzia and others},
  journal={BIOSTEC},
  pages={638--646},
  year={2025}
}

```