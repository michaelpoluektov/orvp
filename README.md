# Optiver Realized Volatility Prediction model

This is the README file for the 7th place solution in the ORVP Kaggle competition. All notebooks were ran exclusively on Kaggle kernels, all `.py` files were also ran on Kaggle kernels, but also tested on the following platform:

`Intel M-5Y10c` dual core 0.80GHz CPU,
`Intel HD Graphics 5300` GPU,
`8 GB` RAM,
`Manjaro Linux - Kernel 5.10.89-1` OS.

This directory contains two models: the model used as a final submission to the competition, scoring 0.20013 RMSPE on the public dataset, and a second, more practical model that could be used in a real life scenario, scoring 0.22955 RMSPE on the public dataset.

The first model is located in the `main_model` folder. I don't expect the host to actually run the first model in production, so it doesn't contain a serialized version of the trained model, a list of dependencies, settings, directory structure etc. You can try the code for yourself by importing the .ipynb file in a Kaggle notebook, importing the competition data and executing it.

The second model is located in the `simple_model` folder. It contains all files described in section B of the winning model documentation guidelines. Read `entry_points.md` for information on how to run the model.

# Files
`README.md`: This file.
`Kaggle_ORVP_report.pdf`: Report as outlined in "Section A -- Model Summary" of the winning model documentation guidelines.
`directory_structure.txt` Directory structure file, as described in section B5.
`main_model`: Directory containing the following notebooks for the main model:
- `optiver-realized-second-submission.ipynb`: Code for pre-processing features, training models and outputting results of the submitted version (0.20013 final score). It would not yield any useful results in production as is.
- `feature-selection.ipynb`: Code used for feature selection, using pre-processed features extracted from the file above.
 
`simple_model`: Directory containing the following code for the simple model:

- `feature_processor.py`, `train.py` and `inference.py`: Model files required to train the model and generate predictions. See `entry_points.md` for usage.
- `entry_points.md`: List of commands used to run the model.
- `feature-selection.ipynb`: Notebook used for feature selection. Downloads BarutaSHAP from `pip` as it was intended for use on Kaggle kernels.
- `requirements.txt`: Requirements file.
- `SETTINGS.json`: Settings file to define paths described in `entry_points.md`
- `submission.csv`: Sample output file.
- `orvp-features.parquet`: Processed features for training.
- `model` folder: Used to store serialized version(s) of the LightGBM model. Can only contain `.pickle` files representing LightGBM "model" objects.

`data` folder: Used to store raw data. You can extract the competition dataset there to test the model.
