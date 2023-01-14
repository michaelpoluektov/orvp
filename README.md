# Optiver Realized Volatility Prediction model

This is the README file for the 7th place solution in the ORVP Kaggle competition. Leaderboard [here](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/leaderboard) (you can click on "team solution" to view the write-up as well). All following sections were written as per the Kaggle winning submission guidelines. It is important to note that despite the high leaderboard score, the real-world applications of this submission are limited. The submission used a few careful reverse engineering tricks to re-order the time_IDs given in the competition:

- First, real stock prices were obtained from the normalised prices and tick sizes.
- These prices were used to construct a directed graph with time_IDs as nodes, where the out distance of vertex $v_1$ to vertex $v_2$ is the L2 distance between the vector of stock prices at the end of $v_1$ to the vector of prices at the start of $v_2$.
- Finally, a shortest Hamilton path was approximated on that graph to re-order time_IDs in chronological order: first, KNN was used to find the 6 closest nodes, then for each node a SHP was obtained by brute force. Since stock prices usually follow something close to Brownian motion, this rolling window approach worked.

I would also like to note that this was my very first Python/data science project. I was a first year when I did this competition, and I didn't actually expect anyone to ever see this: the code isn't particularly readable.

# System specification

All notebooks were ran exclusively on Kaggle kernels, all `.py` files were also ran on Kaggle kernels, but also tested on the following platform:

CPU: `Intel M-5Y10c` (dual core 0.80GHz)<br />
GPU: `Intel HD Graphics 5300`<br />
RAM: `8 GB`<br />
OS: `Manjaro Linux - Kernel 5.10.89-1`<br />

This directory contains two models: the model used as a final submission to the competition, scoring 0.20013 RMSPE on the public dataset, and a second, more practical model that could be used in a real life scenario, scoring 0.22955 RMSPE on the public dataset.

The first model is located in the `main_model` folder. I don't expect the host to actually run the first model in production, so it doesn't contain a serialized version of the trained model, a list of dependencies, settings, directory structure etc. You can try the code for yourself by importing the .ipynb file in a Kaggle notebook, importing the competition data and executing it.

The second model is located in the `simple_model` folder. It contains all files described in section B of the winning model documentation guidelines. Read `entry_points.md` for information on how to run the model.

# Files
- `README.md`: This file.
- `Kaggle_ORVP_report.pdf`: Report as outlined in "Section A -- Model Summary" of the winning model documentation guidelines.
- `directory_structure.txt` Directory structure file, as described in section B5.
- `main_model`: Directory containing the following notebooks for the main model:
- `optiver-realized-second-submission.ipynb`: Code for pre-processing features, training models and outputting results of the submitted version (0.20013 final score). It would not yield any useful results in production as is.
- `feature-selection.ipynb`: Code used for feature selection, using pre-processed features extracted from the file above.
 
`simple_model`: Directory containing the following code for the simple model:

- `feature_processor.py`, `train.py` and `inference.py`: Model files required to train the model and generate predictions. See `entry_points.md` for usage.
- `entry_points.md`: List of commands used to run the model.
- `feature-selection.ipynb`: Notebook used for feature selection. Downloads BorutaSHAP from `pip` as it was intended for use on Kaggle kernels.
- `requirements.txt`: Requirements file.
- `SETTINGS.json`: Settings file to define paths described in `entry_points.md`
- `submission.csv`: Sample output file.
- `orvp-features.parquet`: Processed features for training.
- `model` folder: Used to store serialized version(s) of the LightGBM model. Can only contain `.pickle` files representing LightGBM "model" objects.

`data` folder: Used to store raw data. You can extract the competition dataset there to test the model.
