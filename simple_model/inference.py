import numpy as np
import pandas as pd
import pickle
import json
from glob import glob
from feature_processor import generate_df

with open('SETTINGS.json', 'r') as f:
  data = json.load(f)
sub_file = data["SUBMISSION_FILE"]
model_dir = data["MODEL_DIR"]

test = generate_df(False)
test_predictions = np.zeros(test.shape[0])
gl = glob(f'{model_dir}*')
for path in gl:
    file = open(path, "rb")
    model = pickle.load(file)
    test_predictions += model.predict(test[[col for col in test.columns if col not in {"time_id", "target", "row_id"}]]) / len(gl)
test["row_id"] = test["stock_id"].astype(str) + "-" + test["time_id"].astype(str) 
test['target'] = test_predictions
test[['row_id', 'target']].to_csv(sub_file,index = False)
