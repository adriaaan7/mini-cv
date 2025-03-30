import pandas as pd
import json
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.ensemble import RandomForestRegressor

import os
import json
import pandas as pd

class DataReader:
    def __init__(self):
        print('DataReader initialized')

    def read_dataset_into_df(self, dataset_path, num_of_folders_to_read):
        dfs = []
        num_of_folders_read = 0

        for date_folder in os.listdir(dataset_path):
            date_folder_path = os.path.join(dataset_path, date_folder)
            if num_of_folders_read >= num_of_folders_to_read:
                break
            
            if os.path.isdir(date_folder_path):
                print(f'operating on: {date_folder_path}')
                for file in os.listdir(date_folder_path):
                    file_path = os.path.join(date_folder_path, file)
                    try:
                        df = self.read_json_file_to_df(file_path)
                        dfs.append(df)
                    except Exception as e:
                        print(f'Exception occurred while trying to read {file_path} to dataframe {e}')

            num_of_folders_read += 1

        final_df = pd.concat(dfs, ignore_index=True)
        return final_df
    
    def read_json_file_to_df(self, file):
        if file.endswith('.json'):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            
                df = pd.DataFrame(data)
                print(f'Processed file: {file}')
            except Exception as e:
                print(f"Error processing {file}: {e}")

        return df
    