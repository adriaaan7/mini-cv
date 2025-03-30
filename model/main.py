from data_reader import DataReader
from neural_network_model import NeuralNetworkModel
from data_preprocessing import DataPreprocessing
from model_updater import ModelUpdater
import os
import shutil
from collections import defaultdict
import json
import pandas as pd
from model_usage import ModelUsage

DATASET_PATH = '.././dataset'
NUM_OF_FOLDERS_TO_READ = 1
MODEL_NAME = 'nn-model-1-v4'

# split_dataset_into_folders(".././dataset_raw_files", ".././dataset_split", batch_size=15)

def main():
    # create_model()
    # update_model()
    use_model()
    print('main')

def use_model():
    sample_data = {
        'city': ['Warszawa'],  
        'primary_skill': ['java'],  
        'Java': [1],  
        'TypeScript': [1],  
        'Angular': [1],  
        'Ruby': [0], 
        'Spring Boot': [1],  
        'Python': [1],  
        'Kotlin': [0], 
        'JavaScript': [1],  
        'Spring': [1],  
        'C#': [1], 
        'PHP': [0],  
        'Swift': [0],  
        'SQL': [1],  
        'React': [0], 
        'Objective-C': [0],  
        'Rust': [0],  
        'C++': [0],  
        'C': [0],  
        'Scala': [0], 
        'Elixir': [0],  
        'Go': [0],  
        'PyTorch': [0],  
        'R': [0],  
        'Haskell': [0],  
        'Vue': [0],  
        'Perl': [0],  
        'Dart': [0],  
        'Pandas': [0],  
        'Shell': [0],  
        'Numpy': [0],  
        'VHDL': [0],  
        'Clojure': [0],  
        'MATLAB': [0], 
        'Lua': [0],  
        '.NET': [0],  
        'Tensorflow': [0],  
        'workplace_type_office': [0], 
        'workplace_type_partly_remote': [1],  
        'workplace_type_remote': [1], 
        'experience_level_junior': [0],  
        'experience_level_mid': [0], 
        'experience_level_senior': [1] 
    }

    print(sample_data)
    # Convert to pandas DataFrame
    X_test = pd.DataFrame(sample_data)

    # Assume that ModelUsage class is already initialized
    model_usage = ModelUsage()
    pred = model_usage.predict(MODEL_NAME, X_test)
    print(pred)


def create_model():
    data_reader = DataReader()
    df = data_reader.read_dataset_into_df(DATASET_PATH, NUM_OF_FOLDERS_TO_READ)
    print(df.head())
    nn_model = NeuralNetworkModel(MODEL_NAME)
    nn_model.create_model(df)
    print(f'Mean Absolute Error: {nn_model.mae:.2f}')

def update_model():
    # Initialize the ModelUpdater with the model name and dataset directory
    model_updater = ModelUpdater(model_name=MODEL_NAME, dataset_path=".././dataset_split")

    # Start the training process, it will read directories one by one, update the model, and save it
    model_updater.update_model()
    
if __name__ == '__main__':
    main()