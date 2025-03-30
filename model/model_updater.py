from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import joblib
from data_reader import DataReader
from model_loader import ModelLoader
from data_preprocessing import DataPreprocessing

class ModelUpdater:
    def __init__(self, model_name, dataset_path):
        # Load model, scalers, and encoders
        self.model_loader = ModelLoader(model_name)
        self.dataset_path = dataset_path
        self.data_reader = DataReader()

    def update_model(self):
        for date_folder in os.listdir(self.dataset_path):
            date_folder_path = os.path.join(self.dataset_path, date_folder)
            if os.path.isdir(date_folder_path):
                print(f'Processing next directory: {date_folder_path}')
                self._process_and_train_next_directory(date_folder)

        print('Training complete and model saved after processing all directories.')

    def _process_and_train_next_directory(self, folder):
        '''
        Process a single directory, load the data, preprocess it, and continue training the model
        '''
        df = self.data_reader.read_dataset_into_df2(self.dataset_path, folder)  # Read data from a single directory
        if df.empty:
            print(f'No valid data found in {folder}, skipping...')
            return
        
        data_preprocessing = DataPreprocessing()
        df = data_preprocessing.preprocess(df)
        X, y = self._extract_feature_target_data(df)

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Preprocess and scale data
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self._preprocess_data(X_train, X_test, y_train, y_test)

        # Continue training the model
        self._train(X_train_scaled, y_train_scaled)
        
        # Evaluate the model after training on the new batch
        self._evaluate(X_test_scaled, y_test_scaled)

    def _extract_feature_target_data(self, df):
        '''
        Extract features and target data from the DataFrame
        '''
        target_variables = ['salary_b2b', 'salary_permanent']
        y = df[target_variables]
        X = df.drop(columns=target_variables)
        return X, y

    def _preprocess_data(self, X_train, X_test, y_train, y_test):
        '''
        Scale feature values and target values using the scalers from the ModelLoader class
        '''
        X_train_scaled = self.model_loader.scaler_X.fit_transform(X_train)
        X_test_scaled = self.model_loader.scaler_X.transform(X_test)

        y_train_scaled = self.model_loader.scaler_y.fit_transform(y_train)
        y_test_scaled = self.model_loader.scaler_y.transform(y_test)

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def _train(self, X_train_scaled, y_train_scaled):
        '''
        Train the model on the provided data
        '''
        self.model_loader.model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=32, verbose=1)
        
        # Save the updated model after training
        self._save_model()

    def _save_model(self):
        '''
        Save the updated model and related objects (scalers and encoders)
        '''
        self.model_loader.model.save(os.path.join(self.model_loader.model_dir, 'model.keras'))
        joblib.dump(self.model_loader.scaler_X, os.path.join(self.model_loader.model_dir, 'scaler_X.pkl'))
        joblib.dump(self.model_loader.scaler_y, os.path.join(self.model_loader.model_dir, 'scaler_y.pkl'))
        joblib.dump(self.model_loader.label_encoder_primary_skill, os.path.join(self.model_loader.model_dir, 'label_encoder_primary_skill.pkl'))
        joblib.dump(self.model_loader.label_encoder_city, os.path.join(self.model_loader.model_dir, 'label_encoder_city.pkl'))
        print('Model and related objects saved successfully.')

    def _evaluate(self, X_test_scaled, y_test_scaled):
        '''
        Evaluate the model using the provided test data
        '''
        loss, mae = self.model_loader.model.evaluate(X_test_scaled, y_test_scaled, verbose=1)
        print(f'Mean Absolute Error on test data: {mae:.2f}')