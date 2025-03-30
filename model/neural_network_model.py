import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from base_model import BaseModel
import os
from data_preprocessing import DataPreprocessing

class NeuralNetworkModel(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
        print('NeuralNetworkModel initialized')

    def create_model(self, df):
        '''
        Preprocess data
        '''
        data_preprocessing = DataPreprocessing()
        df = data_preprocessing.preprocess(df)

        X, y = self._extract_feature_target_data(df, ['salary_b2b', 'salary_permanent'])
        X_train, X_test, X_val, y_train, y_test, y_val = self._split_data(X, y, 0.7)
        self.input_shape = X_train.shape[1]
        print(df.head())
        print(df.columns)
        print(df.size)

        '''
        Scale data
        '''
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = data_preprocessing.scale_data(X_train, X_test, y_train, y_test, self.model_name)
        data_preprocessing._save_scalers(self.model_name)

        self._create_nn()
        self._train(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
        self._evaluate(X_test_scaled, y_test_scaled)

    def _create_nn(self):
        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.input_shape,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='linear')  # Output layer (2 targets: salary_b2b, salary_permanent)
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def _train(self, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled):
        self.history = self.model.fit(
                    X_train_scaled, y_train_scaled, 
                    validation_data=(X_test_scaled, y_test_scaled),
                    epochs=20, 
                    batch_size=32
                    )
        
        self._save_model()

    def _save_model(self):
        model_dir = os.path.join('saved_models', self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        self.model.save(os.path.join(model_dir, 'model.keras'))
        
    def _evaluate(self, X_test_scaled, y_test_scaled):
        self.loss, self.mae = self.model.evaluate(X_test_scaled, y_test_scaled)
        print(f'Mean Absolute Error: {self.mae:.2f}')