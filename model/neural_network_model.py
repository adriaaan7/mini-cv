import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from base_model import BaseModel
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class NeuralNetworkModel(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
        print('NeuralNetworkModel initialized')

    def create_model(self, df):
        X, y = self._extract_feature_target_data(df, ['salary_b2b', 'salary_permanent'])
        X_train, X_test, X_val, y_train, y_test, y_val = self._split_data(X, y, 0.7)
        self.input_shape = self.X_train.shape[1]
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self._preprocess_data(X_train, X_test, y_train, y_test)

        self._create_nn()
        self._train(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
        self._evaluate(X_test_scaled, y_test_scaled)

    def _preprocess_data(self, X_train, X_test, y_train, y_test):
        return self._scale_train_test_data(self.scaler_X, self.scaler_y, X_train, X_test, y_train, y_test)

    def _scale_train_test_data(self, scaler_X, scaler_y, X_train, X_test, y_train, y_test):
        '''
        Scale feature values
        '''
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        '''
        Scale target variables' values
        '''
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        self._save_scalers()
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

    def _save_scalers(self):
        model_dir = os.path.join('saved_models', self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(model_dir, 'scaler_y.pkl'))

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
                    epochs=100, 
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