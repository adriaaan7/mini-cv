import os
import joblib
from tensorflow import keras

class ModelUsage:

    def __init__(self):
        print('ModelUsage initialized')

    '''
    X - dataframe or numpy array with features
    '''
    def predict(self, model_name, X):
        model_dir = os.path.join('saved_models', model_name)
        scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))
        model = keras.models.load_model(os.path.join(model_dir, 'model.keras'))

        X_scaled = scaler_X.transform(X)
        predictions_scaled = model.predict(X_scaled)

        predictions = scaler_y.inverse_transform(predictions_scaled)

        return predictions