import os
import joblib
from tensorflow.keras.models import load_model

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_dir = os.path.join('saved_models', self.model_name)
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.label_encoder_primary_skill = None
        self.label_encoder_city = None
        self._load_model()

    def _load_model(self):
        try:
            # Load the Keras model
            self.model = load_model(os.path.join(self.model_dir, 'model.keras'))
            print(f'Model {self.model_name} loaded successfully.')
            
            # Load the scalers and encoders
            self.scaler_X = joblib.load(os.path.join(self.model_dir, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(self.model_dir, 'scaler_y.pkl'))
            self.label_encoder_primary_skill = joblib.load(os.path.join(self.model_dir, 'label_encoder_primary_skill.pkl'))
            self.label_encoder_city = joblib.load(os.path.join(self.model_dir, 'label_encoder_city.pkl'))
            print('Scalers and encoders loaded successfully.')
        except Exception as e:
            print(f'Error loading model and related files: {e}')