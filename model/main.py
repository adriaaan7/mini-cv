from data_reader import DataReader
from neural_network_model import NeuralNetworkModel
from data_preprocessing import DataPreprocessing

DATASET_PATH = '.././dataset'
NUM_OF_FOLDERS_TO_READ = 5

def main():
    data_reader = DataReader()
    df = data_reader.read_dataset_into_df(DATASET_PATH, NUM_OF_FOLDERS_TO_READ)
    data_preprocessing = DataPreprocessing()
    df = data_preprocessing.preprocess(df)
    print(df.head())
    nn_model = NeuralNetworkModel('nn-model-5')
    nn_model.create_model(df)
    
if __name__ == '__main__':
    main()