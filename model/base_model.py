from sklearn.model_selection import train_test_split

class BaseModel:
    def __init__(self):
        print('BaseModel initialized')

    '''
    target_variables - list of column names to be treated as target variables
    '''
    def _extract_feature_target_data(self, df, target_variables):
        y = df[target_variables]
        X = df.drop(columns=target_variables)
        return X, y


    '''
    train_size - size of dataset for training, value range: [0, 1]
    '''
    def _split_data(self, X, y, train_size):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size= 1 - train_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        return X_train, X_test, X_val, y_train, y_test, y_val

    