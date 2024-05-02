import pickle
import numpy as np


class ReadData:
    def __init__(self, data_path):
        self.data_path = data_path

    def read_data(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_data(self):
        data = self.read_data()
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    def get_labels(self):
        data = self.read_data()
        return data['labels']