import pickle

class Data:
    def __init__(self, data):
        pass

    def read_EEG_embeddings_labels(self, path):
        with open(path, 'rb') as file:
            EEG_word_level_embeddings = pickle.load(file)
            EEG_word_level_labels = pickle.load(file)
        return EEG_word_level_embeddings, EEG_word_level_labels