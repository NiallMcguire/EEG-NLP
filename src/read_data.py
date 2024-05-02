import pickle
import numpy as np


class ReadData:
    def __init__(self, data_path, task_array, subject_choice = 'ALL', eeg_type = 'GD', eeg_bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], data_setting = 'unique'):
        self.data_path = data_path

        if "task1-SR-dataset" in task_array:
            self.task1 = True
        elif "task2-NR-dataset" in task_array:
            self.task2 = True
        elif "task3-TSR-dataset" in task_array:
            self.task3 = True
        elif "task2-NR-2.0-dataset" in task_array:
            self.task2_NRv2 = True

    def get_eeg_word_embedding(self, word, eeg_type='GD', bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']):
        EEG_frequency_features = []
        word_label = word['content']
        for band in bands:
            EEG_frequency_features.append(word['word_level_EEG'][eeg_type][eeg_type + band])
        EEG_word_token = np.concatenate(EEG_frequency_features)
        if len(EEG_word_token) != 105 * len(bands):
            print(
                f'expect word eeg embedding dim to be {105 * len(bands)}, but got {len(EEG_word_token)}, return None')
            EEG_word_token = None
        else:
            EEG_word_token = EEG_word_token.reshape(105, 8)

        return EEG_word_token, word_label

    def read_file(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def get_task_data(self):
        task_data_list = []

        if self.task1:
            self.task1_data = self.read_file()
            task_data_list.append(self.task1_data)

        elif self.task2:
            self.task2_data = self.read_file()
            task_data_list.append(self.task2_data)

        elif self.task3:
            self.task3_data = self.read_file()
            task_data_list.append(self.task3_data)

        elif self.task2_NRv2:
            self.task2_NRv2_data = self.read_file()
            task_data_list.append(self.task2_NRv2_data)

        Task_Dataset_List = task_data_list
        if not isinstance(task_data_list, list):
            Task_Dataset_List = [task_data_list]

        return Task_Dataset_List

    def create_train_test_datasets(self):












