import pickle
import numpy as np
import torch


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


    def read_sentence(self, subjects, Task_Dataset, divider):
        EEG_word_tokens = []
        word_labels = []
        EEG_Sentences = []

        for key in subjects:
            print(f'key = {key}')
            for i in range(divider):
                if Task_Dataset[key][i] is not None:
                    sentence_object = Task_Dataset[key][i]
                    sentence = sentence_object['content']
                    # print(sentence_object['content'])
                    none_switch = False
                    for word in sentence_object['word']:
                        word_eeg_embedding, EEG_word_level_label = self.get_eeg_word_embedding(word)
                        if word_eeg_embedding is not None and torch.isnan(
                                torch.from_numpy(word_eeg_embedding)).any() == False:
                            EEG_word_tokens.append(word_eeg_embedding)
                            word_labels.append(EEG_word_level_label)
                        else:
                            none_switch = True
                    if none_switch == False:
                        EEG_Sentences.append(sentence)

        return EEG_word_tokens, word_labels, EEG_Sentences

    def create_train_test_datasets(self, train_dev_test_status, train_test_split = 0.8, dev_split = 0.1):


        Task_Dataset_List = self.get_task_data()
        for Task_Dataset in Task_Dataset_List:
            subjects = list(Task_Dataset.keys())
            print('[INFO]using subjects: ', subjects)

            total_num_sentence = len(Task_Dataset[subjects[0]])

            train_divider = int(train_test_split * total_num_sentence)
            dev_divider = train_divider + int(dev_split * total_num_sentence)


            print(f'train size = {train_divider}')
            print(f'dev size = {dev_divider}')


            if train_dev_test_status == 'train':
                print('[INFO]initializing a train set...')
                EEG_word_tokens, word_labels, EEG_Sentences = self.read_sentence(subjects, Task_Dataset, train_divider)



            if train_dev_test_status == 'dev':
                print('[INFO]initializing a dev set...')
                EEG_word_tokens, word_labels, EEG_Sentences = self.read_sentence(subjects, Task_Dataset, dev_divider)

            if train_dev_test_status == 'test':
                print('[INFO]initializing a test set...')
                EEG_word_tokens, word_labels, EEG_Sentences = self.read_sentence(subjects, Task_Dataset, total_num_sentence)














