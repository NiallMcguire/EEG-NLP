import pickle
import numpy as np


class ReadData:
    def __init__(self, data_path, task_array):
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
        if self.task1:
            self.task1_data = self.read_file()

        elif self.task2:
            self.task2_data = self.read_file()

        elif self.task3:
            self.task3_data = self.read_file()

        elif self.task2_NRv2:
            self.task2_NRv2_data = self.read_file()




