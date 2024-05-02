from src import read_data



if __name__ == "__main__":

    data_path = "path"
    task_array = ["task1-SR-dataset", "task2-NR-dataset", "task3-TSR-dataset", "task2-NR-2.0-dataset"]

    data_reader = read_data.ReadData(data_path, task_array)
    train_split = 0.8
    val_split = 0.1

    EEG_word_tokens, word_labels, EEG_Sentences = data_reader.create_train_test_datasets("train", train_split, val_split)





