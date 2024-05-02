from src import data
from src import utils
from src import Networks
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")


    train_path = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\train_NER.pkl"

    test_path = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\test_NER.pkl"

    EEG_path = r"C:\Users\gxb18167\PycharmProjects\EEG-To-Text\SIGIR_Development\EEG-GAN\EEG_Text_Pairs.pkl"

    d = data.Data()
    util = utils.Utils()



    train_NE, train_EEG_segments, train_Classes = d.NER_save_lists_to_file(train_path)
    test_NE, test_EEG_segments, test_Classes = d.NER_save_lists_to_file(test_path)

    EEG_word_tokens, EEG_word_labels = d.read_EEG_embeddings_labels(EEG_path)

    # padding
    X_train, y_train, NE_list = util.NER_padding_x_y(train_EEG_segments, train_Classes, train_NE)
    X_train_numpy = np.array(X_train)
    X_train_numpy = util.NER_reshape_data(X_train_numpy)
    y_train_categorical = util.encode_labels(y_train)

    X_test, y_test, NE_list_test = util.NER_padding_x_y(test_EEG_segments, test_Classes, test_NE)
    X_test_numpy = np.array(X_test)
    X_test_numpy = util.NER_reshape_data(X_test_numpy)
    y_test_categorical = util.encode_labels(y_test)

    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(X_train_numpy, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_categorical, dtype=torch.float32)  # Assuming your labels are integers

    # Create a custom dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

    # Define batch size
    batch_size = 32  # Adjust according to your preference
    # Define model parameters
    input_size = 840
    hidden_size = 64
    num_layers = 2
    num_classes = 3

    # Create the train loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = Networks.BLSTM(input_size, hidden_size, num_layers, num_classes)
    model.to(device)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



