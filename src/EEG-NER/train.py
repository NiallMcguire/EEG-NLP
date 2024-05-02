from src import data

def padding_x_y(EEG_segments, Classes, named_entity_list):
    X = []
    y = []
    NE = []
    for i in range(len(EEG_segments)):
        named_entity = named_entity_list[i]
        label = Classes[i][0]
        #print(label)
        EEG_list = EEG_segments[i]
        for EEG in EEG_list:
            if EEG != []:
                X.append(EEG)
                y.append(label)
                NE.append(named_entity)
    max_seq_length = max([len(x) for x in X])
    #paddding
    for i in range(len(X)):
        padding_count = max_seq_length - len(X[i])
        #print(padding_count)
        for j in range(padding_count):
            X[i].append(np.zeros((105,8)))

    return X, y, NE






if __name__ == "__main__":
    train_path = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\train_NER.pkl"

    test_path = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\test_NER.pkl"

    EEG_path = r"C:\Users\gxb18167\PycharmProjects\EEG-To-Text\SIGIR_Development\EEG-GAN\EEG_Text_Pairs.pkl"

    d = data.Data()

    train_NE, train_EEG_segments, train_Classes = d.NER_save_lists_to_file(train_path)
    test_NE, test_EEG_segments, test_Classes = d.NER_save_lists_to_file(test_path)

    EEG_word_tokens, EEG_word_labels = d.read_EEG_embeddings_labels(EEG_path)

    # padding
    X_train, y_train, NE_list = padding_x_y(train_EEG_segments, train_Classes, train_NE)
    X_train_numpy = np.array(X_train)
    X_train_numpy = reshape_data(X_train_numpy)
    y_train_categorical = encode_labels(y_train)

    X_test, y_test, NE_list_test = padding_x_y(test_EEG_segments, test_Classes, test_NE)
    X_test_numpy = np.array(X_test)
    X_test_numpy = reshape_data(X_test_numpy)
    y_test_categorical = encode_labels(y_test)

