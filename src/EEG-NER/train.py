from src import data



if __name__ == "__main__":
    train_path = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\train_NER.pkl"

    test_path = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\test_NER.pkl"

    EEG_path = r"C:\Users\gxb18167\PycharmProjects\EEG-To-Text\SIGIR_Development\EEG-GAN\EEG_Text_Pairs.pkl"

    d = data.Data()

    train_NE, train_EEG_segments, train_Classes = d.NER_save_lists_to_file(train_path)
    test_NE, test_EEG_segments, test_Classes = d.NER_save_lists_to_file(test_path)

    EEG_word_tokens, EEG_word_labels = d.read_EEG_embeddings_labels(EEG_path)

