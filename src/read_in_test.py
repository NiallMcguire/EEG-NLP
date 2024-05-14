import data



if __name__ == "__main__":
    task_name = "task1, task2, taskNRv2"

    data = data.Data()
    EEG_word_level_embeddings, EEG_word_level_labels = data.create_custom_dataset(task_name)

    EEG_word_level_sentences, EEG_sentence_embeddings = data.NER_get_sentences_EEG(EEG_word_level_labels,
                                                                          EEG_word_level_embeddings)

    list_of_words = data.NER_combine_sentences(EEG_word_level_sentences)



    print("Data created successfully!")