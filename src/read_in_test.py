import data



if __name__ == "__main__":
    data = data.Data()

    task_name = "task1, task2, taskNRv2"

    path_normal_reading = r'/users/gxb18167/Datasets/ZuCo/ZuCo_NER_Labels/zuco1_normalreading_ner.txt'

    path_task_reading = r"/users/gxb18167/Datasets/ZuCo/ZuCo_NER_Labels/zuco1_taskreading_ner.txt"

    path_sentiment = r"/users/gxb18167/Datasets/ZuCo/ZuCo_NER_Labels/zuco1_sentiment_ner.txt"

    normal_reading_sentences, normal_reading_classes = data.NER_read_sentences(path_normal_reading)
    task_reading_sentences, task_reading_classes = data.NER_read_sentences(path_task_reading)
    sentiment_sentences, sentiment_classes = data.NER_read_sentences(path_sentiment)

    # combine
    Sentences_labels = [item for sublist in [normal_reading_sentences, task_reading_sentences, sentiment_sentences] for
                        item in sublist]
    Sentence_Classes = [item for sublist in [normal_reading_classes, task_reading_classes, sentiment_classes] for item
                        in sublist]


    EEG_word_level_embeddings, EEG_word_level_labels = data.create_custom_dataset(task_name)

    EEG_word_level_sentences, EEG_sentence_embeddings = data.NER_get_sentences_EEG(EEG_word_level_labels,
                                                                          EEG_word_level_embeddings)

    list_of_words = data.NER_combine_sentences(EEG_word_level_sentences)


    # Get the named entities
    List_of_NE, List_of_NE_Labels = data.NER_get_named_entities(Sentences_labels, Sentence_Classes)

    print("Data created successfully!")