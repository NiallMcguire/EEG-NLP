import pickle
import re

class Data:
    def __init__(self, data):
        pass

    def read_EEG_embeddings_labels(self, path):
        with open(path, 'rb') as file:
            EEG_word_level_embeddings = pickle.load(file)
            EEG_word_level_labels = pickle.load(file)
        return EEG_word_level_embeddings, EEG_word_level_labels

    def NER_read_sentences(self, path):
        Sentences = []
        Sentence_Classes = []
        Temp_Sentence = []
        Temp_Sentence_Classes = []
        with open(path, 'r', encoding="utf-8") as file:
            # Iterate through each line
            for line in file:
                # Split the line by the tab character
                parts = line.split('\t')
                # Check if there are at least two parts
                if len(parts) >= 2:
                    word = parts[0]
                    class_ = parts[1].strip()  # strip() to remove leading/trailing whitespace

                    Temp_Sentence.append(word)
                    Temp_Sentence_Classes.append(class_)
                else:

                    Sentences.append(Temp_Sentence)
                    Sentence_Classes.append(Temp_Sentence_Classes)
                    Temp_Sentence = []
                    Temp_Sentence_Classes = []
        return Sentences, Sentence_Classes

    def NER_get_sentences_EEG(self, labels, EEG_embeddings):
        Sentences = []
        current_sentence = []

        EEG_Sentencs = []
        EEG_index = 0
        for i in range(len(labels)):
            # Check if the word marks the start of a new sentence
            word = labels[i]
            if word == "SOS":
                # If it does, append the current sentence to the list of sentences
                if len(current_sentence) > 0:
                    Sentences.append(current_sentence)
                    sentence_length = len(current_sentence)
                    EEG_segment = EEG_embeddings[EEG_index:EEG_index + sentence_length]
                    EEG_index += sentence_length
                    EEG_Sentencs.append(EEG_segment)

                    # Start a new sentence
                    current_sentence = []
            else:
                # Add the word to the current sentence
                current_sentence.append(word)

        return Sentences, EEG_Sentencs

    def NER_combine_sentences(self, sentences):
        # Combine each internal list into a string
        combined_sentences = [' '.join(sentence) for sentence in sentences]
        # Initialize an empty list to store the list of words for each sentence
        list_of_words = []
        # Iterate over each sentence in the list
        for sentence in combined_sentences:
            # Split the sentence into a list of words including punctuation
            words = re.findall(r'\b\w+\b|[^\w\s]', sentence)
            # Append the list of words to the list of lists
            list_of_words.append(words)
        return list_of_words


    def NER_get_named_entities(self, Sentences_labels, Sentence_Classes):
        List_of_NE = []
        List_of_NE_Labels = []
        Named_Entity = []
        Named_Entity_Label = []

        for i in range(len(Sentences_labels)):
            current_sentence = Sentences_labels[i]
            current_sentence_label = Sentence_Classes[i]

            for j in range(len(current_sentence)):
                current_word = current_sentence[j]
                #print(current_word)
                current_word_label = current_sentence_label[j]
                #print(current_word_label)
                if current_word_label != 'O':
                    Named_Entity.append(current_word)
                    Named_Entity_Label.append(current_word_label)
                else:
                    if Named_Entity:
                        List_of_NE.append(Named_Entity)
                        List_of_NE_Labels.append(Named_Entity_Label)

                        Named_Entity = []
                        Named_Entity_Label = []
        return List_of_NE, List_of_NE_Labels

    def NER_align_sentences(self, path_normal_reading, path_task_reading, path_sentiment):
        normal_reading_sentences, normal_reading_classes = self.NER_read_sentences(path_normal_reading)
        task_reading_sentences, task_reading_classes = self.NER_read_sentences(path_task_reading)
        sentiment_sentences, sentiment_classes = self.NER_read_sentences(path_sentiment)

        Sentences_labels = [item for sublist in [normal_reading_sentences, task_reading_sentences, sentiment_sentences]
                            for item in sublist]
        Sentence_Classes = [item for sublist in [normal_reading_classes, task_reading_classes, sentiment_classes] for
                            item in sublist]

        train_path = r"C:\Users\gxb18167\PycharmProjects\EEG-To-Text\SIGIR_Development\EEG-GAN\EEG_Text_Pairs_Sentence.pkl"
        test_path = r"C:\Users\gxb18167\PycharmProjects\EEG-To-Text\SIGIR_Development\EEG-GAN\Test_EEG_Text_Pairs_Sentence.pkl"

        EEG_word_tokens, EEG_word_labels = self.read_EEG_embeddings_labels(train_path)
        Test_EEG_word_tokens, Test_EEG_word_labels = self.read_EEG_embeddings_labels(test_path)

        EEG_word_level_sentences, EEG_sentence_embeddings = self.NER_get_sentences_EEG(EEG_word_tokens,
                                                                              EEG_word_labels)
        Test_EEG_word_level_sentences, Test_EEG_sentence_embeddings = self.NER_get_sentences_EEG(Test_EEG_word_tokens,
                                                                                        Test_EEG_word_labels)

        # Combine the sentences
        list_of_words = self.NER_combine_sentences(EEG_word_level_sentences)
        Test_list_of_words = self.NER_combine_sentences(Test_EEG_word_level_sentences)

        # Get the named entities
        List_of_NE, List_of_NE_Labels = self.NER_get_named_entities(Sentences_labels, Sentence_Classes)