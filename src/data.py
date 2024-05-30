import pickle
import re
import numpy as np
import torch

class Data:
    def __init__(self):
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

    def is_named_entity_in_sentences(self, named_entity, sentences, embeddings):

        named_entity_eeg_list = []
        for sentence_index in range(len(sentences)):
            sentence = sentences[sentence_index]
            for i in range(len(sentence) - len(named_entity) + 1):
                if sentence[i:i + len(named_entity)] == named_entity:
                    named_entity_eeg = embeddings[sentence_index][i:i + len(named_entity)]
                    if named_entity_eeg != [] and len(named_entity_eeg) == len(named_entity):
                        named_entity_eeg_list.append(named_entity_eeg)

        return named_entity_eeg_list

    def NER_get_EEG_Class_NE(self, words, embeddings, List_of_NE, List_of_NE_Labels):
        # Check if each named entity is in the list of sentences
        NE_EEG_segment = []
        NE_EEG_Class = []
        NE_list = []

        for i in range(len(List_of_NE)):
            Class = List_of_NE_Labels[i]
            NE = List_of_NE[i]
            # print(NE)
            EEG_list = self.is_named_entity_in_sentences(NE, words, embeddings)
            if EEG_list != []:
                NE_EEG_segment.append(EEG_list)
                NE_EEG_Class.append(Class)
                NE_list.append(NE)

        return NE_EEG_segment, NE_EEG_Class, NE_list

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

    def NER_get_unique_entities(self, entities, EEG_segments, Classes):
        seen_entities = set()
        unique_entities = []
        unique_EEG_segments = []
        unique_Classes = []

        for i in range(len(entities)):
            sublist = entities[i]
            if tuple(sublist) not in seen_entities:
                unique_entities.append(sublist)
                seen_entities.add(tuple(sublist))
                unique_EEG_segments.append(EEG_segments[i])
                unique_Classes.append(Classes[i])

        return unique_entities, unique_EEG_segments, unique_Classes

    def NER_read_custom_files(self, path):
        # Open the pickle file in binary write mode
        with open(path, 'rb') as f:
            # Load each list from the file
            NE = pickle.load(f)
            EEG_segments = pickle.load(f)
            Classes = pickle.load(f)

        return NE, EEG_segments, Classes

    def NER_align_sentences(self, path_normal_reading, path_task_reading, path_sentiment, train_path, test_path):
        path_normal_reading = r'C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\zuco1_normalreading_ner.txt'

        path_task_reading = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\zuco1_taskreading_ner.txt"

        path_sentiment = r"C:\Users\gxb18167\PycharmProjects\SIGIR_EEG_GAN\Development\Named-Entity-Classification\Data-Management\zuco1_sentiment_ner.txt"


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

        # Get the EEG embeddings for the named entities
        NE_EEG_segment, NE_EEG_Class, NE = self.NER_get_EEG_Class_NE(list_of_words, EEG_sentence_embeddings, List_of_NE, List_of_NE_Labels)

        # Test set
        Test_NE_EEG_segment, Test_NE_EEG_Class, Test_NE = self.NER_get_EEG_Class_NE(Test_list_of_words, Test_EEG_sentence_embeddings, List_of_NE, List_of_NE_Labels)

        # Get the unique named entities
        unique_entities, unique_EEG_segments, unique_Classes = self.NER_get_unique_entities(NE, NE_EEG_segment, NE_EEG_Class)

        # Test set
        Test_unique_entities, Test_unique_EEG_segments, Test_unique_Classes = self.NER_get_unique_entities(Test_NE, Test_NE_EEG_segment, Test_NE_EEG_Class)


        #with open('train_NER.pkl', 'wb') as f:

        #    pickle.dump(unique_entities, f)
        #    pickle.dump(unique_EEG_segments, f)
        #    pickle.dump(unique_Classes, f)

        #with open('train_NER.pkl', 'wb') as f:

        #    pickle.dump(unique_entities, f)
        #    pickle.dump(unique_EEG_segments, f)
        #    pickle.dump(unique_Classes, f)


        return unique_entities, unique_EEG_segments, unique_Classes, Test_unique_entities, Test_unique_EEG_segments, Test_unique_Classes


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

    def create_custom_dataset(self, task_name):
        whole_dataset_dicts = []

        if 'task1' in task_name:
            dataset_path_task1 = r'/users/gxb18167/Datasets/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
            with open(dataset_path_task1, 'rb') as handle:
                whole_dataset_dicts.append(pickle.load(handle))

        if 'task2' in task_name:
            dataset_path_task2 = r'/users/gxb18167/Datasets/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
            with open(dataset_path_task2, 'rb') as handle:
                whole_dataset_dicts.append(pickle.load(handle))

        if 'task3' in task_name:
            dataset_path_task3 = r'/users/gxb18167/Datasets/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
            with open(dataset_path_task3, 'rb') as handle:
                whole_dataset_dicts.append(pickle.load(handle))

        if 'taskNRv2' in task_name:
            dataset_path_taskNRv2 = r'/users/gxb18167/Datasets/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
            with open(dataset_path_taskNRv2, 'rb') as handle:
                whole_dataset_dicts.append(pickle.load(handle))

        print("Loaded in", len(whole_dataset_dicts), "task datasets")

        Task_Dataset_List = whole_dataset_dicts
        if not isinstance(whole_dataset_dicts, list):
            Task_Dataset_List = [whole_dataset_dicts]

        EEG_word_level_embeddings = []
        EEG_word_level_labels = []
        # Main loop, looping through each task
        for Task_Dataset in Task_Dataset_List:
            subjects = list(Task_Dataset.keys())
            print('[INFO]using subjects: ', subjects)

            total_num_sentence = len(Task_Dataset[subjects[0]])

            for key in subjects:
                print(f'key = {key}')
                for i in range(total_num_sentence):
                    if Task_Dataset[key][i] is not None:
                        sentence_object = Task_Dataset[key][i]

                        Sentence_EEG_word_level_embeddings = []
                        Sentence_word_level_labels = []

                        Sentence_word_level_labels.append("SOS")
                        for word in sentence_object['word']:

                            word_eeg_embedding, EEG_word_level_label = self.get_eeg_word_embedding(word)

                            if word_eeg_embedding is not None and torch.isnan(
                                    torch.from_numpy(word_eeg_embedding)).any() == False:
                                Sentence_EEG_word_level_embeddings.append(word_eeg_embedding)
                                Sentence_word_level_labels.append(EEG_word_level_label)
                            else:
                                Sentence_EEG_word_level_embeddings = []
                                Sentence_word_level_labels = []
                                break

                        for word_label in Sentence_word_level_labels:
                            EEG_word_level_labels.append(word_label)
                        for word_embedding in Sentence_EEG_word_level_embeddings:
                            EEG_word_level_embeddings.append(word_embedding)

        return EEG_word_level_embeddings, EEG_word_level_labels


    def pre_training_NER_encoding(self, pre_train_model, loader, device, vector_size, inputs):
        #NOTE if adding version 5 model, will need to include the input to the model i.e. will need if state to determine the model.

        pre_train_model.to(device)
        pre_train_model.eval()

        aligned_EEG = torch.empty((0, 7, vector_size)).to(device)
        aligned_NE = torch.empty((0, 7, vector_size)).to(device)
        aligned_y = torch.empty((0, 3)).to(device)

        if inputs == 'EEG+Text':
            for batch in loader:
                batch_EEG, batch_NE, batch_y = batch
                batch_EEG,batch_NE, batch_NE, batch_y = batch_EEG.to(device), batch_NE, batch_y.to(device)
                aligned_EEG_outputs = pre_train_model(batch_EEG)
                aligned_EEG = torch.cat((aligned_EEG, aligned_EEG_outputs), dim=0)
                aligned_NE = torch.cat((aligned_NE, batch_NE), dim=0)
                aligned_y = torch.cat((aligned_y, batch_y), dim=0)
        else:
            for batch in loader:
                batch_EEG, batch_y = batch
                batch_EEG, batch_y = batch_EEG.to(device), batch_y.to(device)
                aligned_EEG_outputs = pre_train_model(batch_EEG)
                aligned_EEG = torch.cat((aligned_EEG, aligned_EEG_outputs), dim=0)
                aligned_y = torch.cat((aligned_y, batch_y), dim=0)

        return aligned_EEG, aligned_NE, aligned_y









