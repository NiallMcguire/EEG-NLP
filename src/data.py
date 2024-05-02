import pickle

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


