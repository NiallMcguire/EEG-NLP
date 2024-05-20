import torch
import numpy as np
import os
import gc

# Function to save pairs in batches
def save_pairs_in_batches(X, train_NE_expanded, save_path, batch_size=100):
    eeg_pairs_list = []
    bert_pairs_list = []
    labels_list = []

    # Process pairs in batches
    for i in range(0, len(X), batch_size):
        # Create positive pairs
        for j in range(batch_size):
            if i + j >= len(X):
                break
            eeg_pairs_list.append(X[i + j])
            bert_pairs_list.append(train_NE_expanded[i + j])
            labels_list.append(1)

        # Create negative pairs
        for j in range(batch_size):
            if i + j >= len(X):
                break
            for k in range(batch_size):
                if k != j and i + k < len(train_NE_expanded):
                    eeg_pairs_list.append(X[i + j])
                    bert_pairs_list.append(train_NE_expanded[i + k])
                    labels_list.append(0)

        # Convert lists to tensors
        eeg_pairs = torch.tensor(eeg_pairs_list, dtype=torch.float32)
        bert_pairs = torch.tensor(bert_pairs_list, dtype=torch.float32)
        labels = torch.tensor(labels_list, dtype=torch.float32)

        # Save tensors
        torch.save(eeg_pairs, os.path.join(save_path, f"Pre_NER_eeg_pairs_{i}.pt"))
        torch.save(bert_pairs, os.path.join(save_path, f"Pre_NER_bert_pairs_{i}.pt"))
        torch.save(labels, os.path.join(save_path, f"Pre_NER_labels_{i}.pt"))

        # Clear lists and free memory
        eeg_pairs_list.clear()
        bert_pairs_list.clear()
        labels_list.clear()
        del eeg_pairs, bert_pairs, labels
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == "__main__":
    import sys
    sys.path.append('/users/gxb18167/EEG-NLP/')
    import data
    import utils
    train_path = r"/users/gxb18167/EEG-NLP/NER.pkl"
    save_path = r"/users/gxb18167/EEG-NLP/"
    vector_size = 768

    d = data.Data()
    util = utils.Utils()

    train_NE, train_EEG_segments, train_Classes = d.NER_read_custom_files(train_path)
    test_size = 0.2

    ner_bert = utils.NER_BERT()
    train_NE_embedded = ner_bert.get_embeddings(train_NE)
    train_NE_expanded = util.NER_expanded_NER_list(train_EEG_segments, train_NE_embedded, vector_size)
    train_NE_expanded = np.array(train_NE_expanded)

    X, y = util.NER_padding_x_y(train_EEG_segments, train_Classes)
    X = np.array(X)
    X = util.NER_reshape_data(X)
    y_categorical = util.encode_labels(y)

    # Save pairs in batches
    save_pairs_in_batches(X, train_NE_expanded, save_path)
