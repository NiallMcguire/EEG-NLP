import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig

def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model


if __name__ == '__main__':
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model = get_model('castorini/repllama-v1-7b-lora-doc')

    # Define query and document inputs
    query = "What is llama?"
    title = "Llama"
    url = "https://en.wikipedia.org/wiki/Llama"
    document = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
    query_input = tokenizer(f'query: {query}</s>', return_tensors='pt')
    document_input = tokenizer(f'passage: {url} {title} {document}</s>', return_tensors='pt')

    # Run the model forward to compute embeddings and query-document similarity score
    with torch.no_grad():
        # compute query embedding
        query_outputs = model(**query_input)
        query_embedding = query_outputs.last_hidden_state[0][-1]
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

        # compute document embedding
        document_outputs = model(**document_input)
        document_embeddings = document_outputs.last_hidden_state[0][-1]
        document_embeddings = torch.nn.functional.normalize(document_embeddings, p=2, dim=0)

        # compute similarity score
        score = torch.dot(query_embedding, document_embeddings)
        print(score)
