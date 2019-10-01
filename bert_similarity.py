import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class bert_semantic:
    def vectorize(self,sentence):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        marked_text = "[CLS] " + sentence + " [SEP]"

        #print(marked_text)
        tokenized_text = tokenizer.tokenize(marked_text)
        #print(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        #print(segments_ids)
        segments_tensors = torch.tensor([segments_ids])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        #print(tokens_tensor)
        model = BertModel.from_pretrained('bert-base-uncased')
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
        sentence_embedding = torch.mean(encoded_layers[11], 1)
        #print(sentence_embedding)
        return sentence_embedding

    """def trained_sentence_vectorizer(sentence):
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained('./directory/to/save/')
        marked_text = "[CLS] " + sentence + " [SEP]"

        print(marked_text)
        tokenized_text = tokenizer.tokenize(marked_text)
        print(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        print(segments_ids)
        segments_tensors = torch.tensor([segments_ids])
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        print(tokens_tensor)
        model_class = BertModel
        model = model_class.from_pretrained('./directory/to/save/')
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
        sentence_embedding = torch.mean(encoded_layers[11], 1)
        print(sentence_embedding)
        return sentence_embedding
    """
    def predict_similiarity(self,text1, text2):
        vec1 = self.vectorize(text1)
        vec2 = self.vectorize(text2)
        #print(vec1.size)
        #print(vec1.shape)
        return cosine_similarity(vec1, vec2)



if __name__ == '__main__':
    #text= ["I am unable to create new user", "I cannot create new user"]
    #text = ["What's in a name? That which we call a rose" ,"by any other name would smell as sweet"]
    #text = ["An example of pragmatics is how the same word can have different meanings in different settings.", "An example of pragmatics is the study of how people react to different symbols."]
    #text = ["Here is the sentence I want embeddings for.", "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."]
    #text = ["the branch of linguistics concerned with meaning in context, or the meanings of sentences in terms of the speaker's intentions in using them",
    #"the branch of semiotics dealing with the relationships of signs and symbols to their users"]
    text =["I am good.", "I am not bad."]
    print(test_similiarity(text[0], text[1]))
