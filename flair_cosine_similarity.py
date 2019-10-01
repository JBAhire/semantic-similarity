from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Sentence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flair.data import Sentence
import torch as torch
from tqdm import tqdm

class flair_semantic:
    def vectorise(self, text):
        flair_forward  = FlairEmbeddings('news-forward')
        flair_backward = FlairEmbeddings('news-backward')
        stacked_embeddings = StackedEmbeddings( embeddings = [ 
                                                            flair_forward, 
                                                            flair_backward
                                                            ])
        sentence = Sentence(text)
        stacked_embeddings.embed(sentence)
        for token in sentence:
            #print(token.embedding)
            #print(type(token.embedding))
            z = token.embedding.size()[0]
        s = torch.zeros(0,z)
        w = torch.zeros(0,z)   
        w = torch.cat((w,token.embedding.view(-1,z)),0)
        s = torch.cat((s, w.mean(dim = 0).view(-1, z)),0)
        return s

    def predict_similarity(self, text1, text2):
        vector1 = self.vectorise(text1)
        vector2 = self.vectorise(text2)
        similarity = cosine_similarity(vector1, vector2)
        #print("cosine similarity between this two sentences is : "+ str(similarity))
        return similarity


if __name__ == "__main__":
    text1 = "I am good."
    text2 = "I am not bad." 
    flair = flair_semantic()
    flair.predict_similarity(text1, text2)

            