import spacy
import pandas as pd 
from common import similarity_test

nlp = spacy.load('en_core_web_md')
'''
common = similarity_test()
data = pd.read_csv('data/sick_dev.csv')
sent_1_list = list(data.sent_1)

data = common.normalize(data, ['sim'])
sim_list = list(data.sim)

sent_2_list = list(data.sent_2)
min_list1 = sent_1_list[:10]
min_list2 = sent_2_list[:10]
min_list3 = sim_list[:10]

for text1, text2 in zip(min_list1, min_list2):
    search_doc = nlp(text1)
    main_doc = nlp(text2)
    search_doc_no_stop_words = nlp(' '.join([str(t) for t in search_doc if not t.is_stop]))
    main_doc_no_stop_words = nlp(' '.join([str(t) for t in main_doc if not t.is_stop]))
    print(search_doc_no_stop_words.similarity(main_doc_no_stop_words))
    print('========================================================================')

'''
class spacy_semantic:
    def vectorise(self, sentence):
        text = nlp(sentence)
        vector = nlp(' '.join([str(t) for t in text if not t.is_stop]))
        return vector

    def predict_similarity(self, text1, text2):
        vector1 = self.vectorise(text1)
        vector2 = self.vectorise(text2)
        return vector1.similarity(vector2)


if __name__ == "__main__":
    pass

        

