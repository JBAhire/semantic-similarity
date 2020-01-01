from flair_cosine_similarity import flair_semantic
from elmo_cosine_similarity import elmo_semantic
from spacy_cosine_similarity import spacy_semantic
from common import similarity_test
from bert_similarity import bert_semantic
import pandas as pd

flair_similarity = []
elmo_similarity = []
bert_similarity = []
spacy_similarity = []

flair = flair_semantic()
elmo = elmo_semantic()
bert = bert_semantic()
spacy = spacy_semantic()

common = similarity_test()
data = pd.read_csv('data/sick_dev.csv')
sent_1_list = list(data.sent_1)

data = common.normalize(data, ['sim'])
sim_list = list(data.sim)

sent_2_list = list(data.sent_2)
min_list1 = sent_1_list[:10]
min_list2 = sent_2_list[:10]
min_list3 = sim_list[:10]

for text1, text2, text3 in zip(sent_1_list, sent_2_list, sim_list):
    f_similarity = flair.predict_similarity(text1, text2)
    e_similarity = elmo.predict_similarity(text1, text2)
    b_similarity = bert.predict_similiarity(text1, text2)
    s_similarity = spacy.predict_similarity(text1, text2)
    flair_similarity.append(f_similarity)
    elmo_similarity.append(e_similarity)
    bert_similarity.append(b_similarity)
    spacy_similarity.append(s_similarity)
    print('Similarity measures are as follows:')
    print (' elmo: '+ str(e_similarity) + '|' + ' flair: ' + str(f_similarity) + '|' + ' bert: ' + str(b_similarity) + '|' + ' Spacy: ' +'[[' +str(s_similarity)+']]'+ '|' + ' human predicted: ' +'[[' +str(text3)+']]')
    print("======================================================================================")


series1 = pd.Series('flair_similarity')
series2 = pd.Series('elmo_similarity')
series3 = pd.Series('bert_similarity')
series4 = pd.Series('spacy_similarity')
data['flair_similarity'] = series1.values
data['elmo_similarity'] = series2.values
data['bert_similarity'] = series3.values
data['spacy_similarity'] = series4.values

data.tocsv('result/result.csv')



