import allennlp
import tensorflow as tf
import tensorflow_hub as hub 
import nltk
from sklearn.metrics.pairwise import cosine_similarity

class elmo_semantic:     
    def vectorise(self, sentence):
        self.elmo = hub.Module("module/module_elmo2/", trainable=True)
        embeddings = self.elmo([sentence], signature="default", as_dict=True)["elmo"]
        print(embeddings.shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
                            # return average of ELMo features
            vects = sess.run(tf.reduce_mean(embeddings,1))
        return vects

    def predict_similarity(self, text1, text2):
        vector1 = self.vectorise(text1)
        vector2 = self.vectorise(text2)
        similarity = cosine_similarity(vector1, vector2)
        #print("cosine similarity between this two sentences is : "+ str(similarity))
        return similarity

if __name__ == "__main__":
    text1 = "I am good."
    text2 = "I am not bad." 
    elmo = elmo_semantic()
    elmo.predict_similarity(text1, text2)
