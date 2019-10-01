# What is Semantic Similarity?
semantic similarity is implementation of a technology called text embedding. One of the most useful, new technologies for natural language processing, text embedding transforms words into a numerical representation (vectors) that approximates the conceptual distance of word meaning.

Many NLP applications need to compute the similarity in meaning between two short texts. Search engines, for example, need to model the relevance of a document to a query, beyond the overlap in words between the two. Similarly, question-and-answer sites such as Quora need to determine whether a question has already been asked before. This type of text similarity is often computed by first embedding the two short texts and then calculating the cosine similarity between them. 

# What embeddings we're using?
We're using following embeddings:
1. BERT
2. Elmo
3. Spacy
4. W2V

# Requirements
1. Python (3.0 and above)
2. Flask
3. TensorFlow
4. Download Bert pre-trained model from here(https://github.com/google-research/bert#pre-trained-models)
5. AllenNLP
6. Spcay

# Steps
``` pip install -r requirements.py ```

