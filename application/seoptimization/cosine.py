import re
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer 
from sentence_transformers import SentenceTransformer
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))


def Doct2vect(documents):
    """
    used to create embeddings and find simmilarities between the first index of list and other indexes of the list.
    : param documnets : a list for calculating the consine similarity of first element with others

    reference : https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630
    """

    documents_df=pd.DataFrame(documents,columns=['documents'])
    documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words))
    
    tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(documents_df.documents_cleaned)]
    model_d2v = Doc2Vec(vector_size=100,alpha=0.025, min_count=1)
    
    model_d2v.build_vocab(tagged_data)

    for epoch in range(100):
        model_d2v.train(tagged_data,
                    total_examples=model_d2v.corpus_count,
                    epochs=model_d2v.epochs)
        
    document_embeddings=np.zeros((documents_df.shape[0],100))

    for i in range(len(document_embeddings)):
        document_embeddings[i]=model_d2v.dv[i]
        
    pairwise_similarities= np.array(cosine_similarity(document_embeddings)[0])
    print(pairwise_similarities)
    normalizedData = (pairwise_similarities-np.min(pairwise_similarities))/(np.max(pairwise_similarities)-np.min(pairwise_similarities))
    print(normalizedData)
    return normalizedData

# def Semantic_sim(documents):
#     """
#     used to create embeddings and find simmilarities between the first index of list and other indexes of the list.
#     : param documnets : a list for calculating the consine similarity of first element with others

#     reference : https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630
#     """

#     documents_df=pd.DataFrame(documents,columns=['documents'])
#     documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words))
    
#     model = SentenceTransformer('sentence-transformers/stsb-roberta-large')
#     embeddings = model.encode(documents_df.documents_cleaned)
#     # print(embeddings)
       
#     pairwise_similarities= np.array(cosine_similarity(embeddings)[0])
#     normalizedData = (pairwise_similarities-np.min(pairwise_similarities))/(np.max(pairwise_similarities)-np.min(pairwise_similarities))
#     # print(normalizedData)
#     return normalizedData

# from keras.preprocessing.text import Tokenizer

# from keras_preprocessing.sequence import pad_sequences
# def glove_em(documents):

#     tokenizer=Tokenizer()
    
#     documents_df=pd.DataFrame(documents,columns=['documents'])
#     documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words))
    
#     tokenizer.fit_on_texts(documents_df.documents_cleaned)
#     tokenized_documents=tokenizer.texts_to_sequences(documents_df.documents_cleaned)
#     tokenized_paded_documents=pad_sequences(tokenized_documents,maxlen=64,padding='post')
#     tfidfvectoriser=TfidfVectorizer()
#     tfidfvectoriser.fit(documents_df.documents_cleaned)
#     words=tfidfvectoriser.get_feature_names()
#     tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)
#     vocab_size=len(tokenizer.word_index)+1

#     # reading Glove word embeddings into a dictionary with "word" as key and values as word vectors
#     embeddings_index = dict()

#     with open('glove.6B.100d.txt', encoding="utf8") as file:
#         for line in file:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             embeddings_index[word] = coefs
        
#     # creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
#     embedding_matrix=np.zeros((vocab_size,100))

#     for word,i in tokenizer.word_index.items():
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
            
#     # calculating average of word vectors of a document weighted by tf-idf
#     document_embeddings=np.zeros((len(tokenized_paded_documents),100))
    

#     # instead of creating document-word embeddings, directly creating document embeddings
#     for i in range(documents_df.shape[0]):
#         for j in range(len(words)):
#             document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors[i][j]
            

#     pairwise_similarities=cosine_similarity(document_embeddings)
#     normalizedData = (pairwise_similarities-np.min(pairwise_similarities))/(np.max(pairwise_similarities)-np.min(pairwise_similarities))
#     print(normalizedData)
#     return normalizedData

def Semantic_sim(documents):
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    documents_df = pd.DataFrame(documents, columns=['documents'])
    documents_df['documents_cleaned'] = documents_df.documents.apply(lambda x: " ".join(re.sub(
        r'[^a-zA-Z]', ' ', w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]', ' ', w).lower() not in stop_words))
    document_embeddings = sbert_model.encode(documents_df['documents_cleaned'])
    pairwise_similarities = np.array(cosine_similarity(document_embeddings)[0])
    normalizedData = (pairwise_similarities-np.min(pairwise_similarities)) / \
        (np.max(pairwise_similarities)-np.min(pairwise_similarities))
    return normalizedData
    


if __name__ == "__main__":
    #  # Sample corpus
    # documents = ['Machine learning is the study of computer algorithms that improve automatically through experience.\
    # Machine learning algorithms build a mathematical model based on sample data, known as training data.\
    # The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
    # where no fully satisfactory algorithm is available.',
    # 'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
    # The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
    # 'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
    # It involves computers learning from data provided so that they carry out certain tasks.',
    # 'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal"\
    # or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
    # 'Software engineering is the systematic application of engineering approaches to the development of software.\
    # Software engineering is a computing discipline.',
    # 'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
    # about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
    # Developing a machine learning application is more iterative and explorative process than software engineering.'
    # ]
    documents = ['cooking video', 'a baby is playing with a ball', 'a person is slicing a potato', 'a man is riding a motorcycle', 'someone is stirring rice in a pot', 'a woman is cooking something', 'a person is cutting a']

    print(Semantic_sim(documents))
    # print(Doct2vect(documents))
    