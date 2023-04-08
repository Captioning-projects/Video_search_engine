from sentence_transformers import SentenceTransformer, util
import numpy as np

sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/stsb-roberta-large')
embeddings = model.encode(sentences)
print(embeddings)
 