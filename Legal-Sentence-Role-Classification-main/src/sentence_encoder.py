"""
This File contains methods to encode text into embeddings
@author: Philipp
"""
import numpy as np
# Imports sentence bert stuf
from sentence_transformers import SentenceTransformer


#TODO this is not tested yet and requires tensorflow_hub to be installed
def usc_embeddings(df_sentences):
    raise NotImplementedError("usc_embeddings requires tensorflow_hub (not installed). Use sentence_bert_embeddings instead.")

#sentence bert
def sentence_bert_embeddings(df_sentences,model=0):
    df_sentences['embedding']=np.nan
    df_sentences['embedding']=df_sentences['embedding'].astype(object)
    sbert_model=0

    if model == 1:
        sbert_model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
    else:
        sbert_model = SentenceTransformer('all-mpnet-base-v2')


    for index,row in df_sentences.iterrows():
        sentence =row['text']
        embedding =sbert_model.encode(sentence).tolist()
        df_sentences.at[index,'embedding']=embedding

    return df_sentences
