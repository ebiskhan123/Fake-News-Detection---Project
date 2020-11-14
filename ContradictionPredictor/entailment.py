from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Input, Embedding, LSTM, Merge, Dense,Concatenate,merge, multiply
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras


class ENTAILMENT():

    def init(self) : 
        S = Sentence_ent()
        S.load_dep()
        self.sentence = S
        print(self.sentence)

    def load_model_malstm(self):
        self.malstm_model = keras.models.load_model('MALSTM.h5')

    def get_parameters(self,sentence_A_processed,sentence_B_processed):
        features_function = K.function([left_input,right_input], [left_output,right_output])
        features = features_function([sentence_A_processed,sentence_B_processed])
        np_feature_difference = np.array(features)
        left_matrix = np_feature_difference[0]
        right_matrix = np_feature_difference[1]
        diff_features= np.abs(np.subtract(left_matrix, right_matrix))
        prod = np.multiply(left_matrix,right_matrix)
        self.extracted_features = np.concatenate([diff_features,prod], axis = 1)

    def load_model_entailment(self):
        self.dense_model = keras.models.load_model('CONTRADICTION_PREDICTION.h5')

    def entailment_prediction(self):
        pred_val =np.argmax(self.dense_model.predict(self.extracted_features))
        return pred_val

    def predict_value(self, A, B):
#         "CONTRADICTION" : 0 , "ENTAILMENT" : 1 ,"NEUTRAL" : 2
        S = self.sentence

        S.sentence = A
        A = S.preprocess()
        S.sentence = B
        B = S.preprocess()

        sentence_A_processed = A
        sentence_B_processed = B 
        self.load_model_malstm()
        self.get_parameters(sentence_A_processed,sentence_B_processed)
        self.load_model_entailment()
        entailment = self.entailment_prediction()
        return entailment

class Sentence_ent():
    
    def __init__(self):
        self.vocab_location = 'sick_vocab.pickle'
        self.embeddings_location = 'sick_embedding.pickle'
        # self.word2vec_location = '/Users/ebby/Documents/Fake news/MALSTM/GoogleNews-vectors-negative300.bin'
        
    def preprocess(self):
        text = self.sentence
        text = str(text)
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.split()
        self.word_list = text
        self.to_index_and_pad()
        
        return self.index_rep
        
    def load_dep(self) : 
        file = open(self.vocab_location,"rb")
        self.vocabulary = pickle.load(file)
        file.close()
        file = open(self.embeddings_location, "rb")
        self.embeddings = pickle.load(file)
        file.close()
        print("Dependencies loaded!")
        
    def to_index_and_pad(self) : 
        
        self.index_rep = []
        for word in self.word_list : 
            if word in self.vocabulary : 
                self.index_rep.append(vocabulary[word])
        self.index_rep = pad_sequences([self.index_rep], maxlen=26)


A = "the dog was barking"
B = "the dog was running in the park"

#

a = Sentence_ent()
a.sentence = A
print(a.preprocess())