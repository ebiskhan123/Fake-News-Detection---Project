from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
import datetime
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Input, Embedding, LSTM, Merge,merge,Dense,Dropout
import keras.backend as K
from keras.optimizers import Adadelta,Adam
from keras.callbacks import ModelCheckpoint


# def text_to_word_list(text):
#     ''' Pre process and convert texts to a list of words '''
#     text = str(text)
#     text = text.lower()

#     # Clean the text
#     text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
#     text = re.sub(r"what's", "what is ", text)
#     text = re.sub(r"\'s", " ", text)
#     text = re.sub(r"\'ve", " have ", text)
#     text = re.sub(r"can't", "cannot ", text)
#     text = re.sub(r"n't", " not ", text)
#     text = re.sub(r"i'm", "i am ", text)
#     text = re.sub(r"\'re", " are ", text)
#     text = re.sub(r"\'d", " would ", text)
#     text = re.sub(r"\'ll", " will ", text)
#     text = re.sub(r",", " ", text)
#     text = re.sub(r"\.", " ", text)
#     text = re.sub(r"!", " ! ", text)
#     text = re.sub(r"\/", " ", text)
#     text = re.sub(r"\^", " ^ ", text)
#     text = re.sub(r"\+", " + ", text)
#     text = re.sub(r"\-", " - ", text)
#     text = re.sub(r"\=", " = ", text)
#     text = re.sub(r"'", " ", text)
#     text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
#     text = re.sub(r":", " : ", text)
#     text = re.sub(r" e g ", " eg ", text)
#     text = re.sub(r" b g ", " bg ", text)
#     text = re.sub(r" u s ", " american ", text)
#     text = re.sub(r"\0s", "0", text)
#     text = re.sub(r" 9 11 ", "911", text)
#     text = re.sub(r"e - mail", "email", text)
#     text = re.sub(r"j k", "jk", text)
#     text = re.sub(r"\s{2,}", " ", text)

#     text = text.split()

#     return text


# def PREDICT_CONTRADICTION(test_df):

#     print("Contradiction")
#     train_df = pd.read_csv('./SICK.txt', sep="\t", header=None)
#     train_df.columns = train_df.iloc[0]
#     train_df=train_df.reindex(train_df.index.drop(0))

#     EMBEDDING_FILE = './GoogleNews-vectors-negative300.bin'
#     word2vec_location = './GoogleNews-vectors-negative300.bin'
    
#     stops = set(stopwords.words('english'))
#     #test_df = #pd.read_csv('test file location', sep="\t", header=None)
#     vocabulary = dict()
#     inverse_vocabulary = ['<unk>']  
#     word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
#     questions_cols = ['sentence_A' , 'sentence_B']

#     print("word2vec loaded!")

#     for dataset in [train_df,test_df]:
#         for index, row in dataset.iterrows():
#             for question in questions_cols:

#                 q2n = []  
#                 for word in text_to_word_list(row[question]):

#                     # Check for unwanted words
#                     if word in stops and word not in word2vec.vocab:
#                         continue

#                     if word not in vocabulary:
#                         vocabulary[word] = len(inverse_vocabulary)
#                         q2n.append(len(inverse_vocabulary))
#                         inverse_vocabulary.append(word)
#                         #print(word)
#                     else:
#                         q2n.append(vocabulary[word])
#                 dataset.set_value(index, question, q2n)

#     embedding_dim = 300
#     embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
#     embeddings[0] = 0 
#     for word, index in vocabulary.items():
#         if word in word2vec.vocab:
#             embeddings[index] = word2vec.word_vec(word)

#     del word2vec
    
#     print("Vocab created!")
#     max_seq_length = max(train_df.sentence_A.map(lambda x: len(x)).max(),
#                          train_df.sentence_B.map(lambda x: len(x)).max(),
#                          test_df.sentence_A.map(lambda x: len(x)).max(),
#                          test_df.sentence_B.map(lambda x: len(x)).max())

#     X_train = train_df[questions_cols]
#     Y_train = train_df['relatedness_score']
#     X_test = test_df[questions_cols]
#     X_train = {'left': X_train.sentence_A, 'right': X_train.sentence_B}
#     X_test = {'left': test_df.sentence_A, 'right': test_df.sentence_B}
#     Y_train = Y_train.values
#     for dataset, side in itertools.product([X_train], ['left', 'right']):
#         dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
        
#     X_train1 = train_df[questions_cols]
#     Y_train1 = train_df['entailment_label']
#     X_train1 = {'left': X_train1.sentence_A, 'right': X_train1.sentence_B}
#     Y_train1 = Y_train1.values
#     X_test1 = test_df[questions_cols]
#     X_test1 = {'left': test_df.sentence_A, 'right': test_df.sentence_B}


#     for dataset, side in itertools.product([X_train1,X_test1], ['left', 'right']):
#         dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
        
#     n_hidden = 50
#     gradient_clipping_norm = 1
#     batch_size = 64
#     n_epoch = 25
#     print("model 1")
#     def exponent_neg_manhattan_distance(left, right):
#         return (K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True)))*4.0 +1

#     left_input = Input(shape=(max_seq_length,), dtype='int32')
#     right_input = Input(shape=(max_seq_length,), dtype='int32')
#     embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)
#     encoded_left = embedding_layer(left_input)
#     encoded_right = embedding_layer(right_input)
#     shared_lstm = LSTM(n_hidden)
#     left_output = shared_lstm(encoded_left)
#     right_output = shared_lstm(encoded_right)
#     malstm_distance = Merge(mode=lambda x: ((K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)))*4.0 +1), output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
#     malstm = Model([left_input, right_input], [malstm_distance])
#     optimizer = Adadelta(clipnorm=gradient_clipping_norm)
#     malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
#     training_start_time = time()
#     malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch)
#     print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

#     features_function = K.function([left_input,right_input], [left_output,right_output])
#     features = features_function([X_train1['left'],X_train1['right']])
#     np_feature_difference = np.array(features)
#     left_matrix = np_feature_difference[0]
#     right_matrix = np_feature_difference[1]
#     diff_features= np.abs(np.subtract(left_matrix, right_matrix))
#     prod = np.multiply(left_matrix,right_matrix)
#     extracted_features = np.concatenate([diff_features,prod], axis = 1)
    
#     features = features_function([X_test1['left'],X_test1['right']])
#     np_feature_difference = np.array(features)
#     left_matrix = np_feature_difference[0]
#     right_matrix = np_feature_difference[1]
#     diff_features= np.abs(np.subtract(left_matrix, right_matrix))
#     prod = np.multiply(left_matrix,right_matrix)
#     extracted_features_test = np.concatenate([diff_features,prod], axis = 1)
    
#     K.clear_session()

#     print("model 2")
#     model = Sequential()
#     model.add(Dense(200, input_shape=(100,),kernel_initializer='he_normal', activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(100, kernel_initializer='he_normal', activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(50, kernel_initializer='he_normal', activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(3, activation='softmax'))
#     model.compile(Adam(lr = 0.001),'categorical_crossentropy', metrics=['accuracy'])
#     y_TRAIN_cat = pd.get_dummies(Y_train1)
#     model.fit(extracted_features, y_TRAIN_cat.values,batch_size=16,epochs=10,verbose=1)
#     print("Entailmet model finished training")
    
#     pred_val = model.predict_classes(extracted_features_test)
#     return pred_val

class MALSTM : 
    
    def __init__(self, model_json, model_weights, vocab) :
        
        sess = tf.Session()
        K.set_session(sess)
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(model_weights)
#         self.model.summary()
        self.vocab = pickle.load(open(vocab, "rb"))

    def output(self, A):
        
        A = self.preprocess(A)                
        output = self.model.predict(A)
        return output
    
    def preprocess(self, text):
        word_list = self.text_to_word_list(text)
        ind = []
        for word in word_list : 
            if(word in self.vocab) : 
                ind.append(self.vocab[word])
        ind = pad_sequences([ind], maxlen=26)
        return ind 
    
    def clear(self) : 
        K.set_session(None)
        del self.model
        
    def text_to_word_list(self, text):
        ''' Pre process and convert texts to a list of words '''
        text = str(text)
        text = text.lower()

        # Clean the text
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

        return text

class Entailment : 
    
    def __init__(self, ent_json, ent_weights) : 
        
        model2 = Sequential()
        model2.add(Dense(200, input_shape=(100,),kernel_initializer='he_normal', activation='relu'))
        model2.add(Dropout(0.2))
        model2.add(Dense(100, kernel_initializer='he_normal', activation='relu'))
        model2.add(Dropout(0.2))
        model2.add(Dense(50, kernel_initializer='he_normal', activation='relu'))
        model2.add(Dropout(0.2))
        model2.add(Dense(3, activation='softmax'))
        model2.compile(Adam(lr = 0.001),'categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model2
        self.model.load_weights(ent_weights)
        
    def clear(self) : 
        K.set_session(None)   
        del self.model
    
    def predict(self, A, B) : 
        left_matrix = A
        right_matrix = B
        diff_features= np.abs(np.subtract(left_matrix, right_matrix))
        prod = np.multiply(left_matrix,right_matrix)
        extracted_features = np.concatenate([diff_features,prod], axis = 1)
        output = self.model.predict(extracted_features)
        return np.argmax(output)


if __name__ == '__main__':

    test_df = pd.read_csv('ent_data.csv', sep=",", header=None)
    test_df.columns = test_df.iloc[0]
    test_df=test_df.reindex(test_df.index.drop(0))
    
    # pred_val = PREDICT_CONTRADICTION(test_df)
    # print(type(pred_val))
    # print(pred_val)
    
    
    pd.DataFrame(pred_val).to_csv('result.csv', sep=',')

  