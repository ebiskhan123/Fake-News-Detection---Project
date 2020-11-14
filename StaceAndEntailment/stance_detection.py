from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string 
import math 
import numpy as np
import pandas as pd
from textblob import TextBlob
import keras

class Sentence :

    def __init__(self, sent) :
        self.sentence = sent
        self.is_tokenized = False
        self.token_list = []
        self.tf = []

    def remove_punct(self) :
        table = str.maketrans('', '', string.punctuation)
        self.sentence = self.sentence.translate(table)

    def remove_num(self) :
        table = str.maketrans('', '', '1234567890')
        self.sentence = self.sentence.translate(table)

    def tokenize(self):
        self.token_list = self.sentence.split()
        self.is_tokenized = True

    def to_lower(self): 
        if self.is_tokenized : 
            for i in range(len(self.token_list)) : 
                self.token_list[i] = self.token_list[i].lower()

    def remove_stop(self) : 
        if self.is_tokenized : 
            stop_words = set(stopwords.words('english'))
            doc_stop = []
            for word in self.token_list :
                if(not word in stop_words) : 
                    doc_stop.append(word)
            self.token_list = doc_stop

    def stem(self): 
        if self.is_tokenized : 
            porter = PorterStemmer()
            for i in range(len(self.token_list)) : 
                self.token_list[i] = porter.stem(self.token_list[i])

    def get_sent(self) :
        return self.sentence

    def get_token_list(self) : 
        return self.token_list
    
    def preprocess(self) : 
        self.remove_punct()
        self.remove_num()
        self.tokenize()
        self.to_lower()
        self.remove_stop()
        self.stem()

    def tf_calc(self, base) :
        tf = []
        for i in range(len(base)) : 
            tf.append(0);
        
        count = []
        for word in self.token_list : 
            c = 0
            for x in self.token_list : 
                if word == x : 
                    c += 1
            count.append(c)
                
        sent_length = len(self.token_list)
        for i in range(len(self.token_list)) : 
            if self.token_list[i] in base :
                index = base.index(self.token_list[i]) 
                tf[index] = count[i] / sent_length
        self.tf = tf

    def get_tf(self) : 
        return self.tf
    

class Corpus(Sentence) : 

    def __init__(self, corpus) : 
        self.corpus = corpus 
        self.corpus_tokens = corpus
        self.is_tokenized = False
        self.idf = []
        self.n = -1
        self.base = []
        self.counted = False
        self.filterd = False
        
    
    def count_words(self) : 
        print("-------Corpus.count_words-------")
        if self.is_tokenized : 
            set_words = set()
            for i in self.corpus_tokens : 
                for word in i : 
                    set_words.add(word)
            dict_words = {}
            for i in set_words : 
                dict_words[i] = 0
            for i in self.corpus_tokens : 
                for word in i : 
                    dict_words[word] += 1
            self.set_words = set_words
            self.dict_words = dict_words

    def contains(self, word) : 
        if self.is_tokenized : 
            count = 0
            for i in self.corpus_tokens :
                if word in i : 
                    count += 1
            return count  

    def filter_top_n(self) : 
        print("-------Corpus.filter_top_n-------")
        list_words = [] 
        for i in self.set_words : 
            list_words.append([self.dict_words[i], i])
        list_words.sort()
        list_words.reverse()
        self.top_n = list_words[0:self.n]

    def preprocess(self) : 
        self.is_tokenized = True
        total = len(self.corpus)
        print("-------Corpus.preprocess-------")
        for i in range(total) : 
            self.sentence = self.corpus[i]
            super().preprocess()
            # print(supre().get_corpus_tokens())
            self.corpus_tokens[i] = super().get_token_list()
            print("-----" + str(i*100 / total) + "-----", end="\r")

    def set_n(self, n) : 
        self.n = n        
    
    def idf_n(self) : 
        if self.counted == False: 
            self.count_words()
            self.counted = True
        if self.filterd == False: 
            self.filter_top_n()
            self.filterd = True
        self.filter_top_n()
        print("-------Corpus.idf_n-------")
        doc_size = len(self.corpus)
        idf = []
        prog = 0
        total = len(self.top_n)
        for i in self.top_n : 
            val = math.log(doc_size / (1 + self.contains(i[1])))
            idf.append(val)
            print("-----" + str(prog*100 / total) + "-----", end="\r")
            prog+= 1
        self.idf = idf

    def tf_n(self) :
        tf = []
        #self.word_base()
        prog = 1
        total = len(self.corpus_tokens)
        for sent in self.corpus_tokens : 
            s = Sentence(sent)
            s.token_list = sent
            s.tf_calc(self.base) 
            tf.append(s.get_tf())
            print("-----" + str(prog*100 / total) + "-----", end="\r")
            prog += 1
        self.tf = tf
    def set_base(self, base) : 
        self.base = base
        
    def get_tf(self) :
        return self.tf 

    def word_base(self) : 
        base = []
        for word in self.top_n : 
            base.append(word[1])
        self.base = base

    def get_base(self) : 
        return self.base

    def get_corpus(self) : 
        return self.corpus
    def get_corpus_tokens(self) : 
        return self.corpus_tokens
    def get_idf(self) : 
        return self.idf


class StanceDetection : 
    
    def __init__(self) : 
        self.model_location_json = "./stance_model_json.json"
        self.model_location_weights = "./stance_weights.h5"
        self.base_bow_location = "./base_bow.npy"
        
    def load(self) : 
        
        # self.base = list(np.load(self.base_bow_location))
        # self.sd_model = keras.models.load_model(self.model_location)
        json_file = open(self.model_location_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        self.sd_model = loaded_model
        
    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
           return 0.0
        else:
           return float(numerator) / denominator

    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)
    
    
    def predict(self,title, body) : 
        
        input_sd = []
        
        t = Sentence(title)
        t.preprocess()
        t.tf_calc(self.base)
        
        b = Sentence(body)
        b.preprocess()
        b.tf_calc(self.base)
        
        input_sd += t.tf
        input_sd += [get_cosine(Counter(t.get_token_list()), Counter(b.get_token_list()))]
        input_sd += b.tf
        
        output_sd = self.sd_model.predict(np.array([input_sd]))
        
        return output_sd
    
    