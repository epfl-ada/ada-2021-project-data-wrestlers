# Code adapted from https://towardsdatascience.com/nlp-preprocessing-and-latent-dirichlet-allocation-lda-topic-modeling-with-gensim-713d516c6c7d
import contractions as ctr
import demoji
import string
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords as stpw
import re

class Preprocessor():
    def __init__(self, lowercase=True, emoji=True, contractions=True, punctuation=True, numbers=True, stopwords=True, stopwords_to_keep=['no', 'not'], lemmatization=True, min_word_length=3):
        assert min_word_length>=0, f'min_word_length must be >= 0.'
        self.lowercase = lowercase
        self.emoji = emoji
        self.contractions = contractions
        self.punctuation = punctuation
        self.numbers = numbers
        self.stopwords = stopwords
        self.stopwords_to_keep = stopwords_to_keep
        self.lemmatization = lemmatization
        self.min_word_length = min_word_length
    
    def transform(self, text):
        if self.lowercase:
            text = text.lower()
            # text = text.apply(lambda x: ' '.join([w.lower() for w in x.split()]))    
        if self.emoji:
            text = demoji.replace(text, '')
            # text = text.apply(lambda x: demoji.replace(x, ""))    
        if self.contractions  :
            text = ' '.join([ctr.fix(word) for word in text.split()])
            # text = text.apply(lambda x: ' '.join([ctr.fix(word) for word in x.split()]))
        if self.punctuation:
            text = ''.join([i for i in text if i not in string.punctuation])
            # text = text.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))    
        if self.numbers:
            text = ' '.join(re.sub("[^a-zA-Z]+", " ", text).split())
            # text = text.apply(lambda x: ' '.join(re.sub("[^a-zA-Z]+", " ", x).split()))
        if self.stopwords:
            stop_words = [sw for sw in stpw.words('english') if sw not in self.stopwords_to_keep]
            text = ' '.join([w for w in text.split() if w not in stop_words])
            # text = text.apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
        if self.lemmatization:
            text = ' '.join([WordNetLemmatizer().lemmatize(w) for w in text.split()])
            # text = text.apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(w) for w in x.split()]))
        if self.min_word_length > 0:
            text = ' '.join([w.strip() for w in text.split() if len(w.strip()) >= self.min_word_length])
            # text = text.apply(lambda x: ' '.join([w.strip() for w in x.split() if len(w.strip()) >= self.min_word_length]))
        return text
    