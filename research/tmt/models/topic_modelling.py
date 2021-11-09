import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

class ClusterTopicModelling:
    def __init__(self, vectorizer, model):
        if type(vectorizer) == CountVectorizer and type(model) == LatentDirichletAllocation:
            self.name = 'cluster_lda'
        elif type(vectorizer) == TfidfVectorizer and type(model) == NMF:
            self.name = 'cluster_nmf'
        else:
            raise AttributeError(f'Unrecognized combination of model and vectorizer.')
        self.vectorizer = vectorizer
        self.model = model
    
    def get_topics_words(self, n_top):
        '''Y is output of transform, we translate to lists of (features, weights)
        for each cluster. this represents the cluster's topic.
        (https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py)'''
        topic_words = []
        feature_names = self.vectorizer.get_feature_names_out()
        for topic in self.model.components_:
            top_features_ind = topic.argsort()[: -n_top - 1 : -1]
            top_words = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            topic_words.append([*zip(top_words, weights)])
        return topic_words

    def get_token_count(self, documents):
        token_count = self.vectorizer.fit_transform(documents)
        if token_count.shape[1] == 1:
            token_count = token_count.reshape(-1, 1)
        elif token_count.shape[0] == 1:
            token_count = token_count.reshape(1, -1)
        return token_count

    def fit(self, documents):
        self.model.fit(self.get_token_count(documents))
        
    def fit(self, documents):
        return self.model.transform(self.get_token_count(documents))

    def fit(self, documents):
        return self.model.fit_transform(self.get_token_count(documents))
