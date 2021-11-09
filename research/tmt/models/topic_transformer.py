import os
import pickle
import json
import numpy as np

class TopicTransformer():
    def __init__(self, encoder, dim_reduction, clustering, topic_modelling, agg='mean'):
        '''TopicModellingTransformer constructor
        Inputs:
        -------
        encoder: [encoder.PretrainedTransformerEncoder]
            Document Encoder
        dim_reduction: [dim_reduction.Tsne|| dim_reduction.Pca || dim_reduction.Umap]
            Dimensionality reduction model
        clustering: [clustering.KMeans || clustering.Hdbscan]
            Clustering model
        topic_modelling: [topic_modelling.ClusterTopicModelling]
            Topic modelling algorithm to be applied to each cluster
        agg: str
            String representation of the aggregation to apply after the transformer encoder (reduce sequence dim)
        '''
        self.encoder_name = encoder.name
        self.dim_reduction_name = dim_reduction.name
        self.clustering_name = clustering.name
        self.topic_modelling_name = topic_modelling.name
        self.encoder = encoder # encode api
        self.dim_reduction = dim_reduction # fit, transform, fit_transform api
        self.clustering = clustering # fit, transform, fit_transform api
        self.topic_modelling = topic_modelling # fit, transform, fit_transform api
        self.agg = agg

    @classmethod
    def from_config(cls, config):
        # encoder
        from .encoder import PretrainedTransformerEncoder
        encoder = PretrainedTransformerEncoder(config['encoder']['name'])
        # dim reduction
        if config['dim_reduction']['algorithm'] == 'pca':
            from .dim_reduction import Pca
            dim_reduction = Pca(**config['dim_reduction']['kwargs'])
        elif config['dim_reduction']['algorithm'] == 'tsne':
            from .dim_reduction import Tsne
            dim_reduction = Tsne(**config['dim_reduction']['kwargs'])
        elif config['dim_reduction']['algorithm'] == 'umap':
            from .dim_reduction import Umap
            dim_reduction = Umap(**config['dim_reduction']['kwargs'])
        else:
            raise AttributeError('Uknown dim_reduction algorithm: {}'.format(config['dim_reduction']['algorithm']))
        # clustering
        if config['clustering']['algorithm'] == 'kmeans':
            from .clustering import KMeans
            clustering = KMeans(**config['clustering']['kwargs'])
        elif config['clustering']['algorithm'] == 'hdbscan':
            from .clustering import Hdbscan
            clustering = Hdbscan(**config['clustering']['kwargs'])
        else:
            raise AttributeError('Uknown clustering algorithm: {}'.format(config['clustering']['algorithm']))
        # topic modelling
        from .topic_modelling import ClusterTopicModelling
        from .utils import Preprocessor
        preprocessor = Preprocessor(**config['topic_modelling']['preprocessing_kwargs']).transform
        vectorizer_kwargs = config['topic_modelling']['vectorizer_kwargs']
        vectorizer_kwargs['preprocessor'] = preprocessor
        if config['topic_modelling']['algorithm'] == 'cluster_lda':
            from sklearn.decomposition import LatentDirichletAllocation
            from sklearn.feature_extraction.text import CountVectorizer            
            model = LatentDirichletAllocation(**config['topic_modelling']['kwargs'])            
            vectorizer = CountVectorizer(**vectorizer_kwargs)        
        elif config['topic_modelling']['algorithm'] == 'cluster_nmf':
            from sklearn.decomposition import NMF
            from sklearn.feature_extraction.text import TfidfVectorizer           
            model = NMF(**config['topic_modelling']['kwargs'])            
            vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        else:
            raise AttributeError('Uknown topic_modelling algorithm: {}'.format(config['topic_modelling']['algorithm']))
        topic_modelling = ClusterTopicModelling(vectorizer=vectorizer, model=model)
        # Final object
        return TopicTransformer(encoder, dim_reduction, clustering, topic_modelling, agg=config.get('agg', 'mean'))

    def fit(self, documents, batch_size):
        encodings = self.encoder.encode(documents, batch_size, return_tensors='np', agg=self.agg)
        encodings = self.dim_reduction.fit_transform(encodings)
        cluster_ids = self.clustering.fit_predict(encodings)
        del encodings
        cluster_texts = [' '.join(np.array(documents)[cluster_ids==k]) for k in np.unique(cluster_ids)]
        _ = self.topic_modelling.fit(cluster_texts)

    def transform(self, documents, batch_size):
        encodings = self.encoder.encode(documents, batch_size, return_tensors='np', agg=self.agg)
        encodings = self.dim_reduction.transform(encodings)
        cluster_ids = self.clustering.fit_predict(encodings)
        del encodings
        cluster_texts = [' '.join(np.array(documents)[cluster_ids==k]) for k in np.unique(cluster_ids)]
        return self.topic_modelling.transform(cluster_texts)

    def fit_transform(self, documents, batch_size):
        print('\tAppling transformer encoder ...')
        encodings = self.encoder.encode(documents, batch_size, return_tensors='np', agg=self.agg)
        print('\tApplying dimensionality reduction ...')
        encodings = self.dim_reduction.fit_transform(encodings)
        print('\tApplying clustering ...')
        cluster_ids = self.clustering.fit_predict(encodings)
        print('Creating text clusters ...')
        del encodings
        cluster_texts = [' '.join(np.array(documents)[cluster_ids==k]) for k in np.unique(cluster_ids)]
        print('\tApplying topic modelling ...')       
        return self.topic_modelling.fit(cluster_texts)

    def get_topics_words(self, n_top):
        return self.topic_modelling.get_topics_words(n_top)

    def save_model(self, save_dir, config=None):
        '''
        check path is dir
        pickle each object except for transfo -> only name
        '''
        with open(os.path.join(save_dir, 'model.pkl'), mode='wb') as h:
            pickle.dump(self, h)
        if config is not None:
            _ = config['topic_modelling']['vectorizer_kwargs'].pop('preprocessor')
            with open(os.path.join(save_dir, 'config.json'), mode='w', encoding='utf-8') as f:
                json.dump(config, f)

    @classmethod
    def from_path(cls, save_dir):
        if os.path.isfile(os.path.join(save_dir, 'config.json')):
            with open(os.path.join(save_dir, 'config.json')) as f:
                config = json.load(f)
        else:
            config = {}
        with open(os.path.join(save_dir, 'model.pkl'), mode='rb') as h:
            topic_transformer = pickle.load(h)
        return topic_transformer