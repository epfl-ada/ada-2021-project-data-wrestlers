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
        self.encoder_name = encoder.model_name
        self.dim_reduction_name = dim_reduction.name
        self.clustering_name = clustering.name
        self.topic_modelling_name = topic_modelling.name
        self.encoder = encoder # encode api
        self.dim_reduction = dim_reduction # fit, transform, fit_transform api
        self.clustering = clustering # fit, transform, fit_transform api
        self.topic_modelling = topic_modelling # fit, transform, fit_transform api
        self.agg = agg

    def fit(self, documents):
        encodings = self.encoder.encode(documents, return_tensors='np', agg=self.agg)
        encodings = self.dim_reduction.fit_transform(encodings)
        cluster_ids = self.clustering.fit_transform(encodings)
        del encodings
        cluster_texts = [' '.join(documents[cluster_ids==k] for k in range(self.clustering.id_start, cluster_ids.max()+1))]
        _ = self.topic_modelling.fit(cluster_texts)

    def transform(self, documents):
        encodings = self.encoder.encode(documents, return_tensors='np', agg=self.agg)
        encodings = self.dim_reduction.transform(encodings)
        cluster_ids = self.clustering.transform(encodings)
        del encodings
        cluster_texts = [' '.join(documents[cluster_ids==k] for k in range(self.clustering.id_start, cluster_ids.max()+1))]
        return self.topic_modelling.transform(cluster_texts)

    def fit_transform(self, documents):
        encodings = self.encoder.encode(documents, return_tensors='np', agg=self.agg)
        encodings = self.dim_reduction.fit_transform(encodings)
        cluster_ids = self.clustering.fit_transform(encodings)
        del encodings
        cluster_texts = [' '.join(documents[cluster_ids==k] for k in range(self.clustering.id_start, cluster_ids.max()+1))]
        return self.topic_modelling.fit(cluster_texts)

    def get_topics_words(self, n_top):
        return self.topic_modelling.get_topics_words(n_top)
    