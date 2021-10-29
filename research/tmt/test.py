from models.encoder import PretrainedTransformerEncoder
from models.dim_reduction import Umap, Pca, Tsne
from models.clustering import KMeans, Hdbscan
from utils.preprocessing import Preprocessor
from models.topic_modelling import ClusterTopicModelling

config = {
    'encoder': {
        'name': 'bert-based-uncased'
    },
    'dim_reduction': {
        'algorithm': 'pca',
        'kwargs': {
            'n_components': 2
        }
    },
    'clustering': {
        'algorithm': 'kmeans',
        'kwargs': {
            # keep going
        }
    }
}

encoder = PretrainedTransformerEncoder('bert-base-uncased')

