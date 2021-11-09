from models.topic_transformer import TopicTransformer
# from models.dim_reduction import Umap, Pca, Tsne
# from models.clustering import KMeans, Hdbscan
# from models.utils import Preprocessor
# from models.topic_modelling import ClusterTopicModelling
from sklearn.datasets import fetch_20newsgroups
import os
import shutil

config = {
    'encoder': {
        'name': 'bert-base-cased'
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
            'n_clusters': 10
        }
    },
    'topic_modelling': {
        'algorithm': 'cluster_lda',
        'kwargs': {
            'n_components': 10,
            'max_iter': 5,
            'learning_method':'online',
            'learning_offset': 50,
            'random_state': 0,
        },
        'vectorizer_kwargs': {
            'max_df': 0.95, 
            'min_df': 2, 
            'max_features': 1000, 
            'stop_words': 'english',
        },
        'preprocessing_kwargs': {}
    },
    'n_top_words': 10,
    'save_dir': 'saved_models',
    'batch_size': 56,
}

if __name__ == '__main__':
    try:
        try:
            print('Starting tests')
            save_dir = os.path.join(os.path.dirname(__file__), config['save_dir'])
            os.mkdir(save_dir)
            print('Loading data')
            data, _ = fetch_20newsgroups(shuffle=True,
                                        random_state=1,
                                        remove=("headers", "footers", "quotes"),
                                        return_X_y=True)
            # data = data[:3*config['batch_size']]
            print('Creating models')
            topic_transformer = TopicTransformer.from_config(config)
            print('Fitting model')
            topic_ids = topic_transformer.fit_transform(data, config['batch_size'])
            print('Translating topics')
            topics = topic_transformer.get_topics_words(config['n_top_words'])
            print('Saving model')
            topic_transformer.save_model(save_dir, config)
            print('Reloading model')
            topic_transformer_2 = TopicTransformer.from_path(save_dir)
            topics_2 = topic_transformer_2.get_topics_words(config['n_top_words'])
            assert topics == topics_2, f'Save/Load failed ...'
            print('passed without error.')
        except Exception as e:
            shutil.rmtree(os.path.join(os.path.dirname(__file__), config['save_dir']), ignore_errors=True)
            raise e
    except KeyboardInterrupt:
        shutil.rmtree(os.path.join(os.path.dirname(__file__), config['save_dir']), ignore_errors=True)