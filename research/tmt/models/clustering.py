from sklearn.cluster import KMeans as Kmeans_
from hdbscan import HDBSCAN

class KMeans(Kmeans_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'kmeans'
        self.id_start = 0
    
class Hdbscan(HDBSCAN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'hdbscan'
        self.id_start = -1
    
    def fit_transform(self, *args, **kwargs):
        return super().fit_predict(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return hdbscan.approximate_predict(self, *args, **kwargs)
