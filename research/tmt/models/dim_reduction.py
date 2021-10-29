from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Umap(UMAP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'umap'

class Pca(PCA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'pca'

class Tsne(TSNE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'tsne'
