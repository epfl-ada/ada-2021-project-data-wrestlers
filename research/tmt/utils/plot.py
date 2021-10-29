import matplolib.pyplot as plt
from math import ceil

def plot_cluster_topic_words(topic_words): 
    '''topic_words from clusterlda.get_topic_words()'''
    fig, axes = plt.subplots(2, ceil(topic_words.shape[0]/2), figsize=(30, 15), sharex=True) 
    axes = axes.flatten()
    for cluster_id, cluster_topic_words in enumerate(topic_words):
        top_words = [t for t,_ in cluster_topic_words]
        weights = [w for _,w in cluster_topic_words]
        ax = axes[cluster_id]
        ax.barh(top_words, weights, height=0.7)
        ax.set_title(f"Topic {cluster_id +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(f'Cluster {cluster_id}', fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()