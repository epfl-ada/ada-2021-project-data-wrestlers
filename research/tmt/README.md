# Topic modelling with Transformer embedding

inspired by <a href='https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6'>this article</a>.

### Usage
```bash
# Create env
conda create -n ada-lm python=3.8
# activate env
conda activate ada-lm
# install requirements
pip install -r requirements_cpu_py38.txt
```

### Notes
- <a href='https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py'>topic modelling and plot example</a>

todo: 
- plotting
- score measure for pipeline -> `TopicTransformer.hyperparameter_search(**params)`
- gpu requirements

