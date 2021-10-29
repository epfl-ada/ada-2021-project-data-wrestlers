inspired by: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

note: cluster -1 in hdscan is no cluster

I want:

quote_df (text_column, cluster_column)
topic_df (topic_id(-1,n), 

topic_df, quote_df = topic_modelling(quote_df)

topic modelling lda + nice plots: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

todo: 
- score measure for pipeline -> hyperparam search
- TopicTransformer.from_config(config)