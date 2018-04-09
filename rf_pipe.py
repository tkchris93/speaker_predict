from __future__ import division
import numpy as np
import pandas as pd
import itertools
from mypipeline import MultinomialNaiveBayesLogProbs, CleanTable, test_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

mnb_rf_pipe = Pipeline(steps=[('log_probs', MultinomialNaiveBayesLogProbs()),
                           ('clean', CleanTable()),
                           ('rf', RandomForestClassifier())])

df = pd.read_pickle("talks_norm_pos.pdpkl")
test_pipeline(df, mnb_rf_pipe.set_params(log_probs__text_column='lemmatized_text', 
                                         log_probs__count_weighting='normalized'))

# Random Forest
param_grid = [['text', 'lemmatized_text'], # text column 
              ['standard', 'normalized', 'tfidf'], # count weighting
              [3,7], #min_df
              [100, 150], # rf__n_estimators
              [11, 15, 19]] # rf__max_depth
results = []
for params in tqdm(itertools.product(*param_grid), total=np.product([len(g) for g in param_grid])):
    text_column, count_weighting, min_df, n_estimators, max_depth = params
    print params,
    accs = test_pipeline(df, mnb_rf_pipe.set_params(log_probs__text_column = text_column,
                                                log_probs__count_weighting = count_weighting,
                                                log_probs__min_df = min_df,
                                                rf__n_estimators = n_estimators,
                                                rf__max_depth = max_depth))
    results.append((params, accs))

print [[i,np.mean(j)] for i,j in results]
