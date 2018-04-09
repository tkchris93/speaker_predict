from __future__ import division
import numpy as np
import pandas as pd
import itertools
import pickle
from mypipeline import MultinomialNaiveBayesLogProbs, CleanTable, test_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

mnb_lr_pipe = Pipeline(steps=[('log_probs', MultinomialNaiveBayesLogProbs()),
                           ('clean', CleanTable()),
                           ('lr', LogisticRegression())])

df = pd.read_pickle("talks_norm_pos.pdpkl")

# Random Forest
param_grid = [['text', 'lemmatized_text'], # text column 
              ['standard'], # count weighting
              [1,5,10,15], #min_df
              [10**i for i in xrange(-2,3)]] # C 

results = []
for params in tqdm(itertools.product(*param_grid), total=np.product([len(g) for g in param_grid])):
    text_column, count_weighting, min_df, C = params
    print params,
    accs = test_pipeline(df, mnb_lr_pipe.set_params(log_probs__text_column = text_column,
                                                log_probs__count_weighting = count_weighting,
                                                log_probs__min_df = min_df,
                                                lr__C = C))
    results.append((params, accs))

with open("lr_results.pypkl", 'w') as f:
    pickle.dump(results, f)
