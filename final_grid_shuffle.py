import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from mypipeline import MultinomialNaiveBayesLogProbs, CleanTable, test_pipeline
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pickle

results_files = ["{}_results.pypkl".format(m) for m in ['lr', 'rf', 'xgb'] ]

results = []
for f in results_files:
    with open(f, 'rb') as in_file:
        results.append(pickle.load(in_file))

lr_results, rf_results, xgb_results = results

lr_acc  = [np.mean(j) for i,j in lr_results]
rf_acc  = [np.mean(j) for i,j in rf_results]
xgb_acc = [np.mean(j) for i,j in xgb_results]

lr_var  = [np.var(j) for i,j in lr_results]
rf_var  = [np.var(j) for i,j in rf_results]
xgb_var = [np.var(j) for i,j in xgb_results]

lr_mask  = np.argsort(lr_acc)[::-1][:7]
rf_mask  = np.argsort(rf_acc)[::-1][:7]
xgb_mask = np.argsort(xgb_acc)[::-1][:7]

lr_models = []
for i in lr_mask:
    print lr_results[i]
    print np.mean(lr_results[i][1])
    print ""
    lr_models.append(lr_results[i])
    
rf_models = []
for i in rf_mask:
    rf_models.append(rf_results[i])
     
xgb_models = []
for i in rf_mask:
    xgb_models.append(xgb_results[i])

def test_pipeline(df, nlp_pipeline, y_column='speaker'):
    label = df[y_column].copy()
    X = df.drop(y_column, axis=1).copy()

    rskf = StratifiedKFold(n_splits=10, shuffle=True)
    accs = []
    for train_index, test_index in tqdm(rskf.split(X, label), total=10):
        X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_test = label[train_index], label[test_index]
        
        nlp_pipeline.fit(X_train.reset_index(), y_train)
        accs.append((nlp_pipeline.predict(X_test.reset_index()) == y_test).mean())

    print "avg accuracies:", np.mean(accs)
    return accs

lr_model_score = []

df = pd.read_pickle("talks_norm_pos.pdpkl")
for params, _ in lr_models:
    accs = []
    lr_pipe = Pipeline(steps=[('log_probs', MultinomialNaiveBayesLogProbs()), 
                              ('clean', CleanTable()), ('lr', LogisticRegression())])

    for _ in xrange(3):
        a = test_pipeline(df, lr_pipe.set_params(log_probs__text_column = params[0],
                                                    log_probs__count_weighting = params[1],
                                                    log_probs__min_df = params[2],
                                                    lr__C = params[3]))

        accs += a
        
    lr_model_score.append(np.mean(accs))

with open("final_grid_lr.pypkl", 'w') as f:
    pickle.dump(zip(lr_models, lr_model_score) , f)






from sklearn.ensemble import RandomForestClassifier

rf_model_score = []

df = pd.read_pickle("talks_norm_pos.pdpkl")
for params, _ in rf_models:
    accs = []
    rf_pipe = Pipeline(steps=[('log_probs', MultinomialNaiveBayesLogProbs()), 
                              ('clean', CleanTable()), ('rf', RandomForestClassifier())])

    for _ in xrange(3):
        a = test_pipeline(df, rf_pipe.set_params(log_probs__text_column = text_column,
                                                log_probs__count_weighting = count_weighting,
                                                log_probs__min_df = min_df,
                                                rf__n_estimators = n_estimators,
                                                rf__max_depth = max_depth))

        accs += a
        
    rf_model_score.append(np.mean(accs))

with open("final_grid_rf.pypkl", 'w') as f:
    pickle.dump(zip(rf_models, rf_model_score) , f)


from xgboost import XGBClassifier

xgb_model_score = []

df = pd.read_pickle("talks_norm_pos.pdpkl")
for params, _ in xgb_models:
    accs = []
    xgb_pipe = Pipeline(steps=[('log_probs', MultinomialNaiveBayesLogProbs()), 
                              ('clean', CleanTable()), ('rf', XGBClassifier())])

    for _ in xrange(3):
        a = test_pipeline(df, xgb_pipe.set_params(log_probs__text_column = text_column,
                                                log_probs__count_weighting = count_weighting,
                                                log_probs__min_df = min_df,
                                                rf__n_estimators = n_estimators,
                                                rf__max_depth = max_depth))


        accs += a
        
    xgb_model_score.append(np.mean(accs))

with open("final_grid_xgb.pypkl", 'w') as f:
    pickle.dump(zip(xgb_models, xgb_model_score) , f)

