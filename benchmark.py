# -*- coding: utf-8 -*-

"""

Beating the benchmark @ KDD 2014

__author__ : Abhishek Thakur

"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
        except:
            return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()

donations = pd.read_csv('data/donations.csv')
projects = pd.read_csv('data/projects.csv')
outcomes = pd.read_csv('data/outcomes.csv')
resources = pd.read_csv('data/resources.csv')
sample = pd.read_csv('data/sampleSubmission.csv')
essays = pd.read_csv('data/essays.csv')


essays = essays.sort('projectid')
projects = projects.sort('projectid')
sample = sample.sort('projectid')
ess_proj = pd.merge(essays, projects, on='projectid')
outcomes = outcomes.sort('projectid')


outcomes_arr = np.array(outcomes)


labels = outcomes_arr[:,1]

ess_proj['essay'] = ess_proj['essay'].apply(clean)

ess_proj_arr = np.array(ess_proj)

train_idx = np.where(ess_proj_arr[:,-1] < '2014-01-01')[0]
test_idx = np.where(ess_proj_arr[:,-1] >= '2014-01-01')[0]


traindata = ess_proj_arr[train_idx,:]
testdata = ess_proj_arr[test_idx,:]


tfidf = TfidfVectorizer(min_df=3,  max_features=1000)

tfidf.fit(traindata[:,5])
tr = tfidf.transform(traindata[:,5])
ts = tfidf.transform(testdata[:,5])


lr = linear_model.LogisticRegression()
lr.fit(tr, labels=='t')
preds =lr.predict_proba(ts)[:,1]


sample['is_exciting'] = preds
sample.to_csv('benchmark_predictions.csv', index = False)