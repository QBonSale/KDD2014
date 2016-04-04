import pandas as pd
import numpy as np
import re
import cPickle as pickle
import math
import MachineLearning as mlp

from sklearn.feature_extraction.text import TfidfVectorizer
from DataFrameImputer import DataFrameImputer
from stemming.porter2 import stem


def stem_essay(s):
    try:
        words = re.findall(r'\w+', s, flags=re.UNICODE | re.LOCALE).lower()
        return " ".join([stem(word) for word in words])
    except: # no \w+ at all
        return " ".join(re.findall(r'\w+', "no_text", flags=re.UNICODE | re.LOCALE)).lower()


def normalize(x):
    return 1 + math.log(float(x)) if x > 0 else 0
'''0.5+0.5*x/max
189715                0.500488
380237                0.500195
353836                0.500195
256352                0.500293
406141                0.500390
480855                0.500098
649513                0.500098
289514                0.500098
617275                0.500098
405893                0.500098
346671                0.500195
181583                0.500195
258501                0.500098

1+log
98255                 0.000000
109457                0.000000
225603                0.000000
168043                0.000000
224242                2.098612
600359                0.000000
288974                0.000000
405003                1.000000
537810                0.000000
345627                0.000000
527215                0.000000
28184                 1.000000
629719                0.000000
412638                0.000000
185084                0.000000
158817                1.000000
533083                0.000000
9567                  0.000000
435975                2.791759
353453                0.000000
497637                0.000000
'''


def find_teacher_history(projects, outcomes):
    try:
        teacher_exciting_counts = pickle.load(open('data/teacher_exciting_count.bin', 'rb'))
        print('teachers\' data loaded')
    except (OSError, IOError) as e:
        print('teachers\' data not found, creating...')
        exciting = outcomes[outcomes['is_exciting'] == 1][['projectid', 'is_exciting']]
        teachers = projects[['projectid', 'teacher_acctid']]
        df = pd.merge(exciting, teachers, on='projectid', how='inner')
        teacher_exciting_counts = {}

        for index, row in df.iterrows():
            teacher = row['teacher_acctid']
            teacher_exciting_counts[teacher] = teacher_exciting_counts.get(teacher, 0) + 1
        # max_count = max(teacher_exciting_counts, key=teacher_exciting_counts.get)

        teacher_exciting_counts = {k: normalize(v) for k, v in teacher_exciting_counts.items()}
        pickle.dump(teacher_exciting_counts, open('data/teacher_exciting_count.bin', 'wb'))
        print('teachers\' data saved')

    projects['teacher_exciting_count'] = projects['teacher_acctid'].map(teacher_exciting_counts)
    projects['teacher_exciting_count'].fillna(0.0, inplace=True)
    return projects


if __name__ == '__main__':

    # load projects data
    try:
        raise OSError
        print('loading processed data...')
        f = open('data/projects.bin', 'rb')
        projects = pickle.load(f)
        outcomes = pickle.load(f)
        train_idx = pickle.load(f)
        test_idx = pickle.load(f)
        f.close()

        print('training models')
        ml = mlp.MachineLearning()
        ml.training_data = projects
        ml.training_labels = outcomes[:,1]
        ml.preprocessTrain(0)
        ml.preprocessTest()
        ml.trainModel(1)
        preds = ml.predict()

        print('saving prediction to file')
        sample = pd.read_csv('./data/sampleSubmission.csv')
        sample['is_exciting'] = preds
        sample.to_csv('predictions.csv', index = False)

        #print projects.ix[train_idx].shape
        #print projects.ix[test_idx].shape

    except (OSError, IOError) as e:
        #print('processed data not found, recomputing...')
        print('loading raw data')
        # donations = pd.read_csv('data/donations.csv')
        projects = pd.read_csv('data/projects.csv')
        outcomes = pd.read_csv('data/outcomes.csv')

        # share the index
        # essays = essays.sort_values(by='projectid')
        projects = projects.sort_values('projectid')
        outcomes = outcomes.sort_values('projectid')
        projects.set_index('projectid', inplace=True)
        outcomes.set_index('projectid', inplace=True)

        projects = projects[projects['date_posted'] >= '2010-04-01']
        train_idx = projects[projects['date_posted'] < '2014-01-01'].index.values.tolist()
        test_idx = projects[projects['date_posted'] >= '2014-01-01'].index.values.tolist()
        outcomes  = outcomes.ix[train_idx]
        print(projects.shape)
        print(outcomes.shape)

        # train_idx = outcomes.index.values.tolist()
        # discarded_idx = list(projects.projectid)
        # train_idx = list(set(train_idx)-set(discarded_idx))
        # all_idx = projects.index.values.tolist()
        # #test_idx = list(set(all_idx)-set(train_idx))

        print('train data shape', projects.ix[train_idx].shape)
        print('test data shape', projects.ix[test_idx].shape)

        print('raw data loaded')

        print('dropping unnecessary columns...')
        drop_labels = ['school_ncesid', 'school_latitude', 'school_longitude', 'school_zip', 'school_district',
                       'school_county', 'secondary_focus_subject', 'secondary_focus_area']
        drop_labels.append('date_posted')
        # drop_labels.append('school_city')
        for label in drop_labels:
            projects.drop(label, axis=1, inplace=True)

        print('imputing missing elements...')  # mean for number, most frequent for nan
        dfi = DataFrameImputer()
        projects = dfi.fit_transform(projects)
        outcomes = dfi.fit_transform(outcomes)

        print('factorizing catagorical values...')
        proj_cat_labels = ['teacher_acctid', 'schoolid', 'school_city', 'school_state', 'school_metro',
                           'school_charter',
                           'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp',
                           'school_charter_ready_promise',
                           'teacher_prefix',
                           'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
                           'primary_focus_subject', 'primary_focus_area', 'resource_type', 'poverty_level',
                           'grade_level', 'eligible_double_your_impact_match', 'eligible_almost_home_match'
                           ]
        # proj_cat_labels.remove('school_city')
        for label in proj_cat_labels:
            projects[label], values = projects[label].factorize()

        outcomes_cat_labels = ['is_exciting', 'at_least_1_teacher_referred_donor', 'fully_funded',
                               'at_least_1_green_donation', 'great_chat',
                               'three_or_more_non_teacher_referred_donors',
                               'one_non_teacher_referred_donor_giving_100_plus', 'donation_from_thoughtful_donor'
                               ]
        for label in outcomes_cat_labels:
            outcomes[label], values = outcomes[label].factorize()

        num_labels = []

        print('replacing teacher_id with his performance')
        find_teacher_history(projects, outcomes)
        projects.drop('teacher_acctid', axis=1, inplace=True)

        '''
        print('saving processed data...')  # save data into a binary file
        with open('data/projects.bin', 'wb') as f:
            pickle.dump(projects, f)
            pickle.dump(outcomes, f)
            pickle.dump(train_idx, f)
            pickle.dump(test_idx, f)
        print('processed data saved')
        '''
        print('training models')
        ml = mlp.MachineLearning()
        ml.training_data = projects.ix[train_idx]
        ml.training_labels = outcomes.is_exciting
        ml.testing_data = projects.ix[test_idx]
        #print( ml.training_data.head)
        #print( ml.training_labels.head)
        ml.preprocessTrain()
        ml.preprocessTest()
        ml.trainSingleModel('LRModel')
        preds = ml.predict()

        print('saving prediction to file')
        sample = pd.read_csv('./data/sampleSubmission.csv')
        print(preds)
        sample.set_index('projectid',inplace = True)
        sample['is_exciting'] = preds
        sample.to_csv('predictions.csv', index = True)

    # load essay data
    '''
    try:
        #raise OSError
        print('loading essay data...')
        f = open('data/essays.bin', 'rb')
        train_essays = pickle.load(f)
        test_essays = pickle.load(f)
        f.close()
    except (OSError, IOError) as e:
        #print('essay data not found, recomputing...')
        print('loading raw essay data...')
        essays = pd.read_csv('data/essays.csv')
        essays.sort_values(by='projectid')
        essays.set_index('projectid')
        print('projects shape', projects.ix[train_idx].shape)
        print('essays shape', essays.ix[train_idx].shape)

        print('cleaning essay...')
        essays = essays.ix[projects.index.values.tolist()]  # align
        essays.fillna('')
        essays['essay'] = essays['essay'].apply(stem_essay)
    '''

    '''
        totalCount = essays.shape[0]
        for i in range(1, essays.shape[1]):
            nullcount = essays[essays[essays.columns[i]].isnull()].shape[0]
            percentage = float(nullcount) / float(totalCount) * 100
            if percentage > 0:
                print(essays.columns[i], percentage, '%')
    '''

    '''
        ('title', 0.002559863152727459, '%')
        ('short_description', 0.019876584480001444, '%')
        ('need_statement', 0.22165403298910702, '%')    too many missing
        ('essay', 0.0004517405563636692, '%')
    '''

    '''
        print('vectorizing essays...')
        # minimum 5 appearances, auto-detectcorpus stop word with rate>=98%
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.98, max_features=500)
        # learn rules on train_data
        train_essays = vectorizer.fit_transform(essays.ix[train_idx]['essay'])
        test_essays = vectorizer.transform(essays.ix[test_idx]['essay'])

        print('saving essay data')
        with open('data/essays.bin', 'wb') as f:
            pickle.dump(train_essays, f)
            pickle.dump(test_essays, f)
        print('processed data saved')

        print('training model')
    '''


