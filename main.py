import pandas as pd
import numpy as np
import cPickle as pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import RFECV
import Stemmer
from sklearn import preprocessing
from datetime import datetime


def cutoff(preds, threshold):
    return (preds > threshold).astype(int)

english_stemmer = Stemmer.Stemmer('en')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: 'no_text' if (doc is np.nan) else english_stemmer.stemWords(analyzer(doc))


def clean_essay(s):
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE | re.LOCALE)).lower()

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

        #discarded_idx = projects[projects['date_posted'] < '2010-01-01'].index.values.tolist()
        train_idx = projects[projects['date_posted'] < '2014-01-01'].index.values.tolist()
        test_idx = projects[projects['date_posted'] >= '2014-01-01'].index.values.tolist()


        #train_idx = list(set(train_idx)-set(discarded_idx))
        # test_idx = list(set(all_idx)-set(train_idx))

        outcomes = outcomes.ix[train_idx]
        print(projects.shape)
        print(outcomes.shape)

        print('train data shape', projects.ix[train_idx].shape)
        print('test data shape', projects.ix[test_idx].shape)

        print('raw data loaded')
        #
        print('dropping unnecessary columns...')
        drop_labels = ['school_ncesid', 'schoolid', 'teacher_acctid', 'secondary_focus_subject', 'secondary_focus_area', 'school_state']
        drop_labels.append('date_posted')
        # drop_labels.append('school_city')
        for label in drop_labels:
            projects.drop(label, axis=1, inplace=True)

        # projects_numeric_columns = ['school_latitude', 'school_longitude',
        #                             'fulfillment_labor_materials',
        #                             'total_price_excluding_optional_support',
        #                             'total_price_including_optional_support']
        # projects = projects[projects_numeric_columns]
        #
        print('imputing missing elements...')  # mean for number, most frequent for nan
        projects = projects.fillna(method='pad')  # fill the missing hole with the previous observation data
        #
        #
        # # dfi = DataFrameImputer()
        # # projects = dfi.fit_transform(projects)
        # #outcomes = dfi.fit_transform(outcomes)
        #
        print('factorizing catagorical values...')
        proj_cat_labels = ['school_metro',
                           'school_charter', 'school_city', 'school_district', 'school_county',
                           'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp',
                           'school_charter_ready_promise',
                           'teacher_prefix',
                           'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
                           'primary_focus_subject', 'primary_focus_area', 'resource_type', 'poverty_level',
                           'grade_level', 'eligible_double_your_impact_match', 'eligible_almost_home_match'
                           ]
        for label in proj_cat_labels:
            projects[label], values = projects[label].factorize()
        # #     projects.drop(label, axis=1, inplace=True)
        #
        # outcomes_cat_labels = ['is_exciting', 'at_least_1_teacher_referred_donor', 'fully_funded',
        #                        'at_least_1_green_donation', 'great_chat',
        #                        'three_or_more_non_teacher_referred_donors',
        #                        'one_non_teacher_referred_donor_giving_100_plus', 'donation_from_thoughtful_donor'
        #                        ]
        # for label in outcomes_cat_labels:
        #     outcomes[label], values = outcomes[label].factorize()
        outcomes['is_exciting'], values = outcomes['is_exciting'].factorize()
        #
        #print('replacing teacher_id with his performance')
        #projects.drop('teacher_acctid', axis=1, inplace=True)
        # print('train data shape', projects.ix[train_idx].shape)
        # print('test data shape', projects.ix[test_idx].shape)
        #
        # # print('saving processed data...')  # save data into a binary file
        # # with open('data/projects.bin', 'wb') as f:
        # #     pickle.dump(projects, f)
        # #     pickle.dump(outcomes, f)
        # #     pickle.dump(train_idx, f)
        # #     pickle.dump(test_idx, f)
        # # print('processed data saved')

        print('training models')
        train = projects.ix[train_idx]
        labels = outcomes['is_exciting']
        test = projects.ix[test_idx]

        scaler = preprocessing.StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
        clf = RandomForestClassifier(class_weight='balanced')
        #clf = GradientBoostingClassifier(learning_rate=0.2, subsample=0.4, n_estimators=150)
        clf.fit(train, labels)
        preds = clf.predict_proba(test)

        print('saving prediction to file')
        sample = pd.read_csv('./data/sampleSubmission.csv')
        sample.sort_values('projectid')
        sample['is_exciting'] = preds[:,1]
        #sample['is_exciting'] = cutoff(preds, 0.5)
        sample.to_csv('gbc_predictions.csv', index=True)
        exit(0)

    # load essay data
    try:
        raise OSError
        startTime = datetime.now()
        print('loading essay data...')
        f = open('data/essays.bin', 'rb')
        train_essays = pickle.load(f)
        test_essays = pickle.load(f)
        f.close()
        print datetime.now() - startTime

    except (OSError, IOError) as e:
        #print('essay data not found, recomputing...')
        print('loading raw essay data...')
        essays = pd.read_csv('data/essays.csv')
        essays.sort_values(by='projectid')
        essays.set_index('projectid', inplace=True)

        print('cleaning essay...')
        essays = essays.ix[projects.index.values.tolist()]  # align
        print('projects shape', projects.shape)
        print('essays shape', essays.shape)
        essays['essay'] = essays['essay'].fillna('garbage').apply(clean_essay)

        # ('title', 0.002559863152727459, '%')
        # ('short_description', 0.019876584480001444, '%')
        # ('need_statement', 0.22165403298910702, '%')    too many missing
        # ('essay', 0.0004517405563636692, '%')

        startTime = datetime.now()
        print('vectorizing essays...')
        #vectorizer = StemmedTfidfVectorizer(min_df=3, max_features=500, stop_words='english')
        vectorizer = TfidfVectorizer(min_df=3, max_features=1000, stop_words='english')
        # learn rules on train_data
        train_essays = vectorizer.fit_transform(essays.ix[train_idx]['essay'])
        test_essays = vectorizer.transform(essays.ix[test_idx]['essay'])

        print datetime.now() - startTime

        print('saving essay data')
        with open('data/essays.bin', 'wb') as f:
            pickle.dump(train_essays, f)
            pickle.dump(test_essays, f)
        print('processed data saved')

    target = outcomes['is_exciting']
    target = np.array(target == 't').astype(int)

    # model = LogisticRegression(class_weight='balanced')
    # model.fit(train_essays, target)
    # preds = model.predict_proba(test_essays)[:, 1]
    #
    # print('saving prediction to file')
    # sample = pd.read_csv('./data/sampleSubmission.csv')
    # sample.sort_values('projectid')
    # sample['is_exciting'] = preds
    # #sample['is_exciting'] = cutoff(preds, 0.5)
    # sample.to_csv('essay_predictions.csv', index=True)

    clf = MultinomialNB()
    print("Finding support.")
    sel = RFECV(clf, step=.01, cv=5, scoring='roc_auc')
    sel.fit(train_essays, target)
    print("Number of support samples = %i" % sel.n_features_)
    print("Training.")
    clf.fit(train_essays.tocsc()[:, sel.support_], target)
    preds = clf.predict_proba(test_essays.tocsc()[:, sel.support_])[:, 1]

    print("Writing predictions.")
    sample['is_exciting'] = preds
    sample.to_csv('NB_predictions.csv', index=False)
