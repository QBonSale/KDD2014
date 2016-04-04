# coding=utf-8
import csv
import pandas as pd
import numpy as np
from DataFrameImputer import DataFrameImputer
import cPickle as pickle
import math
import MachineLearning as mlp


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
    try:
        print('loading processed data...')
        f = file('data/data.bin', 'rb')
        projects = np.load(f)
        outcomes = np.load(f)
        train_idx = np.load(f)
        test_idx = np.load(f)
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
        print('processed data not found, recomputing')
        print('loading raw data')
        # donations = pd.read_csv('data/donations.csv')
        projects = pd.read_csv('data/projects.csv')
        outcomes = pd.read_csv('data/outcomes.csv')
        # essays = pd.read_csv('data/essays.csv')

        # share the index
        # essays = essays.sort_values(by='projectid')
        projects = projects.sort_values(by='projectid')
        outcomes = outcomes.sort_values(by='projectid')
        projects.set_index('projectid')
        outcomes.set_index('projectid')
        print('raw data loaded')

        train_idx = projects[
            (projects['date_posted'] >= '2010-04-01') & (projects['date_posted'] <= '2014-01-01')].index.values
        test_idx = projects[projects['date_posted'] >= '2014-01-01'].index.values

        # TODO: do we need to drop project_id before fitting?
        print('dropping unnecessary columns')
        drop_labels = ['school_ncesid', 'school_latitude', 'school_longitude', 'school_zip', 'school_district',
                       'school_county', 'secondary_focus_subject', 'secondary_focus_area']
        drop_labels.append('date_posted')
        # drop_labels.append('school_city')
        for label in drop_labels:
            projects.drop(label, axis=1, inplace=True)

        print('imputing missing elements')  # mean for number, most frequent for nan
        dfi = DataFrameImputer()
        projects = dfi.fit_transform(projects)
        outcomes = dfi.fit_transform(outcomes)

        print('factorizing catagorical values')
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
        print(projects.head)

        print('saving processed data')  # save data into a binary file
        f = file('data/data.bin', 'wb')
        np.save(f, projects)
        np.save(f, outcomes)
        np.save(f, train_idx)
        np.save(f, test_idx)
        f.close()
        print('processed data saved')

        print('training model')




