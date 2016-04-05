import pandas as pd
import math
from nltk import tokenize as nltok, tag as nltag


def teacher_normalize(x):
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
    exciting = outcomes['is_exciting']
    teachers = projects['teacher_acctid']
    df = pd.concat([exciting, teachers], axis=1)
    df = df[df['is_exciting'] == 't']
    teacher_exciting_counts = {}

    for index, row in df.iterrows():
        teacher = row['teacher_acctid']
        teacher_exciting_counts[teacher] = teacher_exciting_counts.get(teacher, 0) + 1
    # max_count = max(teacher_exciting_counts, key=teacher_exciting_counts.get)
    #print teacher_exciting_counts
    teacher_exciting_counts = {k: teacher_normalize(v) for k, v in teacher_exciting_counts.items()}
    projects['teacher_exciting_count'] = projects['teacher_acctid'].map(teacher_exciting_counts)
    projects['teacher_exciting_count'].fillna(0.0, inplace=True)
    print projects.shape
    #print projects.head
    teacher = projects[['projectid', 'teacher_exciting_count']]
    print teacher.shape
    teacher.to_csv('data/teacher_performance.csv', index=False)

    print('teachers\' data saved')
    return projects


def find_teacher():
    print('loading raw data')
    # donations = pd.read_csv('data/donations.csv')
    projects = pd.read_csv('data/projects.csv')
    outcomes = pd.read_csv('data/outcomes.csv')

    # share the index
    projects = projects.sort('projectid')
    outcomes = outcomes.sort('projectid')

    find_teacher_history(projects, outcomes)


def find_school_history(projects, outcomes):
    exciting = outcomes['is_exciting']
    schools = projects['school_id']
    df = pd.concat([exciting, schools], axis=1)
    df = df[df['is_exciting'] == 't']
    school_exciting_counts = {}

    for index, row in df.iterrows():
        school = row['school_id']
        school_exciting_counts[school] = school_exciting_counts.get(school, 0) + 1
    #print     school_exciting_counts
        school_exciting_counts = {k: teacher_normalize(v) for k, v in school_exciting_counts.items()}
    projects['school_exciting_counts'] = projects['school_id'].map(school_exciting_counts)
    projects['school_exciting_counts'].fillna(0.0, inplace=True)
    print projects.shape
    #print projects.head
    teacher = projects[['projectid', 'school_exciting_counts']]
    print teacher.shape
    teacher.to_csv('data/school_performance.csv', index=False)

    print('schools\' data saved')
    return projects


def find_school():
    print('loading raw data')
    # donations = pd.read_csv('data/donations.csv')
    projects = pd.read_csv('data/projects.csv')
    outcomes = pd.read_csv('data/outcomes.csv')

    # share the index
    projects = projects.sort('projectid')
    outcomes = outcomes.sort('projectid')

    find_school_history(projects, outcomes)


def find_essaylen():
    print('loading raw essay data...')
    essays = pd.read_csv('data/essays.csv')
    essays.sort_values(by='projectid')

    totalCount = essays.shape[0]
    for i in range(1, essays.shape[1]):
        nullcount = essays[essays[essays.columns[i]].isnull()].shape[0]
        percentage = float(nullcount) / float(totalCount) * 100
        if percentage > 0:
            print(essays.columns[i], percentage, '%')

    essays.fillna('no_text')
    essays['essay_len'] = essays['essay'].astype(str).apply(lambda x: len(x))

    essaylen = essays[['projectid', 'essay_len']]
    mean = essaylen['essay_len'].mean()
    std = essaylen['essay_len'].std()
    essaylen['essay_len'] = essays['essay_len'].apply(lambda x: (x-mean)/std)
    print essaylen.head

    print essaylen.shape
    essaylen.to_csv('data/essay_length.csv', index=False)

def find_cd_count():
    projects = pd.read_csv('data/projects.csv')
    outcomes = pd.read_csv('data/outcomes.csv')

    # share the index
    projects = projects.sort_values('projectid')
    outcomes = outcomes.sort_values('projectid')
    projects.set_index('projectid', inplace=True)
    outcomes.set_index('projectid', inplace=True)

    projects = projects[projects['date_posted'] >= '2010-04-01']
    train_idx = projects[projects['date_posted'] < '2014-01-01'].index.values.tolist()
    test_idx = projects[projects['date_posted'] >= '2014-01-01'].index.values.tolist()
    outcomes  = outcomes.ix[train_idx]

    print('loading raw essay data...')
    essays = pd.read_csv('data/essays.csv')
    essays.sort_values(by='projectid')
    essays.set_index('projectid', inplace=True)
    print('projects shape', projects.ix[train_idx].shape)
    print('essays shape', essays.ix[train_idx].shape)

    print('cleaning essay...')
    essays = essays.ix[projects.index.values.tolist()]  # align
    essays.fillna('no_text')

    # POS to calculate the number of Numerical words in the essay
    train_sents = essays.ix[train_idx]['essay'].astype(str)
    test_sents = essays.ix[test_idx]['essay'].astype(str)

    '''
        print('calculating essay length...')
        train_essay_length = [len(t) for s in train_sents for t in nltok.word_tokenize(s.decode('utf-8'))]
        #test_essay_length = [len(t) for s in test_sents for t in nltok.word_tokenize(s.decode('utf-8'))]

        print datetime.now() - startTime

        print('saving essay_length to file...')
        sample = pd.read_csv('./data/trainEssayLength.csv')
        sample['essay_length'] = train_essay_length
        sample.to_csv('trainEssayLength.csv', index = False)
        print('essay_length saved to file')
    '''

    print('calculating essay num...')
    for train_sent_list in train_sents:
        train_sent_list = nltok.sent_tokenize(train_sent_list.decode('utf-8'))
    print('sentences to list')

    tmp = []
    for i in range(0,len(train_sents)-1):
        num = 0
        for j in range(0, len(train_sents[i])-1):
            for wt in nltag.pos_tag(train_sents[i][j]):
                if(wt[1]=='CD'):
                        num+=1
        tmp.append(num)

    train_essay_cd_num = pd.Series(tmp, index = train_idx)


    for test_sent_list in train_sents:
        test_sent_list = nltok.sent_tokenize(test_sent_list.decode('utf-8'))

    test_words = [nltok.word_tokenize(test_sent) for test_sent_list in test_sents for test_sent in test_sent_list ]

    test_word_tags = [nltag.pos_tag(w) for b in test_words for a in b for w in a ]

    tmp = []
    for i in range(0,len(test_word_tags)-1):
        num = 0
        for j in range(0, len(test_word_tags[0])-1):
            for k in range(0, len(nltag.word_tags[0][0])-1):
                if(test_word_tags[i][j][k][0]=='CD'):
                    num+=1
        tmp.append(num)
    test_essay_cd_num = pd.Series(tmp, index = train_idx)

    print('saving essay_num to file')
    sample1 = pd.read_csv('./data/outcomes.csv').ix[train_idx]
    sample2 = pd.read_csv('./data/outcomes.csv').ix[test_idx]
    sample1 = sample1.sort_values(by='projectid')
    sample2 = sample2.sort_values(by='projectid')
    sample1.set_index('projectid', inplace=True)
    sample2.set_index('projectid', inplace=True)
    drop_labels_s = ['is_exciting', 'at_least_1_teacher_referred_donor', 'fully_funded', 'at_least_1_green_donation', 'great_chat',
    'three_or_more_non_teacher_referred_donors', 'one_non_teacher_referred_donor_giving_100_plus',
    'donation_from_thoughtful_donor', 'great_messages_proportion', 'teacher_referred_count', 'non_teacher_referred_count']
    for label in drop_labels_s:
        sample1.drop(label, axis=1, inplace=True)
        sample2.drop(label, axis=1, inplace=True)
    sample1['essay_cd_num'] = train_essay_cd_num
    sample2['essay_cd_num'] = train_essay_cd_num
    sample1.to_csv('./data/trainEssayNum.csv', index = True)
    sample2.to_csv('./data/testEssayNum.csv', index = True)


find_essaylen()
find_school()
find_teacher()
find_cd_count()