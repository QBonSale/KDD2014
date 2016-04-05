from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import numpy as np

__author__ = "Sirui Xie"

class Model:
    '''
    Ada Boosting Classifier
    '''

    def __init__(self):
        self.model = GaussianNB()
        return

    def train(self, data, target):
        self.model.fit(data, target)

    def predict(self, test):
        return self.model.predict(test)

class SVClassifier(Model):
    '''
    Supporting Vector Classifier
    '''
    def __init__(self):
        Model.__init__(self)
        self.model = svm.SVC()

    def predict(self, test):
        self.model.predict(test)

class LRModel(Model):
    '''
    Logistic Regression Model
    '''
    def __init__(self):
        Model.__init__(self)
        self.model = LogisticRegression()

    def train(self, data, target):
        self.model.fit(data, target)

    def predict(self, test):
        return self.model.predict_proba(test)

class ABClassifier(Model):
    '''
    Adaptive Boosting Classifier
    Boosting an initial result of 1-depth decision tress classifier
    '''
    def __init__(self):
        Model.__init__(self)
        self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)

    def predict(self, test):
        return self.model.predict_proba(test)

class GBClassifier(Model):
    '''
    Gradient Boosting Classifier
    '''
    def __init__(self):
        Model.__init__(self)
        self.model = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
    def predict(self, test):
        return self.model.predict_proba(test)

class ExTrClassifier(Model):
    '''
    Extra Trees Classifier
    Can be used to find the importance of features
    '''
    def __init__(self):
        Model.__init__(self)
        self.model = ExtraTreesClassifier(n_estimators=30, random_state=0)

    def predict(self, test):
        return self.model.predict_proba(test)

    def featureImp(self, data):
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(data.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

class VtClassifier(Model):
    '''
    Voting Classfier
    '''

    def __init__(self, *args):
        Model.__init__(self)
        self.modelIndex = ['GNB', 'SVClassifier', 'LRModel', 'ABClassifier', 'GBClassifier']
        self.models = []
        self.estimators = []
        for arg in args:
            index = self.modelIndex.index(arg)
            if index == 0:
                self.models.append(Model())
                self.estimators.append((arg, Model().model))
            elif index == 1:
                self.models.append(SVClassifier())
                self.estimators.append((arg, SVClassifier().model))
            elif index == 2:
                self.models.append(LRModel())
                self.estimators.append((arg, LRModel().model))
            elif index == 3:
                self.models.append(ABClassifier())
                self.estimators.append((arg, ABClassifier().model))
            elif index == 4:
                self.models.append(GBClassifier())
                self.estimators.append((arg, GBClassifier().model))
        self.model = VotingClassifier(estimators=self.estimators, voting='hard')

    def train(self, data, target):
        for model in self.models:
            model.train(data, target)
        self.model.fit(data, target)

    def predict(self, test):
        return self.model.predict_proba(test)





