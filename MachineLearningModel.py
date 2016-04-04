from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

__author__ = "Sirui Xie"

class Model:
    '''
    Ada Boosting Classifier
    '''

    def __init__(self):
        self.model = AdaBoostClassifier()
        return

    def train(self, data, target):
        return

    def predict(self, test):
        return self.model.predict_proba(test)

class LRModel(Model):
    '''
    Logistic Regression Model
    '''
    def __init__(self):
        Model.__init__(self)
        self.model = LogisticRegression()

    def train(self, data, target):
        self.model.fit(data, target[:,0])

class GBClassifier(Model):
    '''
    Gradiant Boostig Classifier
    '''
    def __init__(self):
        Model.__init__()
        self.model = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)

    def train(self, data, target):
        self.model.fit(data, target[:,0])

class GBLRLinear(Model):
    '''
    Linearly combine LRModel and GBClassifier
    '''
    def __init__(self):
        Model.__init__(self)
        self.model1 = LogisticRegression()
        self.model2 = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)

    def train(self, data, target):
        self.model1.fit(data, target[:,0])
        self.model2.fit(data, target[:,0])

    def predict(self, test):
        return 0.5*self.model1.predict_proba(test)+0.5*self.model2.predict_proba(test)

