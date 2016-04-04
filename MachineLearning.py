'''
 For COMP 4332, Project 1
 Group 1
 Name: Sirui Xie
 StuID: 20091029
'''

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
import MachineLearningModel

__author__ = "Sirui Xie"

class MachineLearing:
    '''
    A class that offers mutliple machine learning models and functions
    '''

    def __init__(self):
        self.training_data = []
        self.training_labels = []
        self.validation_data = []
        self.validation_labels = []
        self.testing_data = []
        self.testing_labels = []
        self.scaler = []
        self.ml_model = MachineLearningModel.Model()

    def preprocessTrain(self, scaler = 0):
        '''
        preprocess training data
        if sclaer = 0, use StandardScaler
        if scaler = 1, use min-max scaling [0,1]
        '''

        if not self.training_data:
            return
        else:
            if scaler == 0:
                self.scaler = preprocessing.StandardScaler()
            elif scaler == 1:
                self.scaler = preprocessing.MinMaxScaler()

        self.testing_data = self.scaler.fit_transform(self.training_data)

    def preprocessTest(self):
        '''
        preprocess testing data
        '''
        self.testing_data = self.scaler.transform(self.testing_data)

    def splitData(self, trainingData, trainingLabels):
        self.training_data, self.validation_data, self.training_labels, self.validation_labels = cross_validation.train_test_split(trainingData,trainingLabels)

    def crossValidation(self, numXValidation):
        model = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
        return cross_validation.cross_val_score(model, self.data_train, self.labels_train[:,0], cv=numXValidation)

    def trainModel(self, modelNum):
        if modelNum == 1:
            self.ml_model = MachineLearningModel.LRModel()
        elif modelNum == 2:
            self.ml_model = MachineLearningModel.GBClassifier()

        self.ml_model.train(self.training_data, self.training_labels)

    def predict(self):
        return self.ml_model.predict(self.testing_data)
