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

class MachineLearning:
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


    def preprocess(self, df, scaler=0):
        '''
        preprocess training data
        if sclaer = 0, use StandardScaler
        if scaler = 1, use min-max scaling [0,1]
        '''

        if not len(df):
            return
        else:
            print('preprocessing training data ...')
            if scaler == 0:
                self.scaler = preprocessing.StandardScaler()
            elif scaler == 1:
                self.scaler = preprocessing.MinMaxScaler()

        return self.scaler.fit_transform(df)

    def preprocessTrain(self, scaler = 0):
        '''
        preprocess training data
        if sclaer = 0, use StandardScaler
        if scaler = 1, use min-max scaling [0,1]
        '''

        if not len(self.training_data):
            return
        else:
            print('preprocessing training data ...')
            if scaler == 0:
                self.scaler = preprocessing.StandardScaler()
            elif scaler == 1:
                self.scaler = preprocessing.MinMaxScaler()

            self.training_data = self.scaler.fit_transform(self.training_data)

    def preprocessTest(self):
        '''
        preprocess testing data
        '''
        print('preprocessing testing data ...')
        self.testing_data = self.scaler.transform(self.testing_data)

    def splitData(self, trainingData, trainingLabels):
        self.training_data, self.validation_data, self.training_labels, self.validation_labels = cross_validation.train_test_split(trainingData,trainingLabels)

    def crossValidation(self, numXValidation):
        model = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
        return cross_validation.cross_val_score(model, self.training_data, self.training_labels[:,0], cv=numXValidation, scoring='roc_au')

    def trainSingleModel(self, model_name='GNB'):
        modelIndex = ['GNB', 'SVClassifier', 'LRModel', 'ABClassifier', 'GBClassifier', 'ExTrClassifier']
        modelNum = modelIndex.index(model_name)
        if modelNum == 1:
            self.ml_model = MachineLearningModel.SVClassifier()
        elif modelNum == 2:
            self.ml_model = MachineLearningModel.LRModel()
        elif modelNum == 3:
            self.ml_model = MachineLearningModel.ABClassifier()
        elif modelNum == 4:
            self.ml_model = MachineLearningModel.GBClassifier()
        elif modelNum == 5:
            self.ml_model = MachineLearningModel.ExTrClassifier()

        self.ml_model.train(self.training_data, self.training_labels)

    def trainVtClassifier(self, *model_names):
        self.ml_model = MachineLearningModel.VtClassifier(*model_names)
        self.ml_model.train(self.training_data, self.training_labels)

    def predict(self):
        return self.ml_model.predict(self.testing_data)

    def featImportance(self):
        tempModel = MachineLearningModel.ExTrClassifier()
        tempModel.train(self.training_data, self.training_labels)
        tempModel.featureImp(self.training_data)