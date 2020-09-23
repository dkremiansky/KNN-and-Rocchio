from sys import argv
from Data import Data
from Statistics import Statistics
from AlgorithmRunner import AlgorithmRunner
import  numpy as np
import os
from CompetitiveData import CompetitiveData

"""
Loads data from given csv
:param input_path: path to csv file
:return:data - object of the data file
        data_features - the data after preprocess
"""
def load_data():
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)
    data = Data(input_path)
    data_features = data.preprocess()
    return data, data_features

"""
    calculates stastistics parameters for the required learning algorithm
    :param  data - object of the data file
            data_features - the data after preprocess
"""
def question_1(data,data_features):
    print("Question 1:")
    knn_runner = AlgorithmRunner("KNN")
    rocchio_runner = AlgorithmRunner("Rocchio")
    stats = Statistics()

    #KNN calculations
    kfoldKNN = data.split_to_k_folds()
    sumPrecisionKNN = 0
    sumRecallKNN = 0
    sumAccuracyKNN = 0

    for trainKNN, testKNN in kfoldKNN:
        knn_runner.fit(data_features.loc[:, data_features.columns != 'imdb_score'].iloc[trainKNN], data_features['imdb_score'].iloc[trainKNN])
        pred = knn_runner.algorithm.predict(data_features.loc[:, data_features.columns != 'imdb_score'].iloc[testKNN])
        sumPrecisionKNN = stats.precision(labels = np.array(data_features['imdb_score'].iloc[testKNN]).T,predictions = pred) + sumPrecisionKNN
        sumRecallKNN = stats.recall(labels =  np.array(data_features['imdb_score'].iloc[testKNN]),predictions=pred) +sumRecallKNN
        sumAccuracyKNN = stats.accuracy(labels =  np.array(data_features['imdb_score'].iloc[testKNN]),predictions=pred) + sumAccuracyKNN
    print("KNN classifier: ",sumPrecisionKNN/5,",",sumRecallKNN/5, ",",sumAccuracyKNN/5)

    #Rocchio calculations
    kfoldRocciho = data.split_to_k_folds()
    sumPrecisionRocchio = 0
    sumRecallRocchio = 0
    sumAccuracyRocchio = 0
    for trainRocciho, testRocciho in kfoldRocciho:
        rocchio_runner.fit(data_features.loc[:, data_features.columns != 'imdb_score'].iloc[trainRocciho], data_features['imdb_score'].iloc[trainRocciho])
        pred = rocchio_runner.algorithm.predict(data_features.loc[:, data_features.columns != 'imdb_score'].iloc[testRocciho])
        sumPrecisionRocchio = stats.precision(labels = np.array(data_features['imdb_score'].iloc[testRocciho]).T,predictions = pred) + sumPrecisionRocchio
        sumRecallRocchio = stats.recall(labels =  np.array(data_features['imdb_score'].iloc[testRocciho]),predictions=pred) +sumRecallRocchio
        sumAccuracyRocchio = stats.accuracy(labels =  np.array(data_features['imdb_score'].iloc[testRocciho]),predictions=pred) + sumAccuracyRocchio
    print("Rocchio classifier: ",sumPrecisionRocchio/5,",",sumRecallRocchio/5, ",",sumAccuracyRocchio/5)
    print(" ")


if __name__ == '__main__':
    data, data_features = load_data()
    question_1(data,data_features)
