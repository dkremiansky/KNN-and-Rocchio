from sklearn.neighbors import KNeighborsClassifier , NearestCentroid

"""
class for running algorithms
"""
class AlgorithmRunner:

    """
    initalize the algorithms
    :param  algo_name: the name of the current algorithm
    """
    def __init__(self, algo_name):
        if algo_name == "KNN":
            self.algorithm = KNeighborsClassifier(n_neighbors=10)
        if algo_name == "Rocchio":
            self.algorithm = NearestCentroid()


    """
    call to the fit method from sklearn.neighbours
    :param  train_features: the features that we train on
            train_labels: the labels for the features that we train on
    """
    def fit(self, train_features, train_labels):
        self.algorithm.fit(train_features, train_labels)

    """
    call to the predict method from sklearn.neighbours
    :param  test_features: the features that we test on
    """
    def predict(self, test_features):
        return self.algorithm.predict(test_features)