
"""
class for statistical calculations
"""
class Statistics:
    """
    calculates precision
    :param  lables: real labels of each movie
            predictions: predicted labels of each movie
    :return: precision - the precision of our predictions
    """
    def precision(self,labels,predictions):
        countTP = 0
        countFP = 0
        for movie in range(len(labels)):
            if(labels[movie] == 1 and predictions[movie] == 1):
                countTP = countTP + 1
            if(predictions[movie] == 1 and labels[movie] == 0 ):
                countFP = countFP + 1
        precision = countTP/(countFP + countTP)
        return precision

    """
       calculates recall
       :param  lables: real labels of each movie
               predictions: predicted labels of each movie
       :return: recall - the recall of our predictions
       """
    def recall(self, labels,predictions):
        countTP = 0
        countFN = 0
        for movie in range(len(labels)):
            if (labels[movie] == 1 and predictions[movie] == 1):
                countTP = countTP + 1
            if (predictions[movie] == 0 and labels[movie] == 1):
                countFN = countFN + 1
        recall = countTP / (countFN + countTP)
        return recall

    """
       calculates accuracy
       :param  lables: real labels of each movie
               predictions: predicted labels of each movie
       :return: accuracy - the accuracy of our predictions
       """
    def accuracy(self,labels,predictions):
        count = 0
        for movie in range(len(labels)):
            if(labels[movie] == predictions[movie]):
                count= count+1
        accuracy = count/len(labels)
        return accuracy