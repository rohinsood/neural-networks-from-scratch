from accuracy import Accuracy
import numpy as np

class Accuracy_Regression(Accuracy):

    def __init__(self):
        #create precision
        self.precision = None

    # Calculates precision value
    # based on passed in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision