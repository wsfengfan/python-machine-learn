import numpy as np
class LinearRegressionByMyself(object):
    
    def __init__(self, learning=0.001, epoch=20):
        self.learning = learning
        self.epoch = epoch
        
    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.cont_list = []
        
        for i in range(self.epoch):
            output = self.Regression_input(X)
            error = (y - output)
            self.w[1:] += self.learning * X.T.dot(error)
            self.w[0] += self.learning * error.sum()
            cost = (error ** 2) .sum() / 2.0
            self.cost_list.apend(cost)
            
        return self
    
    def Regression_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]
    
    def predict(self, X):
        return self.Regression_input[X]