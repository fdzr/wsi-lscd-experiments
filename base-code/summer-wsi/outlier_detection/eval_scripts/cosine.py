from scipy.spatial import distance

class Cos:

    def __init__(self):
        pass

    def fit(self, train):
        self.train = train

    def predict(self, test):
        temp = []
        for i, y in enumerate(self.train):
            d = distance.cosine(test, y)
            temp.append(d)
        return min(temp) #-1*max(temp)
