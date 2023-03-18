import numpy as np 
import pandas as pd 	


class  NaiveBayes:
    def __init__(self, features, count_d = {}):
        """
        Define some attributes which need to be used in other functions
        For example, 
        self.features = features
        then, you can call self.features in other functions
        self.features will always point to the same object
        """
        self.features = features
        self.count_d = count_d


    def fit(self, x_train, y_train):
        """
        Predict based on feature x with N samples and D dimensions
        Params:
            x: array, (N, D)
        Return:
            None
        """  

        # Counts
        self.classes = np.unique(y_train)
        for c in self.classes:
            d_tmp = {}
            d_tmp['Total'] = 0
            tmpx = [i for i in range(len(y_train)) if y_train[i] == c]
            for i in tmpx:
                d_tmp['Total'] += 1
                for j in range(len(x_train[i])):                    
                    d_tmp.setdefault(self.features[j], {})                    
                    d_tmp[self.features[j]].setdefault(x_train[i][j], 0)
                    d_tmp[self.features[j]][x_train[i][j]] += 1
            self.count_d[c] = d_tmp
        # print(self.count_d)

    def predict(self, x):
        """
        Predict based on feature x with N samples and D dimensions
        Params:
            x: array, (N, D)
        Return:
            pred: array, (N,)
        """

        # 計算先驗機率
        total_train = 0
        for c in self.classes:
            total_train += self.count_d[c]['Total']

        # 計算各組數據的y
        y_pred = np.empty((x.shape[0],), dtype = object)
        for i in range(len(x)):
            p_tmp = []
            for c in self.classes:
                p_c = self.count_d[c]['Total'] / total_train
                p = 1
                for j in range(len(x[i])):
                    p = p * self.count_d[c][self.features[j]][x[i][j]] / self.count_d[c]['Total']
                p = p * p_c
                p_tmp.append(p)
            p_tmp = [i / np.sum(p_tmp) for i in p_tmp]
            y = self.classes[p_tmp.index(max(p_tmp))]
            y_pred[i] = y
            self.y_pred = y_pred
        
        print("predict something based on the feature x with shape", x.shape)
        print(f'prediction = {y_pred}')

        return y_pred

    def evaluate(self, x_test, y_test):        
        correct = len([i for i in range(len(self.y_pred)) if self.y_pred[i] == y_test[i]])
        precision = correct / len(self.y_pred) * 100
        print(f'precision = {round(precision,3)} %')

        return precision


if __name__ == "__main__":
    path = "./data.xlsx"
    df = pd.read_excel(path, header=0)
    features = list(df.columns)[:-1]
    x_train = df.iloc[:, :-1].values
    y_train = df.iloc[:, -1].values
    # print(features)
    # print(x_train)
    # print(y_train)      
    clf = NaiveBayes(features)
    clf.fit(x_train, y_train)
    x_test, y_test = x_train, y_train
    clf.predict(x_test)
    clf.evaluate(x_test, y_test)
    print


    x = np.empty((1,4), dtype = object) 
    x[0,0] = 'Obvious'
    x[0,1] = 'Yes'
    x[0,2] = 'No'
    x[0,3] = 'No'

    clf.predict(x)
    
