# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 17:59:18 2018

@author: Kaai
"""
import numpy as np
import pandas as pd


class SVC:
    '''
    Support Vector Classification.

    Implementation based on lecture notes from CS 6350-001 at the University of
    Utah.

    Parameters
    ----------------------
    max_epochs : int, optional (defatul=10)

    C : float, optional (default=1)
        Penalty parameter C of the error term.

    gamma_type : int, optional (default=1)
        if gamma_type=1 then gamma_t = gamma / (1 + gamma * t / d),
        otherwise gamma_t = gamma / (1 + t)

    gamma : float, optional (default='auto')
        if gamma is 'auto' then 1/n features will be used

    d : float, optional (default = 1)

    Attributes :
    ------------------------
    weights_ : array-like, shape = (n_features, )
        plain weights

    intecept_ : float
    '''
    def __init__(self, max_epochs=10, C=1.0, gamma_type=1, gamma='auto', d=1):
        self.max_epochs = max_epochs
        self.C = C
        self.gamma = gamma
        self.d = d
        self.gamma_type = gamma_type
        self.w_sub_gradient = []

    def fit(self, X, y):
        '''
        Parameters
        ----------------------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_ features is the number of features.

        y : array-like, shape (n_samples,)
            Target values (class labels)
        '''

        self.validate_X_y(X, y)
        y = np.sign(self.y)
        self.w = np.zeros((self.X.shape[1] + 1, 1))
        self.N = self.X.shape[0]
        self.t = 0

        for n in range(self.max_epochs):
            self.epoch()

    def epoch(self):
        '''
        Run though a single epoch shuffling the sample at each run
        '''
        X = self.X.copy()
        y = self.y.copy()
        X, y = self.shuffle(X, y)

        for x_i, y_i in zip(X, y):
            if y_i*self.w.T*x_i.T <= 1:
                self.w = (1 - self.gamma_t()) * self.w + (self.gamma_t() *
                                                          self.C *
                                                          self.N *
                                                          y_i *
                                                          x_i.T)

            else:
                self.w = (1 - self.gamma_t()) * self.w

            self.t += 1
            self.w_sub_gradient.append(self.w - (self.C *
                                              self.N *
                                              y_i *
                                              x_i.T))

    def shuffle(self, X, y):
        df = X
        df['y'] = y
        df.sample(frac=1)

        X = df.drop(['y'], axis=1)
        X.insert(loc=0, column='bias', value=np.ones((X.shape[0])))
        X = np.matrix(X)

        y = df['y']
        y = np.array(y)
        return X, y

    def validate_X_y(self, X, y):
        '''
        Check to make sure that an np.array or pd.DataFrame/Series are used for
        inputing data.
        '''
        self.X = X
        self.y = y
        if isinstance(X, np.ndarray) is True:
            X = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame) is True:
            return
        else:
            print('invalid data type for X')
            return

        if isinstance(y, np.ndarray) is True:
            X = pd.Series(X)
        elif isinstance(y, pd.Series) is True:
            return
        else:
            print('invalid data type for y')
            return

    def update_weight(self):
        '''
        Get the intercept and weight vector for the hyperplane.
        '''
        self.intercept_ = self.w[0]
        self.weights_ = self.w[1:]

    def gamma_t(self):
        '''
        Grab the current gamma t for each updated weight vector.
        '''
        if self.gamma == 'auto':
            self.gamma = 1 / self.X.shape[1]
        if self.gamma_type == 1:
            gamma_t = self.gamma / (1 + self.gamma / self.d * self.t)
        if self.gamma_type == 'special':
            self.gamma = .01
            gamma_t = self.gamma/(2 * (self.t+1))
        else:
            gamma_t = self.gamma / (1 + self.t)
        return gamma_t

    def predict(self, X):
        '''
        Parameters
        ----------------------
        X : array-like, shape (n_samples, n_features)
            Test vectors, where n_samples is the number of samples and
            n_ features is the number of features.

        Returns
        -------------
        prediction : array-like, shape (n_samples)
        '''
        X = X.copy()
        X.insert(loc=0, column='bias', value=np.zeros((X.shape[0])))
        X = np.matrix(X).T
        prediction = np.array(np.sign(self.w.T * X).T).ravel()
        return prediction

    def score(self, X, y):
        X = X.copy()
        diff = self.predict(X) - y
        incorrect = sum(abs(diff))/2
        accuracy = 1 - (incorrect / X.shape[0])
        return accuracy
        

if __name__ == '__main__':
    Xx = pd.DataFrame([[0.5, -1, 0.3], [-1, -2, -2], [1.5, 0.2, -2.5]])
    yy = pd.Series([1, -1, 1])
    svc = SVC(gamma_type='special')
    svc.fit(Xx, yy)























