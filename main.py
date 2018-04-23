# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:07:58 2018

@author: Kaai
"""

import pandas as pd
import numpy as np
from SVC import SVC

df_train = pd.read_csv('train.csv', header=None)
df_test = pd.read_csv('test.csv', header=None)

columns_train = df_train.columns.values
X_train = df_train[columns_train[:-1]]
y_train = df_train[columns_train[-1]]

columns_test = df_test.columns.values
X_test = df_test[columns_test[:-1]]
y_test = df_test[columns_test[-1]]

y_train.replace(to_replace=0, value=-1, inplace=True)
y_test.replace(to_replace=0, value=-1, inplace=True)


C = [10 / 873, 100 / 873, 300 / 873, 500 / 873, 700 / 873]

# %%
print('\n\n--------------\nHOMEWORK 4\n--------------')
print('\n3. part a')
errors_a = []
for value in C:
    svc = SVC(C=value, gamma_type=1, max_epochs=50, gamma=0.1, d=0.15)
    svc.fit(X_train, y_train)

    y_predict = svc.predict(X_test)
    score = svc.score(X_test, y_test)
    print('C = ' + str.format('{0:.4f}', value) +
          ', error = ' + str.format('{0:.4f}', 1-score))
    errors_a.append([value, 1-score, np.around(svc.w, 4)])

# %%

print('\n3. part b')
errors_b = []
for value in C:
    svc = SVC(C=value, gamma_type=2, max_epochs=50, gamma=0.1, d=0.15)
    svc.fit(X_train, y_train)

    y_predict = svc.predict(X_test)
    score = svc.score(X_test, y_test)
    print('C = ' + str.format('{0:.4f}', value) +
          ', error = ' + str.format('{0:.4f}', 1-score))
    errors_b.append([value, 1-score, np.around(svc.w, 4)])

# %%

print('\n3. part c')
for err_a, err_b in zip(errors_a, errors_b):
    diff = err_a[2]-err_b[2]
    print('\nweight vector differences:\n' + str(diff.ravel()))
    print('--------------------------------------------------\n\t' \
          'test error differences:', str.format('{0:.4f}', err_a[1]-err_b[1]))

