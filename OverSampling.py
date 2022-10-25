#---------------------------重采样-----------------------------------
from collections import Counter
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from DataPreprocessing import X_train, X_test, y_train, y_test

oversample = ADASYN()
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)
# print(f'重采样后的数据规模')
# print(f'X_train shape is {X_train_over.shape}')
# print(f'Y_train shape is {y_train_over.shape}')

X_train_total = np.concatenate((X_train, X_train_over))
y_train_total = np.concatenate((y_train, y_train_over))
# print(X_train_total.shape)
#
# def print_label_dist(label_col):
#     c = Counter(label_col)
#     print(f'label is {c}')
#
#
# print_label_dist(y_train)
# print_label_dist(y_train_over)
# print_label_dist(y_train_total)
