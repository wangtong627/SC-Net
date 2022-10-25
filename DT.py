#  测试一下决策树效果
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from Metrics import calculate_metrics
from DataPreprocessing import X_train, X_test, y_train, y_test
from OverSampling import X_train_over, y_train_over, X_train_total, y_train_total

model = DecisionTreeClassifier()  # 决策树
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
cm, accuracy, recall, precision, f1, classification_report = calculate_metrics(expected, predicted)
# cm = metrics.confusion_matrix(expected, predicted)
# accuracy = accuracy_score(expected, predicted)
# recall = recall_score(expected, predicted, average='macro')
# precision = precision_score(expected, predicted, average='macro')  # 精确率
# f1 = f1_score(expected, predicted, average='macro')

# print(cm, cm[0][0], cm[0][1])  # 混淆矩阵
# tpr = float(cm[0][0]) / np.sum(cm[0])
# fpr = float(cm[1][1]) / np.sum(cm[1])
# print("%.3f" % tpr)
# print("%.3f" % fpr)
print("Accuracy", "%.4f" % accuracy)
print("precision", "%.4f" % precision)
print("recall", "%.4f" % recall)
print("f-score", "%.4f" % f1)
# print("fpr", "%.3f" % fpr)
# print("tpr", "%.3f" % tpr)