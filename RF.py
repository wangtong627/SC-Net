#  测试一下决策树效果
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Metrics import calculate_metrics
from DataPreprocessing import X_train, X_test, y_train, y_test
from OverSampling import X_train_over, y_train_over, X_train_total, y_train_total

model = RandomForestClassifier()  # 决策树
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
cm, accuracy, recall, precision, f1, classification_report = calculate_metrics(expected, predicted)


print("Accuracy", "%.4f" % accuracy)
print("precision", "%.4f" % precision)
print("recall", "%.4f" % recall)
print("f-score", "%.4f" % f1)
