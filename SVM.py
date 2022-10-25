from Metrics import calculate_metrics
from DataPreprocessing import X_train, X_test, y_train, y_test
from OverSampling import X_train_over, y_train_over, X_train_total, y_train_total
from sklearn import svm

# instantiate the model (using the default parameters)
log_clf = svm.SVC()

# fit the model with data
log_clf.fit(X_train, y_train)

expected = y_test
predicted = log_clf.predict(X_test)
cm, accuracy, recall, precision, f1, classification_report = calculate_metrics(expected, predicted)

print("Accuracy", "%.4f" % accuracy)
print("precision", "%.4f" % precision)
print("recall", "%.4f" % recall)
print("f-score", "%.4f" % f1)

print(classification_report)