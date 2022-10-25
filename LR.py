from Metrics import calculate_metrics
from DataPreprocessing import X_train, X_test, y_train, y_test
# from DataPreprocessing_v2 import X_train, X_test, y_train, y_test
# from OverSampling import X_train_over, y_train_over, X_train_total, y_train_total
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
log_clf = LogisticRegression()

# fit the model with data
log_clf.fit(X_train, y_train)

expected = y_test
predicted = log_clf.predict(X_test)
cm, accuracy, recall, precision, f1, classification_report = calculate_metrics(expected, predicted)

print("Accuracy")
print(accuracy)
print(f"precision\n{precision}")
print(f"recall\n{recall}")
print(f"f-score\n{f1}")

print(classification_report)