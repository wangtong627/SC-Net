from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error,
                             mean_absolute_error, roc_curve, classification_report, auc)
from sklearn import metrics


def calculate_metrics(expected, predicted):
    cm = metrics.confusion_matrix(expected, predicted)
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average='macro')  # None 'micro', 'macro', 'samples', 'weighted', 'binary'
    precision = precision_score(expected, predicted, average='macro')  # 精确率
    f1 = f1_score(expected, predicted, average='macro')
    classification_report = metrics.classification_report(expected, predicted)
    return cm, accuracy, recall, precision, f1, classification_report
