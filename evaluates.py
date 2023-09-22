# Evaluation function
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve


def evaluates(y_test, y_pred):
    auc = metrics.roc_auc_score(y_test, y_pred)

    aupr = average_precision_score(y_test, y_pred)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    auprc = metrics.auc(recall, precision)

    pro = [1 if index >= 0.5 else 0 for index in y_pred]

    pre = metrics.precision_score(y_test, pro)

    f1 = metrics.f1_score(y_test, pro)

    rec = metrics.recall_score(y_test, pro)

    acc = metrics.accuracy_score(y_test, pro)

    print(confusion_matrix(y_test, pro))
    return pre, acc, rec, f1, auc, aupr, auprc




def get_class_weight(response):
    n_samples =response['response'].values

    print(len(n_samples),n_samples.sum(),(len(n_samples) -n_samples.sum()))

    x_0 =  len(n_samples) / (2*  (len(n_samples) -n_samples.sum()))
    x_1 =  len(n_samples) / (2*  n_samples.sum())

    print(x_0,x_1)

    return x_0,x_1