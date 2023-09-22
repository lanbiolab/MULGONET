import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree



compare_models = [
    {
        'type': 'sgd',
        'id': 'L2 Logistic Regression',
        'params': {'loss': 'log', 'penalty': 'l2', 'alpha': 0.01}
    },

    {
        'type': 'svc',
        'id': 'RBF Support Vector Machine ',
        'params': {'kernel': 'rbf', 'gamma': 0.001, 'probability': True}
    },

    {
        'type': 'svc', 'id':
        'Linear Support Vector Machine ',
        'params': {'kernel': 'linear', 'C': 0.1, 'probability': True, }
    },

    {
        'type': 'random_forest',
        'id': 'Random Forest',
        'params': {'max_depth': None, 'n_estimators': 50, 'bootstrap': False}
    },

    {
        'type': 'adaboost',
        'id': 'Adaptive Boosting',
        'params': {'learning_rate': 0.1, 'n_estimators': 50}
    },

    {
        'type': 'decision_tree',
        'id': 'Decision Tree',
        'params': {'min_samples_split': 10, 'max_depth': 10}
    },

]






# eval
def get_metrics(true_score, pre_score, pre_probe):
    fpr, tpr, thresholds = metrics.roc_curve(true_score, pre_probe, pos_label=1)

    auc = metrics.auc(fpr, tpr)

    aupr = average_precision_score(true_score, pre_probe)

    pre, rec, thresholds = precision_recall_curve(true_score, pre_probe)
    auprc = metrics.auc(rec, pre)

    accuracy = accuracy_score(true_score, pre_score)

    f1 = metrics.f1_score(true_score, pre_score)

    precision = metrics.precision_score(true_score, pre_score)

    recall = metrics.recall_score(true_score, pre_score)

    return precision, accuracy, recall, f1, auc, aupr, auprc


from sklearn.linear_model import SGDClassifier



#sgd
def Creat_SGD(whole_data_x, whole_data_y, train_index, test_index,class_weight):
    model = SGDClassifier(class_weight = class_weight,**compare_models[0]['params'])

    model.fit(whole_data_x[train_index], whole_data_y[train_index])

    true_score = whole_data_y[test_index]

    pre_score = model.predict(whole_data_x[test_index])

    pre_probe = model.predict_proba(whole_data_x[test_index])[:, 1]

    precision, accuracy, recall, f1, auc, aupr, auprc = get_metrics(true_score, pre_score, pre_probe)

    return precision, accuracy, recall, f1, auc, aupr, auprc


from sklearn.ensemble import RandomForestClassifier




def Creat_RDF(whole_data_x, whole_data_y, train_index, test_index,class_weight):
    model = RandomForestClassifier(class_weight = class_weight,**compare_models[3]['params'])

    model.fit(whole_data_x[train_index], whole_data_y[train_index])

    true_score = whole_data_y[test_index]

    pre_score = model.predict(whole_data_x[test_index])

    pre_probe = model.predict_proba(whole_data_x[test_index])[:, 1]

    precision, accuracy, recall, f1, auc, aupr, auprc = get_metrics(true_score, pre_score, pre_probe)

    return precision, accuracy, recall, f1, auc, aupr, auprc


from sklearn.linear_model import LogisticRegression




def Creat_LR(whole_data_x, whole_data_y, train_index, test_index):
    model = LogisticRegression()

    model.fit(whole_data_x[train_index], whole_data_y[train_index])

    true_score = whole_data_y[test_index]

    pre_score = model.predict(whole_data_x[test_index])

    pre_probe = model.predict_proba(whole_data_x[test_index])[:, 1]

    precision, accuracy, recall, f1, auc, aupr, auprc = get_metrics(true_score, pre_score, pre_probe)

    return precision, accuracy, recall, f1, auc, aupr, auprc






def Creat_BOOST(whole_data_x, whole_data_y, train_index, test_index):
    model = AdaBoostClassifier(**compare_models[4]['params'])

    model.fit(whole_data_x[train_index], whole_data_y[train_index])

    true_score = whole_data_y[test_index]

    pre_score = model.predict(whole_data_x[test_index])

    pre_probe = model.predict_proba(whole_data_x[test_index])[:, 1]

    precision, accuracy, recall, f1, auc, aupr, auprc = get_metrics(true_score, pre_score, pre_probe)

    return precision, accuracy, recall, f1, auc, aupr, auprc


from sklearn import tree




def Creat_DTC(whole_data_x, whole_data_y, train_index, test_index):
    DTC_model = tree.DecisionTreeClassifier()

    DTC_model.fit(whole_data_x[train_index], whole_data_y[train_index])

    true_score = whole_data_y[test_index]

    pre_score = DTC_model.predict(whole_data_x[test_index])

    pre_probe = DTC_model.predict_proba(whole_data_x[test_index])[:, 1]

    precision, accuracy, recall, f1, auc, aupr, auprc = get_metrics(true_score, pre_score, pre_probe)

    return precision, accuracy, recall, f1, auc, aupr, auprc


def Creat_RBFSVM(whole_data_x, whole_data_y, train_index, test_index,class_weight):

    RBFSVM = NuSVC(class_weight = class_weight,**compare_models[1]['params'])

    RBFSVM.fit(whole_data_x[train_index], whole_data_y[train_index])

    true_score = whole_data_y[test_index]

    pre_score = RBFSVM.predict(whole_data_x[test_index])

    pre_probe = RBFSVM.predict_proba(whole_data_x[test_index])[:, 1]

    precision, accuracy, recall, f1, auc, aupr, auprc = get_metrics(true_score, pre_score, pre_probe)

    return precision, accuracy, recall, f1, auc, aupr, auprc




