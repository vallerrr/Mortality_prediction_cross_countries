import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import minimize

def acc(y, yhat):
    return np.sum(np.abs(y - yhat) / y) / len(yhat)


def r2(y, yhat):
    return 1 - (np.sum(y - yhat) ** 2 / np.sum(y - np.mean(y)) ** 2)


def f1_score(confusion):
    ##########################################
    # confusion matrix is like
    # TN   |   FP  |
    # FN   |   TP  |
    ##########################################
    tp = confusion[1][1]
    fp = confusion[0][1]
    fn = confusion[1][0]
    tn = confusion[0][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*(precision*recall)/(precision+recall)
    print("recall is {}, precision is {}, f1 score is {}".format(recall, precision, f1))

class metric():
    def __init__(self, model):
        super(metric, self).__init__()

        train_set_data = model.y_train
        train_set_pred = model.train_set_predict
        train_set_weight = model.train_sample_weight

        test_set_data = model.y_test
        test_set_pred = model.test_set_predict
        test_set_weight = model.test_sample_weight

        self.train_f1_score = metrics.f1_score(train_set_data, train_set_pred, sample_weight=train_set_weight)
        self.train_confusion = metrics.confusion_matrix(train_set_data, train_set_pred, sample_weight=train_set_weight)
        self.train_roc_auc_score = metrics.roc_auc_score(train_set_data, train_set_pred, sample_weight=train_set_weight)

        self.valid_f1_score = metrics.f1_score(test_set_data, test_set_pred, sample_weight=test_set_weight)
        self.valid_confusion = metrics.confusion_matrix(test_set_data, test_set_pred, sample_weight=test_set_weight)
        self.valid_roc_auc_score = metrics.roc_auc_score(test_set_data, test_set_pred, sample_weight=test_set_weight)

def seed_evaluate_metric(true, pred, pred_prob, train, weight):
    metric_dict = {}
    fpr, tpr, threshold = metrics.roc_curve(true, pred, pos_label=1)
    metric_dict['auc'] = metrics.auc(fpr, tpr)  # pr_auc
    metric_dict['f1'] = metrics.f1_score(true, pred, sample_weight=weight)
    metric_dict['efron_r2'] = efron_rsquare(true, pred_prob)
    metric_dict['ffc_r2'] = ffc_rsquare(true, pred_prob, train)  # TODO: pred_prob or pred?
    metric_dict['brier'] = brier(true, pred_prob)
    metric_dict['pr_no_skill'] = len(true[true == 1]) / len(true)
    return metric_dict


def ll(x, p):
    """x is the truth, p is the guess"""
    z = (np.log(p) * x) + (np.log(1 - p) * (1 - x))
    return np.exp(np.sum(z) / len(z))


def get_w(a, guess=0.5, bounds=[(0.001, 0.999)]):
    """argmin calc for 'w'"""
    res = minimize(minimize_me, guess, args=a,
                   options={'ftol': 0, 'gtol': 1e-09},
                   method='L-BFGS-B', bounds=bounds)
    return res['x'][0]


def minimize_me(p, a):
    """ function to be minimized"""
    # abs(p*log(p)+(1-p)*log(1-p)-log(a))
    return abs((p * np.log(p)) + ((1 - p) * np.log(1 - p)) - np.log(a))


def get_ew(w0, w1):
    """calculate the e(w) metric from w0 and w1"""
    return (w1 - w0) / w0

def imv(true,train,pred_prob):
    return get_ew(get_w(ll(true, np.mean(train))), get_w(ll(true, pred_prob)))

def evaluate_metric(true, pred, pred_prob, train, weight):
    metric_dict = {}
    fpr, tpr, threshold = metrics.roc_curve(true, pred, pos_label=1)
    metric_dict['auc'] = metrics.auc(fpr, tpr)  # pr_auc
    metric_dict['f1'] = metrics.f1_score(true, pred, sample_weight=weight)
    metric_dict['efron_r2'] = efron_rsquare(true, pred_prob)
    metric_dict['ffc_r2'] = ffc_rsquare(true, pred_prob, train)  # TODO: pred_prob or pred?
    metric_dict['brier'] = brier(true, pred_prob)
    metric_dict['pr_no_skill'] = len(true[true == 1]) / len(true)
    metric_dict['roc_auc'] = metrics.roc_auc_score(true, pred_prob, sample_weight=weight)
    metric_dict['imv'] = get_ew(get_w(ll(true, np.mean(train))), get_w(ll(true, pred_prob)))
    return metric_dict



def pseudo_rmse(pred, true, train):
    numerator = np.sum((true-pred).pow(2))
    denominator = np.sum((true - train.mean()).pow(2))
    rmse = 1-(numerator/denominator)
    return rmse

def cross_entropy(pred,true):
    if true == 1:
        return -np.log(pred)
    else:
        return -np.log(1 - pred)

def ffc_rsquare(true, pred, train):
    # TODO：pred here should be probs or true lables?
    n = float(len(true))
    t1 = np.sum(np.power(true - pred, 2.0))
    t2 = np.sum(np.power((true - (np.sum(train) / n)), 2.0))
    return 1.0 - (t1 / t2)

def brier(true, pred_prob):
    # pred here is the probability
    n = float(len(true))
    t1 = np.sum(np.power(pred_prob-true, 2.0))
    return t1 / n

def efron_rsquare(true, pred_prob):
    # here the pred is predicted probabilities
    n = float(len(true))
    t1 = np.sum(np.power(true - pred_prob, 2.0))
    t2 = np.sum(np.power((true - (np.sum(true) / n)), 2.0))
    return 1.0 - (t1 / t2)

def full_log_likelihood(coef, X, y):
    score = np.dot(X, coef).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def null_log_likelihood(coef, X, y):
    z = np.array([w if i == 0 else 0.0 for i, w in enumerate(coef.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def mcfadden_rsquare(coef, X, y):
    return 1.0 - (full_log_likelihood(coef, X, y) / null_log_likelihood(coef, X, y))

def mcfadden_adjusted_rsquare(coef, X, y):
    k = float(X.shape[1])
    return 1.0 - ((full_log_likelihood(coef, X, y) - k) / null_log_likelihood(coef, X, y))


