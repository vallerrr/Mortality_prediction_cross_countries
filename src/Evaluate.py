import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import minimize
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score
def acc(y, yhat):
    return np.sum(np.abs(y - yhat) / y) / len(yhat)


def r2(y, yhat):
    return 1 - (np.sum(y - yhat) ** 2 / np.sum(y - np.mean(y)) ** 2)



def seed_evaluate_metric(true, pred, pred_prob, train):
    metric_dict = {}
    fpr, tpr, threshold = metrics.roc_curve(true, pred, pos_label=1)
    metric_dict['auc'] = metrics.auc(fpr, tpr)  # pr_auc
    metric_dict['f1'] = metrics.f1_score(true, pred)
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

def evaluate_metric(true, pred, pred_prob, train):
    metric_dict = {}
    fpr, tpr, threshold = metrics.roc_curve(true, pred, pos_label=1)
    metric_dict['auc'] = metrics.auc(fpr, tpr)  # pr_auc
    metric_dict['f1'] = metrics.f1_score(true, pred)
    metric_dict['efron_r2'] = efron_rsquare(true, pred_prob)
    metric_dict['ffc_r2'] = ffc_rsquare(true, pred_prob, train)  # TODO: pred_prob or pred?
    metric_dict['brier'] = brier(true, pred_prob)
    metric_dict['pr_no_skill'] = len(true[true == 1]) / len(true)
    metric_dict['roc_auc'] = roc_auc_score(true, pred_prob)
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
    # MSE
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


class metric():
    def __init__(self, model):
        super(metric, self).__init__()

        y_train = model.y_train
        train_set_pred = model.train_set_predict

        y_test = model.y_test
        y_test_pred_label = model.test_set_predict
        y_test_pred_prob = model.test_set_predict_prob
        if model.samp_weight_control:

            train_set_weight = model.train_sample_weight
            test_set_weight = model.test_sample_weight
            # train set
            self.train_f1_score_label = metrics.f1_score(y_train, train_set_pred, sample_weight=train_set_weight)
            self.train_confusion_label = metrics.confusion_matrix(y_train, train_set_pred, sample_weight=train_set_weight)
            self.train_roc_auc_score_label = metrics.roc_auc_score(y_train, train_set_pred, sample_weight=train_set_weight)

            # test set calculated with label
            self.test_f1_score_label = metrics.f1_score(y_test, y_test_pred_label, sample_weight=test_set_weight)
            self.test_confusion_label = metrics.confusion_matrix(y_test, y_test_pred_label, sample_weight=test_set_weight)
            self.test_roc_auc_score_label = metrics.roc_auc_score(y_test, y_test_pred_label, sample_weight=test_set_weight)

            # test set calculated with prob
            # pr part
            self.precision_test, self.recall_test, _ = precision_recall_curve(y_test, y_test_pred_prob, sample_weight=test_set_weight)
            self.pr_f1, self.pr_auc = f1_score(y_test, y_test_pred_label, sample_weight=test_set_weight), auc(self.recall_test, self.precision_test)
            self.pr_no_skill = len(y_test[y_test == 1]) / len(y_test)
            self.r2_score = r2(model.y_test, y_test_pred_label)
            self.brier = brier(model.y_test, y_test_pred_prob)
            # roc
            self.auc_score = roc_auc_score(y_test, y_test_pred_prob, sample_weight=test_set_weight)
            self.imv = imv(true=y_test,train=y_train,pred_prob=y_test_pred_prob)
        else:
            self.train_f1_score_label = metrics.f1_score(y_train, train_set_pred)
            self.train_confusion_label = metrics.confusion_matrix(y_train, train_set_pred)
            self.train_roc_auc_score_label = metrics.roc_auc_score(y_train, train_set_pred)

            # test set calculated with label
            self.test_f1_score_label = metrics.f1_score(y_test, y_test_pred_label)
            self.test_confusion_label = metrics.confusion_matrix(y_test, y_test_pred_label)
            self.test_roc_auc_score_label = metrics.roc_auc_score(y_test, y_test_pred_label)

            # test set calculated with prob
            # pr part
            self.precision_test, self.recall_test, _ = precision_recall_curve(y_test, y_test_pred_prob)
            self.pr_f1, self.pr_auc = f1_score(y_test, y_test_pred_label), auc(
                self.recall_test, self.precision_test)
            self.pr_no_skill = len(y_train[y_train == 1]) / len(y_train)

            self.r_score = r2(y_test, y_test_pred_label)
            self.brier = brier(y_test, y_test_pred_prob)

            self.ffc_r2=ffc_rsquare(y_test, y_test_pred_prob, y_train)
            self.efron_rsquare=efron_rsquare(y_test, y_test_pred_prob)
            # roc
            self.auc_score = roc_auc_score(y_test, y_test_pred_prob)
            self.imv = imv(true=y_test, train=y_train, pred_prob=y_test_pred_prob)
