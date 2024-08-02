import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score,fbeta_score
beta = 2
def acc(y, yhat):
    return np.sum(np.abs(y - yhat) / y) / len(yhat)

def r2(y, yhat):
    return 1 - (np.sum(y - yhat) ** 2 / np.sum(y - np.mean(y)) ** 2)



def seed_evaluate_metric(true, pred, pred_prob, train):
    metric_dict = {}
    fpr, tpr, threshold = metrics.roc_curve(true, pred, pos_label=1)
    metric_dict['auc'] = metrics.auc(fpr, tpr)  # pr_auc
    metric_dict['f1'] = f1_score(true, pred,average='micro')
    metric_dict['fb'] =fbeta_score(true, pred, beta=beta)
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
    metric_dict['f1'] = metrics.f1_score(true, pred,average='micro')
    metric_dict['fb'] = metrics.fbeta_score(true, pred, beta=beta)
    metric_dict['efron_r2'] = efron_rsquare(true, pred_prob)
    metric_dict['ffc_r2'] = ffc_rsquare(true, pred_prob, train)  # TODO: pred_prob or pred?
    metric_dict['brier'] = brier(true, pred_prob)
    metric_dict['pr_no_skill'] = len(true[true == 1]) / len(true)
    metric_dict['roc_auc'] = roc_auc_score(true, pred_prob)
    metric_dict['imv'] = get_ew(get_w(ll(true, np.mean(train))), get_w(ll(true, pred_prob)))
    return metric_dict

def evaluate_cox_pred(true,pred_prob):
    metric_dict = {}
    metric_dict['efron_r2'] = efron_rsquare(true, pred_prob)
    metric_dict['brier'] = brier(true, pred_prob)
    # metric_dict['roc_auc'] = roc_auc_score(true, pred_prob)
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
        if len(model.y_train)>=0 & len(model.train_set_predict)>=0:
            train_set_control=True
            train_set_pred = model.train_set_predict
        else:
            train_set_control = False


        y_train = model.y_train
        y_test = model.y_test
        y_test_pred_label = model.test_set_predict
        y_test_pred_prob = model.test_set_predict_prob


        if model.samp_weight_control:
            test_set_weight = model.test_sample_weight
            if train_set_control:
                train_set_weight = model.train_sample_weight

                # train set
                self.train_f1_score_label = metrics.f1_score(y_train, train_set_pred, sample_weight=train_set_weight,average='micro')
                self.train_confusion_label = metrics.confusion_matrix(y_train, train_set_pred, sample_weight=train_set_weight)
                self.train_roc_auc_score_label = metrics.roc_auc_score(y_train, train_set_pred, sample_weight=train_set_weight)

            # test set calculated with label
            self.test_f1_score_label = metrics.f1_score(y_test, y_test_pred_label, sample_weight=test_set_weight,average='micro')
            self.test_fb_score_label = metrics.fbeta_score(y_test, y_test_pred_label, sample_weight=train_set_weight,beta=beta)
            self.test_confusion_label = metrics.confusion_matrix(y_test, y_test_pred_label, sample_weight=test_set_weight)
            self.test_roc_auc_score_label = metrics.roc_auc_score(y_test, y_test_pred_label, sample_weight=test_set_weight)

            # test set calculated with prob
            # pr part
            self.precision_test, self.recall_test, _ = precision_recall_curve(y_test, y_test_pred_prob, sample_weight=test_set_weight)
            self.pr_f1, self.pr_auc = f1_score(y_test, y_test_pred_label, sample_weight=test_set_weight,average='micro'), auc(self.recall_test, self.precision_test)
            self.pr_no_skill = len(y_test[y_test == 1]) / len(y_test)
            self.r2_score = r2(model.y_test, y_test_pred_label)
            self.brier = brier(model.y_test, y_test_pred_prob)
            # roc
            self.auc_score = roc_auc_score(y_test, y_test_pred_prob, sample_weight=test_set_weight)
            self.imv = imv(true=y_test,train=y_train,pred_prob=y_test_pred_prob)
        else:
            if train_set_control:
                self.train_f1_score_label = metrics.f1_score(y_train, train_set_pred,average='micro')
                self.train_confusion_label = metrics.confusion_matrix(y_train, train_set_pred)
                self.train_roc_auc_score_label = metrics.roc_auc_score(y_train, train_set_pred)

            # test set calculated with label
            self.test_f1_score_label = metrics.f1_score(y_test, y_test_pred_label,average='micro')
            self.test_fb_score_label = fbeta_score(y_test, y_test_pred_label, beta=beta)
            self.test_confusion_label = metrics.confusion_matrix(y_test, y_test_pred_label)
            self.test_roc_auc_score_label = metrics.roc_auc_score(y_test, y_test_pred_label)

            # test set calculated with prob
            # pr part
            self.precision_test, self.recall_test, _ = precision_recall_curve(y_test, y_test_pred_prob)
            self.pr_f1, self.pr_auc = f1_score(y_test, y_test_pred_label,average='micro'), \
                                      auc(self.recall_test, self.precision_test)

            self.pr_no_skill = len(y_train[y_train == 1]) / len(y_train)

            self.r_score = r2(y_test, y_test_pred_label)
            self.brier = brier(y_test, y_test_pred_prob)

            self.ffc_r2=ffc_rsquare(y_test, y_test_pred_prob, y_train)
            self.efron_rsquare=efron_rsquare(y_test, y_test_pred_prob)
            # roc
            self.auc_score = roc_auc_score(y_test, y_test_pred_prob)
            self.imv = imv(true=y_test, train=y_train, pred_prob=y_test_pred_prob)

def print_model_fits(evas):
    print(f'imv={evas.imv},\nroc-auc={evas.auc_score},\npr-auc={evas.pr_auc},\nf1={evas.pr_f1},\nfb={evas.test_fb_score_label},\nefron_r2={evas.efron_rsquare},\nffc_r2={evas.ffc_r2},\nIP={evas.pr_no_skill}')

def sl_eva(superlearner):
    models = list(superlearner.base_models.keys()) + ['sl']
    df_base_pred = superlearner.df_base_pred
    y_train = superlearner.y_train['death']
    y_test = df_base_pred['ori_data']

    df_eva = pd.DataFrame(columns=['model', 'pr_auc', 'roc_auc', 'f1','fb', 'efron', 'ffc', 'ip','imv'])
    for model in models:
        y_test_pred_prob = df_base_pred[model]
        y_test_pred_label = df_base_pred[f'{model}_label']
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_prob)
        pr_auc = auc(recall_test, precision_test)
        roc_auc = roc_auc_score(y_test, y_test_pred_prob)
        f1 = f1_score(y_test, y_test_pred_label,average='micro')
        # f1 = None
        fb = fbeta_score(y_test, y_test_pred_label, beta=beta)
        ffc = ffc_rsquare(true=y_test, train=y_train, pred=y_test_pred_prob)
        efron = efron_rsquare(y_test, y_test_pred_prob)

        imv_ = imv(true=y_test,train=y_train,pred_prob=y_test_pred_prob)
        ip = len(y_train[y_train == 1]) / len(y_train)

        df_eva.loc[len(df_eva)] = [model, pr_auc, roc_auc, f1,fb, efron, ffc, ip, imv_]
    return df_eva



def sl_only_eva(superlearner):

    df_base_pred = superlearner.df_base_pred
    y_train = superlearner.y_train['death']
    y_test = df_base_pred['ori_data']
    y_test_pred_prob = df_base_pred['sl']
    y_test_pred_label = df_base_pred[f'sl_label']

    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_prob)

    dict_eva = {'test_auc_score': roc_auc_score(y_test, y_test_pred_prob),
                'test_f1_score': f1_score(y_test, y_test_pred_label,average='micro'),
                'test_fb_score': fbeta_score(y_test, y_test_pred_label, beta=beta),

                'test_pr_auc': auc(recall_test, precision_test),

                'test_pr_no_skill': len(y_test[y_test == 1]) / len(y_test),
                'test_efron_r2': efron_rsquare(y_test, y_test_pred_prob),

                'test_ffc_r2': ffc_rsquare(true=y_test, train=y_train, pred=y_test_pred_prob),

                'test_briern_r2': brier(true=y_test,pred_prob=y_test_pred_prob),
                'test_imv_r2': imv(true=y_test, train=y_train, pred_prob=y_test_pred_prob)}

    return dict_eva
