# Adopted from https://github.com/JHLiu7/EarlyDRGPrediction


import math
import pandas as pd
import numpy as np
import pickle as pk
import json


from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

with open('paths.json', 'r') as f:
    path = json.load(f)
    drg_34_dissection_path = path["drg_34_dissection_path"]

# read a csv file but only read in the column of principal_diagnosis_lable and CC/MCC
pc_cc_mapping = pd.read_csv(drg_34_dissection_path)[["principal_diagnosis_lable", "CC/MCC"]]
num_labels_pc = pc_cc_mapping.principal_diagnosis_lable.nunique()

# make a dictinoary where key is principal_diagnosis_lable and value is CC/MCC. For one key there can be multiple values

# subsequently realized all [1,0] is due to typo in the original rule and should be fixed!!
pc_cc_dict = {}
for index, row in pc_cc_mapping.iterrows():
    if row["principal_diagnosis_lable"] not in pc_cc_dict:
        pc_cc_dict[row["principal_diagnosis_lable"]] = [row["CC/MCC"]]
    else:
        pc_cc_dict[row["principal_diagnosis_lable"]].append(row["CC/MCC"])



def map_rule(pred_pc, label_pc, pred_cc, label_cc):
    if pred_pc == label_pc:
        if pred_cc == label_cc:
            return True
        # if there's only one cc/MCC code of this principal diagnosis code, then any predcitons would be right
        elif len(pc_cc_dict[label_pc]) == 1:
            return True
        elif pred_cc in pc_cc_dict[label_pc] and pred_cc != label_cc:
            return False
        # for this group, default is to group 0
        elif set(pc_cc_dict[label_pc]) == {2, 1, 0}:
            mark_cc = 0
            if mark_cc == label_cc:
                return True 
            else:
                return False
        # for this group, default is to group 3: without MCC
        elif set(pc_cc_dict[label_pc]) == {2, 3}:
            mark_cc = 3
            if mark_cc == label_cc:
                return True
            else:
                return False
        # for this group, there are two scenario. With MCC wil default to group 1, others will default to group 0
        elif set(pc_cc_dict[label_pc]) == {1, 0}:
            if pred_cc == 2:
                mark_cc = 1
            else:
                mark_cc = 0
            if mark_cc == label_cc:
                return True
            else:
                return False
    return False

map_rule(12,12,1,2)


def accuracies_map(y_pred_pc, labels_pc, y_pred_cc, labels_cc):
    acc = 0.0
    num = len(y_pred_pc)
    
    for i in range(num):
        if map_rule(y_pred_pc[i], labels_pc[i], y_pred_cc[i], labels_cc[i]):
            acc += 1.
    acc /= num

    return acc


# full evaluation 
def full_metrics(y_pred, y, drg_rule, d2i):
    y_pred_w, y_w = map2weight(y_pred, y, drg_rule=drg_rule, d2i=d2i) 

    reg_dict = reg_metrics(y_pred_w, y_w)

    full_dict = {}
    full_dict.update(reg_dict)

    cls_dict = cls_metrics(y_pred, y, len(d2i))
    full_dict.update(cls_dict)

    return full_dict

def cls_metrics(y_pred, y, class_num):
    # class_num = args.Y
    y_pred_ = softmax(y_pred)
    y_ = onehot_encode(y, class_num)

    macroAUC, microAUC, appeared, cases = ave_auc_scores(y_pred_, y_)
    macroF1, microF1 = ave_f1_scores(y_pred, y)

    metric_dict = {
        'microF1':microF1, 'macroF1':macroF1,
        'microAUC':microAUC, 'macroAUC':macroAUC,
        'labels': appeared, 'count': cases
    }
    
    metric_dict['acc10'], metric_dict['acc5'], metric_dict['acc'], _ = accuracies(y_pred, y)
    return metric_dict

def cls_metrics_eval(y_pred, y, class_num):
    # class_num = args.Y
    y_pred_ = softmax(y_pred)
    y_ = onehot_encode(y, class_num)

    macroAUC, microAUC, appeared, cases = ave_auc_scores(y_pred_, y_)
    macroF1, microF1 = ave_f1_scores(y_pred, y)

    metric_dict = {
        'microF1':microF1, 'macroF1':macroF1,
        'microAUC':microAUC, 'macroAUC':macroAUC,
        'labels': appeared, 'count': cases
    }
    
    metric_dict['acc10'], metric_dict['acc5'], metric_dict['acc'], _ = accuracies(y_pred, y)
    metric_dict['y_label'] = y
    metric_dict['y_raw'] = y_pred
    # https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
    metric_dict['y_raw_top5'] = (-y_pred).argsort(axis=1)[:, :5]
    metric_dict['y_pred'] = np.argmax(y_pred_, axis=1)

    return metric_dict

def cls_metrics_multi(y_pred, y):
    # class_num = args.Y
    predictions_pc = y_pred[:, :num_labels_pc]
    y_pred_pc = softmax(predictions_pc)
    labels_onehot_pc = y[:, :num_labels_pc]
    labels_pc = np.argmax(labels_onehot_pc, axis=1)
    predictions_cc = y_pred[:, num_labels_pc:]
    y_pred_cc = softmax(predictions_cc)
    labels_onehot_cc = y[:, num_labels_pc:]
    labels_cc = np.argmax(labels_onehot_cc, axis=1)

    ## Need to double check it mirrows the original methods
    
    macroAUC_pc, microAUC_pc, appeared_pc, cases_pc = ave_auc_scores(y_pred_pc, labels_onehot_pc)
    macroF1_pc, microF1_pc = ave_f1_scores(predictions_pc, labels_pc)
    acc10_pc, acc5_pc, acc_pc, _ = accuracies(predictions_pc, labels_pc)

    macroAUC_cc, microAUC_cc, appeared_cc, cases_cc = ave_auc_scores(y_pred_cc, labels_onehot_cc)
    macroF1_cc, microF1_cc = ave_f1_scores(predictions_cc, labels_cc)
    acc10_cc, acc5_cc, acc_cc, _ = accuracies(predictions_cc, labels_cc)

    y_pred_pc_single = np.argmax(y_pred_pc, axis=1)
    y_pred_cc_single = np.argmax(y_pred_cc, axis=1)

    acc_map = accuracies_map(y_pred_pc_single, labels_pc, y_pred_cc_single, labels_cc)

    metric_dict = {
        'acc_map': acc_map,
        'microF1_pc':microF1_pc, 'macroF1_pc':macroF1_pc,
        'microAUC_pc':microAUC_pc, 'macroAUC_pc':macroAUC_pc,
        'labels_pc': appeared_pc, 'count_pc': cases_pc,
        'acc10_pc': acc10_pc, 'acc5_pc': acc5_pc, 'acc_pc': acc_pc,
        'microF1_cc':microF1_cc, 'macroF1_cc':macroF1_cc,
        'microAUC_cc':microAUC_cc, 'macroAUC_cc':macroAUC_cc,
        'labels_cc': appeared_cc, 'count_cc': cases_cc,
        'acc10_cc': acc10_cc, 'acc5_cc': acc5_cc, 'acc_cc': acc_cc
    }

    return metric_dict

def reg_metrics(y_pred, y):    
    mae = mean_absolute_error(y_pred, y)
    mse = mean_squared_error(y_pred,  y)
    spearman, p = spearmanr(y_pred, y)

    metric_dict = {
        'MAE': mae, 'MSE': mse, 'RMSE': math.sqrt(mse),
        'spearman': spearman, 'corr_p': p
    }

    dist= y_pred - y
    cmi = np.mean(dist)
    overshot, undershot = len(dist[dist>0]), len(dist[dist<0])
    
    metric_dict.update({
        'CMI_error': cmi/np.mean(y), 'CMI_raw':cmi, 'overshot': overshot, 'undershot': undershot
    })
    return metric_dict
    
def cls_metrics_multi_eval(y_pred, y):
    # class_num = args.Y

    predictions_pc = y_pred[:, :num_labels_pc]
    y_pred_pc = softmax(predictions_pc)
    labels_onehot_pc = y[:, :num_labels_pc]
    labels_pc = np.argmax(labels_onehot_pc, axis=1)
    predictions_cc = y_pred[:, num_labels_pc:]
    y_pred_cc = softmax(predictions_cc)
    labels_onehot_cc = y[:, num_labels_pc:]
    labels_cc = np.argmax(labels_onehot_cc, axis=1)

    ## Need to double check it mirrows the original methods
    
    macroAUC_pc, microAUC_pc, appeared_pc, cases_pc = ave_auc_scores(y_pred_pc, labels_onehot_pc)
    macroF1_pc, microF1_pc = ave_f1_scores(predictions_pc, labels_pc)
    acc10_pc, acc5_pc, acc_pc, _ = accuracies(predictions_pc, labels_pc)

    macroAUC_cc, microAUC_cc, appeared_cc, cases_cc = ave_auc_scores(y_pred_cc, labels_onehot_cc)
    macroF1_cc, microF1_cc = ave_f1_scores(predictions_cc, labels_cc)
    acc10_cc, acc5_cc, acc_cc, _ = accuracies(predictions_cc, labels_cc)

    y_pred_pc_single = np.argmax(y_pred_pc, axis=1)
    y_pred_cc_single = np.argmax(y_pred_cc, axis=1)

    acc_map = accuracies_map(y_pred_pc_single, labels_pc, y_pred_cc_single, labels_cc)

    metric_dict = {
        'acc_map': acc_map,
        'microF1_pc':microF1_pc, 'macroF1_pc':macroF1_pc,
        'microAUC_pc':microAUC_pc, 'macroAUC_pc':macroAUC_pc,
        'labels_pc': appeared_pc, 'count_pc': cases_pc,
        'acc10_pc': acc10_pc, 'acc5_pc': acc5_pc, 'acc_pc': acc_pc,
        'microF1_cc':microF1_cc, 'macroF1_cc':macroF1_cc,
        'microAUC_cc':microAUC_cc, 'macroAUC_cc':macroAUC_cc,
        'labels_cc': appeared_cc, 'count_cc': cases_cc,
        'acc10_cc': acc10_cc, 'acc5_cc': acc5_cc, 'acc_cc': acc_cc
    }

    metric_dict['y_label'] = y
    metric_dict['y_raw'] = y_pred

    return metric_dict


# to print out results
def result2str(d):
    try:
        mif, maf = d['microF1'], d['macroF1']
        mia, maa = d['microAUC'], d['macroAUC']
        a10, a5, a = d['acc10'], d['acc5'], d['acc']
        la, ct = d['labels'], d['count']
    except:
        pass
    ma, rm = d['MAE'], d['RMSE']
    sp, p = d['spearman'], d['corr_p']
    cm,ov,ud = d['CMI_error'], d['overshot'], d['undershot']

    title = "****" * 5 + '\n'
    try:
        s1 = "{} cases, {} labels".format(ct, la)
        s2 = "MACRO-AUC     \tMICRO-AUC      \tMACRO-F1     \tMICRO-F1  "
        s3 = "{:.4f}  \t{:.4f}  \t{:.4f}  \t{:.4f}".format(maa, mia, maf, mif)
        s4 = "Acc10  \tAcc5  \tAcc "
        s5 = "{:.4f}  \t{:.4f}  \t{:.4f}  \n".format(a10, a5, a)
        title = title+'\n'.join([s1, s2, s3, s4, s5])
    except:
        pass
    r1 = "MAE: {:.4f}  RMSE: {:.4f}  Corr: {:.4f}  \n".format(ma, rm, sp)
    r2 = "CMI_error: {:.2%}  overshot: {}  undershot: {}  \n\n".format(cm,ov,ud)

    title = title+'\n'.join([r1, r2])

    return title


# running evaluation
def score_f1(y_pred, y):
    """
        y_pred: logit
    """
    y_flat = np.argmax(y_pred, axis=1)
    return f1_score(y, y_flat, average='micro')

def score_mae(y_pred, y):
    return mean_absolute_error(y_pred, y)


# utils
def map2weight(y_pred, y, drg_rule, d2i):

    idx2drg = {v:k for k,v in d2i.items()}
    drg2weight = {}
    for _, row in drg_rule.iterrows():
        drg2weight[row['DRG_CODE']] = row['WEIGHT']

    y_pred = [drg2weight[idx2drg[d]] for d in np.argmax(y_pred, axis=1)]
    y = [drg2weight[idx2drg[d]] for d in y]
    return np.array(y_pred), np.array(y)

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.expand_dims(e_x.sum(axis=1), 1)

def onehot_encode(y, class_num):
    """
        y: a flat array of labels
    """
    yone = []
    for i in y:
        onehot = np.zeros(class_num)
        onehot[i] = 1
        yone.append(onehot)
    return np.array(yone)

def accuracies(y_pred, y, onlyAcc=False):
    """
    y_pred: logits
    y: a list of labels
    """
    acc10 = 0.0
    acc5 = 0.0
    acc1 = 0.0
    num = len(y)
    
    for i in range(num):

        pred = y_pred[i]
        top10_pred = set(pred.argsort()[-10:])
        top5_pred = set(pred.argsort()[-5:])
        top1_pred = set(pred.argsort()[-1:])

        label = y[i]
        # label = np.argmax(y[i])

        if label in top10_pred:
            acc10 += 1.
        if label in top5_pred:
            acc5 += 1.
        if label in top1_pred:
            acc1 += 1.

    acc10 /= num
    acc5 /= num
    acc1 /= num

    if onlyAcc:
        return acc1
    return acc10, acc5, acc1, num

def ave_auc_scores(y_pred, y):
    # micro/macro auc based on classes
    """
        y.shape: [sample, classes] float
        y_pred.shape: [sample, classes] int
        numpy
    """

    aucroc_cases = {}
    for i in range(y.shape[1]):
        if y[:, i].sum()>0: # class appears in test set
            fp, tp, _ = roc_curve(y[:, i], y_pred[:, i])
            if len(fp) >1 and len(tp) >1:
                auc_roc = auc(fp, tp)
                aucroc_cases[i] = auc_roc

    fp_mic, tp_mic, _ = roc_curve(y.ravel(), y_pred.ravel())

    # appearing classes
    labels = list(aucroc_cases.keys())

    # roc
    auc_roc_macro = np.mean(list(aucroc_cases.values()))
    auc_roc_micro = auc(fp_mic, tp_mic)
    return auc_roc_macro, auc_roc_micro, len(labels), len(y)

def ave_f1_scores(y_pred, y):
    # f1 
    # require y_pred, y being flat list
    y_flat = np.argmax(y_pred, axis=1)

    f1_macro = f1_score(y, y_flat, average='macro', labels=np.unique(y))
    f1_micro = f1_score(y, y_flat, average='micro', labels=np.unique(y))

    return f1_macro, f1_micro
