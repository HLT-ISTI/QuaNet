import os, sys
import signal
import math, random
import shutil
from sklearn.metrics import *

#--------------------------------------------------------------
# Run helpers
#--------------------------------------------------------------
def err_exception(exit_condition=True, err_msg=""):
    if exit_condition:
        if not err_msg:
            raise ValueError(sys.argv[0]+ " Error")
        else:
            raise ValueError(err_msg)

def err_param_range(param_name, param_value, valid_values):
    err_exception(exit_condition=param_value not in valid_values,
                  err_msg='Param error: %s=%s should be one in %s' % (param_name, str(param_value), str(valid_values)))


def notexist_exit(path):
    if isinstance(path, list):
        [notexist_exit(p) for p in path]
    elif not os.path.exists(path):
        print("Error. Path <%s> does not exist or is not accessible." %path)
        sys.exit()

def create_if_not_exists(dir):
    if not os.path.exists(dir): os.makedirs(dir)
    return dir


def tee(outstring, fout):
    print(outstring)
    fout.write(outstring + '\n')


def split_at(list_, prop):
    split_point = int(prop * len(list_))
    return list_[:split_point], list_[split_point:]

def shuffle_tied(l1, l2, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    l1_l2_tied = zip(l1, l2)
    random.shuffle(l1_l2_tied)
    l1_, l2_ = zip(*l1_l2_tied)
    return list(l1_), list(l2_)

"""
def evaluation_metrics(predictions, true_labels):
    no_test_examples = (sum(true_labels) == 0)
    no_predictions = (sum(predictions) == 0)
    acc = accuracy_score(true_labels, predictions)
    if no_test_examples and no_predictions:
        f1 = 1.0
    else:
        f1 = f1_score(true_labels, predictions, average='binary', pos_label=1)
    if no_predictions:
        p = 1.0
    else:
        p = precision_score(true_labels, predictions, average='binary', pos_label=1)
    if no_test_examples:
        r=1.0
    else:
        r = recall_score(true_labels, predictions, average='binary', pos_label=1)
    return acc, f1, p, r
"""







