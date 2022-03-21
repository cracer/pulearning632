import os, sys, glob
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from src.utils import get_proj_root
import pandas as pd
import numpy as np
import detect_modality, find_thresholds, create_synthetic_data
import data_loader_latency
from sklearn.model_selection import train_test_split
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from scipy import sparse
import tsfresh
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from pulearn import (
    ElkanotoPuClassifier,
    # WeightedElkanotoPuClassifier,
)

import explore_anonymize_caida
import warnings

warnings.filterwarnings("ignore")
random_seed = 222

def concatenate_train_val(X_train, X_val):
    X_train_new = np.concatenate([X_train, X_val], axis=0)
    return X_train_new


def init_unlabeled_neg(set_length):
    y_train = np.full((set_length,), -1., dtype=int)
    return y_train

def create_new_y_train(y_train, y_val):
    y_train_df = pd.Series(y_train)
    y_train_new = y_train_df.append(y_val)
    return y_train_new

def calculate_pu_scores(y_train_new, X_train_new, X_test, y_test):
    pu_f1_scores = []
    reg_f1_scores = []
    prec_scores = []
    rec_scores = []
    n_sacrifice_iter = range(0, len(np.where(y_train_new == +1.)[0]))
    for n_sacrifice in n_sacrifice_iter:
        print("PU transformation in progress...")
        # print("Making {} good data examples noise.".format(n_sacrifice))
        y_train_pu = np.copy(y_train_new)
        pos = np.where(y_train_new == -1.)[0]
        np.random.shuffle(pos)
        sacrifice = pos[:n_sacrifice+1]
        y_train_pu[sacrifice] = +1.
        pos = len(np.where(y_train_pu == +1.)[0])
        unlabelled = len(np.where(y_train_pu == -1.)[0])
        
        # print("PU transformation applied. We now have:")
        # print("{} are noise.".format(len(np.where(y_train_pu == -1.)[0])))
        # print("{} are good data.".format(len(np.where(y_train_pu == +1.)[0])))
        print("-------------------")
        print((
            "Fitting PU classifier (using a random forest as an inner "
            "classifier)..."
        ))
        estimator = RandomForestClassifier(
            n_estimators=100,
            criterion='gini',
            bootstrap=True,
            n_jobs=1,
        )
        # pu_estimator = WeightedElkanotoPuClassifier(
        #    estimator, pos, unlabelled)
        pu_estimator = ElkanotoPuClassifier(estimator)
    #     print(pu_estimator)
    #     print("train_prim_pu shape: ", train_prim_pu.shape)
    #     print("y_train_pu shape: ", y_train_pu.shape)
        try:
            pu_estimator.fit(X_train_new, y_train_pu)
            y_pred = pu_estimator.predict(X_test)
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                y_test, y_pred)
            pu_f1_scores.append(f1_score[1])
            prec_scores.append(precision[1])
            rec_scores.append(recall[1])
            print("F1 score: {}".format(f1_score[1]))
            print("Precision: {}".format(precision[1]))
            print("Recall: {}".format(recall[1]))
    #         print("Regular learning (w/ a random forest) in progress...")
    #         estimator = RandomForestClassifier(
    #             n_estimators=100,
    #             bootstrap=True,
    #             n_jobs=1,
    #         )
    #         estimator.fit(train_prim_pu, y_train_pu)
    #         y_pred = estimator.predict(test_primitive_matrix)
    #         precision, recall, f1_score, _ = precision_recall_fscore_support(
    #             test_ground, y_pred)
    #         reg_f1_scores.append(f1_score[1])
    #         print("F1 score: {}".format(f1_score[1]))
    #         print("Precision: {}".format(precision[1]))
    #         print("Recall: {}".format(recall[1]))
        except ValueError:
            break

    return pu_f1_scores[-1], prec_scores[-1], rec_scores[-1]
            
