import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import *
import glob, sys, os
from pathlib import Path
from src.utils import get_proj_root

import warnings
warnings.filterwarnings("ignore")


def label(x, thres):
    '''
    Assign positive/negative labels
    :param x: numerical value
    :param thres: numerical value to compare
    :return y as either -1 or 1
    '''
    if x >= thres:
        y = -1
    else:
        y = 1
    return y

def find_iqr(df, col_name_str):
    '''
    Calculate InterQuartile Range
    :param df: DataFrame
    :param col_name_str: Column name for which we want IQR (column must be int or float)
    :return iqr value (float)
    '''

    # find 25%, med, 75%
    q1, med, q3 = df[col_name_str].quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    return iqr  


def find_mean_plus_sd_thres(df, col_name_str):
    '''
    Calculate mean+1sd threshold
    :param df: DataFrame
    :param col_name_str: Column name for which we want IQR (column must be int or float)
    :return mean1sd_value (float)
    '''

    # Rename 'Unnamed: 0' to 'index'
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    # Calculate mean
    Mean = round(np.mean(df[col_name_str]), 5)

    # Calculate standard deviation
    SD = round(np.std(df[col_name_str]), 5)

    # Find mean+1sd threshold
    mean1sd_good = []
    mean1sd_bad = []

    for i in df['index']:
        if df[col_name_str][i] >= Mean + SD:
            mean1sd_bad.append(df[col_name_str][i])
        else:
            mean1sd_good.append(df[col_name_str][i])
    
    mean1sd_thres = round(np.mean(mean1sd_good), 4)
    return mean1sd_thres

def find_mean_plus_sd_thres_no_avg(df, col_name_str):
    '''
    Calculate mean+1sd threshold without using average of the classes
    :param df: DataFrame
    :param col_name_str: Column name for which we want IQR (column must be int or float)
    :return mean1sd_value (float)
    '''

    # Rename 'Unnamed: 0' to 'index'
    df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)

    # Calculate mean
    Mean = round(np.mean(df[col_name_str]), 5)

    # Calculate standard deviation
    SD = round(np.std(df[col_name_str]), 5)

    # Find mean+1sd threshold
    mean1sd_thres = round((Mean + SD), 4)
    return mean1sd_thres



def record_threshold(df_name_str, dataset_name_str, client_list, thres_list):
    '''
    Write threshold values on csv
    Format: | Client | Threshold |
    :param df_name_str: string of df name
    :param dataset_name_str: string of dataset name
    :param client_list: list of clients
    :param thres_list: list of threshold values
    '''
    # make sure dataset_name does not contain '.csv'
    try:
        assert dataset_name_str[-4:] != '.csv', 'Please remove .csv from dataset name'
    except AssertionError as msg:
        print(msg)

    # create new DataFrame
    df_thresholds = pd.DataFrame({'Client': client_list, 'Threshold': thres_list})
    # Write new DataFrame to a csv file
    
    print("threshold name: ", 'Threshold-' + df_name_str + '-' + dataset_name_str + '.csv')
    '''
    UNCOMMENT LATER
    '''
    df_thresholds.to_csv('Threshold-' + df_name_str + '-' + dataset_name_str + '.csv', encoding='utf-8')





