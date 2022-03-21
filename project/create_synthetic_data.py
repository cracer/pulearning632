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




def find_size(df, thres, col_name_str):
    '''
    Calculate the size for synthetic data (binary classification)
    :param df: DataFrame containing the variable for which we want to create syn data
    :param thres: float or int value separating the two classes
    :param col_name_str: the column name for which we want to create syn data
    :return int or float value of the synthetic data size
    '''
    initial_class1_count = df[df[col_name_str] >= thres].count()
    class1_ct = initial_class1_count[col_name_str]
    class2_ct = len(df) - class1_ct
    syn_size = class2_ct - class1_ct # default is class2 size > class1 size
    return syn_size  

def oversample(low_bound, upper_bound, syn_size):
    '''
    Create synthetic random data for the minority class (only for binary classification)
    
    :param low_bound: float or int value for the lower bound
    :param upper_bound: float or int value for the higher bound
    :return a list of the new synthetic values
    '''
    
    syn_data = np.random.uniform(low=low_bound, high=upper_bound, size=abs(syn_size))

    return syn_data 

def combine_syn_original(ori_list, syn_list):
    '''
    Concatenate the two lists, maintain time order
    :param ori_list: list containing original values
    :param syn_list: list containing synthetic values
    :return a new list from combining the two lists
    '''

    # Default is len(syn_list) < len(ori_list)
    longer_list = ori_list 
    shorter_list = syn_list 

    # If len(syn_list) > len(ori_list)
    if len(syn_list) > len(ori_list):
        longer_list = syn_list 
        shorter_list = ori_list 
    
    # if len(syn_list) == len(ori_list)
    if len(syn_list) == len(ori_list):
        # pick either ori or syn for longer and shorter
        longer_list = ori_list
        shorter_list = ori_list

    rem_longer_list = len(longer_list)
    rem_shorter_list = len(shorter_list)

    new_data = []
    count = 0

    # Alternately add each value of the lists to maintain time order
    while rem_shorter_list != 0:
        new_data.append(longer_list[count])
        new_data.append(shorter_list[count])
        rem_shorter_list -= 1
        count += 1

    if (count != rem_longer_list):
        new_data.extend(longer_list[len(shorter_list):])

    return new_data 
