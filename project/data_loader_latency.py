"""
Read latency data
Calculate features
Split into test (100), val (100), train (the rest)
Label test and val sets: 
    Weak supervision labels positive and negative classes
    PU learning labels a subset of positive class

INPUT: 
preprocessed data

OUTPUT:
train_primitive_matrix (primitive = features)
val_primitive_matrix
test_primitive_matrix

train_original (pandas DataFrame)
val_original
test_original

val_ground
test_ground
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import *
import glob, sys, os
from pathlib import Path
from src.utils import get_proj_root
import scipy
import json
from sklearn.model_selection import train_test_split
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")




def parse_file(filename):
    """
    Parse file into pandas DataFrame
    :parameter filename: a path to the file
    :return pandas DataFrame of the data
    """
    dataset = pd.read_csv(filename, names=['rtt', 'id', 'time'])
    data = dataset[['time', 'rtt', 'id']]

    return data


def split_data(X, data):
    """
    Split data into test, validation, and train sets
    :parameter X: extracted features of entire data in numpy array
    :parameter data: pandas DataFrame of entire data
    :return: 
        extracted features of train set
        extracted features of val set
        extracted features of test set
        pandas DataFrame of train set
        pandas DataFrame of val set
        pandas DataFrame of test set
    """
    np.random.seed(1234)
    num_sample = np.shape(X)[0]

    # Take 10% for test
    num_test = int(0.1 * num_sample)

    X_test = X[0:num_test, :]
    X_train = X[num_test:, :]

    data_test = data[0:num_test]
    data_train = data[num_test:]

    # split dev/test

    # Take 25% for val
    num_validation = int(0.25 * num_sample)

    # Note that the first 10% of the data are already in the test set
    X_val = X_train[0:num_validation, :]
    X_train = X_train[num_validation:, :]

    data_val = data_train[0:num_validation]
    data_train = data_train[num_validation:]

    return np.array(X_train), np.array(X_val), np.array(X_test), \
        data_train, data_val, data_test


def split_data_1010(X, data):
    """
    Split data into test, validation, and train sets
    :parameter X: extracted features of entire data in numpy array
    :parameter data: pandas DataFrame of entire data
    :return: 
        extracted features of train set
        extracted features of val set
        extracted features of test set
        pandas DataFrame of train set
        pandas DataFrame of val set
        pandas DataFrame of test set
    """
    np.random.seed(1234)
    num_sample = np.shape(X)[0]

    # Take 10% for test
    num_test = int(0.1 * num_sample)

    X_test = X[0:num_test, :]
    X_train = X[num_test:, :]

    data_test = data[0:num_test]
    data_train = data[num_test:]

    # split dev/test

    # Take 10% for val
    num_validation = int(0.1 * num_sample)

    # Note that the first 10% of the data are already in the test set
    X_val = X_train[0:num_validation, :]
    X_train = X_train[num_validation:, :]

    data_val = data_train[0:num_validation]
    data_train = data_train[num_validation:]

    return np.array(X_train), np.array(X_val), np.array(X_test), \
        data_train, data_val, data_test


def split_data_2020(X, data):
    """
    Split data into test, validation, and train sets
    :parameter X: extracted features of entire data in numpy array
    :parameter data: pandas DataFrame of entire data
    :return: 
        extracted features of train set
        extracted features of val set
        extracted features of test set
        pandas DataFrame of train set
        pandas DataFrame of val set
        pandas DataFrame of test set
    """
    np.random.seed(1234)
    num_sample = np.shape(X)[0]

    # Take 20% for test
    num_test = int(0.2 * num_sample)

    X_test = X[0:num_test, :]
    X_train = X[num_test:, :]

    data_test = data[0:num_test]
    data_train = data[num_test:]

    # split dev/test

    # Take 20% for val
    num_validation = int(0.2 * num_sample)

    # Note that the first 20% of the data are already in the test set
    X_val = X_train[0:num_validation, :]
    X_train = X_train[num_validation:, :]

    data_val = data_train[0:num_validation]
    data_train = data_train[num_validation:]

    return np.array(X_train), np.array(X_val), np.array(X_test), \
        data_train, data_val, data_test



def split_data_2520(X, data):
    """
    Split data into test, validation, and train sets
    :parameter X: extracted features of entire data in numpy array
    :parameter data: pandas DataFrame of entire data
    :return: 
        extracted features of train set
        extracted features of val set
        extracted features of test set
        pandas DataFrame of train set
        pandas DataFrame of val set
        pandas DataFrame of test set
    """
    np.random.seed(1234)
    num_sample = np.shape(X)[0]

    # Take 25% for test
    num_test = int(0.25 * num_sample)

    X_test = X[0:num_test, :]
    X_train = X[num_test:, :]

    data_test = data[0:num_test]
    data_train = data[num_test:]

    # split dev/test

    # Take 20% for val
    num_validation = int(0.2 * num_sample)

    # Note that the first 20% of the data are already in the test set
    X_val = X_train[0:num_validation, :]
    X_train = X_train[num_validation:, :]

    data_val = data_train[0:num_validation]
    data_train = data_train[num_validation:]

    return np.array(X_train), np.array(X_val), np.array(X_test), \
        data_train, data_val, data_test


def random_split_data_2520(X, data):
    """
    Split data into test, validation, and train sets
    :parameter X: extracted features of entire data in numpy array
    :parameter data: pandas DataFrame of entire data
    :return: 
        extracted features of train set
        extracted features of val set
        extracted features of test set
        pandas DataFrame of train set
        pandas DataFrame of val set
        pandas DataFrame of test set
    """
    np.random.seed(1234)
    # Split into train and test
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=222, stratify=y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=222)

    # Split X_train into X_train and X_dev
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=test_size, random_state=222, stratify=y_train)
    # X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=test_size, random_state=222)


    return np.array(X_train), np.array(X_val), np.array(X_test), \
        data_train, data_val, data_test




def label(x, thres):
    if x >= thres:
        y = -1
    else:
        y = 1
    return y


def extract_8_statistical_features(data):
    """
    Extract 8 statistical features: mean, standard deviation, length, 
                                    minimum val, first quartile, second quartile,
                                    third quartile, maximum val
    :parameter data: pandas DataFrame of the data whose features to be extracted
    :return X: extracted features in numpy matrix
    """
    extraction_settings = MinimalFCParameters()
    data_extracted_feats = extract_features(data,
                                            column_id='id', column_sort='time',
                                            default_fc_parameters=extraction_settings,
                                            impute_function=impute)

    # X_list = data_extracted_feats.values.tolist()
    # X = np.array(X_list)
    # return X

    # X_list = data_extracted_feats.values.tolist()
    X = np.array(X_list)
    return X




class DataLoader(object):
    """ 
    A class to load, calculate features, split data
    """

    def load_data(self, dataset, data_path, client, vantage_pt):
        """
        Load, calculate features, split data
        :parameter dataset: dataset filename
        :parameter data_path: path to dataset
        :parameter client: client name
        :parameter vantage_pt: vantage_pt of the client

        :return 8 sets: (1) train extracted features (numpy)
                        (2) val extracted features (numpy)
                        (3) test extracted features (numpy)
                        (4) val labels (numpy)
                        (5) test labels (numpy)
                        (6) train DataFrame
                        (7) val DataFrame
                        (8) test DataFrame
        """
        
        
        # Parse Files
        data = parse_file(data_path+dataset)
        # Convert time into str to prepare for feature extraction using tsfresh
        data['time'] = data['time'].apply(lambda x: str(x))

        # Extract features
        X = extract_8_statistical_features(data)

        # Split Dataset into Train, Val, Test
        
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_original, val_original, test_original = split_data(X, data)

        os.chdir(data_path)
        df_threshold = pd.read_csv(
            'Threshold-' + vantage_pt + '-' + str(client[:-4]) + '.csv')
        thres = df_threshold['Threshold'][0]

        # label test data
        test_ground = test_original['rtt'].apply(label, thres=thres)

        # label val data
        val_ground = val_original['rtt'].apply(label, thres=thres)

        return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            np.array(val_ground), np.array(test_ground), \
            train_original, val_original, test_original


    def load_rtt_with_thres(self, dataset, data_path, threshold_csv_path):
        """
        Load, calculate features, split data
        :parameter dataset: dataset filename (str)
        :parameter data_path: path to dataset (str)
        :parameter client: client name (str)
        :parameter threshold_csv_path: path to csv file containing threshold to separate cls (str)

        :return 8 sets: (1) train extracted features (numpy)
                        (2) val extracted features (numpy)
                        (3) test extracted features (numpy)
                        (4) val labels (numpy)
                        (5) test labels (numpy)
                        (6) train DataFrame
                        (7) val DataFrame
                        (8) test DataFrame
        """
        
        
        # Parse Files
        data = parse_file(data_path+dataset)
        # Convert time into str to prepare for feature extraction using tsfresh
        data['time'] = data['time'].apply(lambda x: str(x))

        # Extract features
        X = extract_8_statistical_features(data)

        # Split Dataset into Train, Val, Test
        
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_original, val_original, test_original = split_data_2520(X, data)

        os.chdir(data_path)
        df_threshold = pd.read_csv(threshold_csv_path)
        thres = df_threshold['Threshold'][0]

        # label test data
        test_ground = test_original['rtt'].apply(label, thres=thres)

        # label val data
        val_ground = val_original['rtt'].apply(label, thres=thres)

        return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            np.array(val_ground), np.array(test_ground), \
            train_original, val_original, test_original



def load_rtt_with_thres_parsed(self, dataset, data_path, threshold_csv_path):
        """
        Load, calculate features, split data
        :parameter dataset: dataset filename (str)
        :parameter data_path: path to dataset (str)
        :parameter client: client name (str)
        :parameter threshold_csv_path: path to csv file containing threshold to separate cls (str)

        :return 8 sets: (1) train extracted features (numpy)
                        (2) val extracted features (numpy)
                        (3) test extracted features (numpy)
                        (4) val labels (numpy)
                        (5) test labels (numpy)
                        (6) train DataFrame
                        (7) val DataFrame
                        (8) test DataFrame
        """
        
        
        # Parse Files
        # data = parse_file(data_path+dataset)
        
        # Convert time into str to prepare for feature extraction using tsfresh
        data['time'] = data['time'].apply(lambda x: str(x))

        # Extract features
        X = extract_8_statistical_features(data)

        # Split Dataset into Train, Val, Test
        
        train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            train_original, val_original, test_original = split_data_2520(X, data)

        os.chdir(data_path)
        df_threshold = pd.read_csv(threshold_csv_path)
        thres = df_threshold['Threshold'][0]

        # label test data
        test_ground = test_original['rtt'].apply(label, thres=thres)

        # label val data
        val_ground = val_original['rtt'].apply(label, thres=thres)

        return train_primitive_matrix, val_primitive_matrix, test_primitive_matrix, \
            np.array(val_ground), np.array(test_ground), \
            train_original, val_original, test_original


