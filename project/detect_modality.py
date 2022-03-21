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

def find_peaks(df, col_name_str, min_peak_fraction):
    """
    Find peak points and indices via KDE
    
    :param df: DataFrame containing the desired column

    :param col_name_str: The desired column name in string
    
    :return a list containing the peak indices, the density value of the peaks, KDE x-values, KDE y-values
    """
    from scipy.signal import find_peaks, peak_prominences
    plt.figure()
    
    # get the x-values and y-values of the kde plot
    density_x, density_y = sns.kdeplot(df[col_name_str]).get_lines()[0].get_data()
    # want to find peaks that are greater than min_peak_fraction of max peak 
    # to exclude peaks that are too small
    min_peak = min_peak_fraction * max(density_y)
    # find peaks with density more than min_peak
    peaks, _ = find_peaks(x=density_y, prominence=min_peak)
    prominences = peak_prominences(density_y, peaks)[0]
    x_peaks = [density_x[idx] for idx in peaks]
    return peaks, x_peaks, density_x, density_y


def find_peaks_no_header(df, col_num, min_peak_fraction):
    """
    Find peak points and indices via KDE
    
    :param df: DataFrame containing the desired column

    :param col_name_str: The desired column name in string
    
    :return a list containing the peak indices, the density value of the peaks, KDE x-values, KDE y-values
    """
    from scipy.signal import find_peaks, peak_prominences
    plt.figure()
    # get the x-values and y-values of the kde plot
    density_x, density_y = sns.kdeplot(df.iloc[col_num]).get_lines()[0].get_data()    
    # want to find peaks that are greater than min_peak_fraction of max peak 
    # to exclude peaks that are too small
    min_peak = min_peak_fraction * max(density_y)
    # find peaks with density more than min_peak
    peaks, _ = find_peaks(x=density_y, prominence=min_peak)
    prominences = peak_prominences(density_y, peaks)[0]
    x_peaks = [density_x[idx] for idx in peaks]
    return peaks, x_peaks, density_x, density_y



def find_lowest_points_between_peaks_no_header(df, col_num, min_peak_fraction=0.1):
    """
    Search for the lowest points between peaks, store the points in a list
    
    :param: df: DataFrame containing the desired column

    :param col_name_str: The desired column name in string
    
    :param min_peak_fraction: float between 0.0 and 1.0 to specify peak size to be considered a peak

    :return a list containing the lowest points
    """
    # find the peaks
    peaks, x_peaks, density_x, density_y = find_peaks_no_header(df, col_num, min_peak_fraction)
    valley_indices = []
    valley_points = []
    for i in range(len(peaks)-1):
        peak = peaks[i]
        next_peak = peaks[i+1]
        valley_idx = peak + np.argmin(density_y[peak:next_peak+1])
        valley_indices.append(valley_idx)
        valley_points.append(density_x[valley_idx])
        
    # plot the density
    plt.plot(density_x, density_y)
    # plot the peaks
    plt.plot(x_peaks, density_y[peaks], "x")
    # plot the lowest point between the peaks
    plt.plot(valley_points, density_y[valley_indices], "o", c="red")
    # plot a vertical line for the peaks
    plt.vlines(x=x_peaks, ymin=0, ymax=density_y[peaks])
    # plot a vertical line for the lowest point
    plt.vlines(x=valley_points, ymin=0, ymax=density_y[valley_indices])
    plt.title("Density Plot with Peaks and Lowest Point Between Peaks")
    plt.ylabel("Density")
    plt.xlabel("RTT (ms)")
    
    return valley_points



def find_lowest_points_between_peaks(df, col_name_str, min_peak_fraction=0.1):
    """
    Search for the lowest points between peaks, store the points in a list
    
    :param: df: DataFrame containing the desired column

    :param col_name_str: The desired column name in string
    
    :param min_peak_fraction: float between 0.0 and 1.0 to specify peak size to be considered a peak

    :return a list containing the lowest points
    """
    # find the peaks
    peaks, x_peaks, density_x, density_y = find_peaks(df, col_name_str, min_peak_fraction)
    valley_indices = []
    valley_points = []
    for i in range(len(peaks)-1):
        peak = peaks[i]
        next_peak = peaks[i+1]
        valley_idx = peak + np.argmin(density_y[peak:next_peak+1])
        valley_indices.append(valley_idx)
        valley_points.append(density_x[valley_idx])
        
    # plot the density
    plt.plot(density_x, density_y)
    # plot the peaks
    plt.plot(x_peaks, density_y[peaks], "x")
    # plot the lowest point between the peaks
    plt.plot(valley_points, density_y[valley_indices], "o", c="red")
    # plot a vertical line for the peaks
    plt.vlines(x=x_peaks, ymin=0, ymax=density_y[peaks])
    # plot a vertical line for the lowest point
    plt.vlines(x=valley_points, ymin=0, ymax=density_y[valley_indices])
    plt.title("Density Plot with Peaks and Lowest Point Between Peaks")
    plt.ylabel("Density")
    plt.xlabel("RTT (ms)")
    
    return valley_points    

def split_2_peaks(df, col_name_str, valley_points):
    '''
    In case of bimodal data, split data into 2 subsets with the valley point as the separator

    :param: df: DataFrame containing the desired column

    :param col_name_str: The desired column name in string

    :param valley_points: A list containing the lowest point between peaks on the x-axis

    :return List of two DataFrames separated by the valley_points
    '''
    df_1 = df[df[col_name_str] < valley_points[0]]
    df_2 = df[df[col_name_str] >= valley_points[0]]
    return [df_1, df_2]




def split_2_peaks_no_header(df, col_num, valley_points):
    '''
    In case of bimodal data, split data into 2 subsets with the valley point as the separator

    :param: df: DataFrame containing the desired column

    :param col_name_str: The desired column name in string

    :param valley_points: A list containing the lowest point between peaks on the x-axis

    :return List of two DataFrames separated by the valley_points
    '''
    df_1 = df[df.iloc[col_num] < valley_points[0]]
    df_2 = df[df.iloc[col_num] >= valley_points[0]]
    return [df_1, df_2] 


def divide_data_based_on_peaks(df, col_name_str, valley_points):
    """
    Split data at the lowest points to create unimodal subsets
    
    :param: df: DataFrame containing the desired column

    :param col_name_str: The desired column name in string

    :param valley_points: A list containing the lowest point on the x-axis
    
    :return DataFrame of the subsets
    """
    subset_list = []
    num_valleys = len(valley_points)
    
    num_peaks = num_valleys + 1

    # if data has no valley, thus it is unimodal
    if num_valleys == 0:
        subset_list.append(df)
    # if data has 1 valley and 2 peaks (bimodal)
    elif num_valleys == 1:
        two_subsets_list = split_2_peaks(df, col_name_str=col_name_str, valley_points=valley_points)
        subset_list.extend(two_subsets_list)
    # if there are at least 2 valleys (at least 3 peaks)
    else:
        # the first subset is always the data less than the first valley point
        df_first = df[df[col_name_str] < valley_points[0]]
        subset_list.append(df_first)
       
        i = 0
        # remember num_valleys is at least 2
        while (i < num_valleys-1):
            df_subset = df[(df[col_name_str] >= valley_points[i]) & (df[col_name_str] < valley_points[i+1])]
            subset_list.append(df_subset)
            i += 1

        # the last subset is always the data greater than the last valley point
        df_last = df[df[col_name_str] >= valley_points[-1]]     
        subset_list.append(df_last)       

    return subset_list



def divide_data_based_on_peaks_no_header(df, col_num, valley_points):
    """
    Split data at the lowest points to create unimodal subsets
    
    :param: df: DataFrame containing the desired column

    :param col_name_str: The desired column name in string

    :param valley_points: A list containing the lowest point on the x-axis
    
    :return DataFrame of the subsets
    """
    subset_list = []
    num_valleys = len(valley_points)
    
    num_peaks = num_valleys + 1

    # if data has no valley, thus it is unimodal
    if num_valleys == 0:
        subset_list.append(df)
    # if data has 1 valley and 2 peaks (bimodal)
    elif num_valleys == 1:
        two_subsets_list = split_2_peaks_no_header(df, col_num=col_num, valley_points=valley_points)
        subset_list.extend(two_subsets_list)
    # if there are at least 2 valleys (at least 3 peaks)
    else:
        # the first subset is always the data less than the first valley point
        df_first = df[df.iloc[col_num] < valley_points[0]]
        subset_list.append(df_first)
       
        i = 0
        # remember num_valleys is at least 2
        while (i < num_valleys-1):
            df_subset = df[(df.iloc[col_num] >= valley_points[i]) & (df.iloc[col_num] < valley_points[i+1])]
            subset_list.append(df_subset)
            i += 1

        # the last subset is always the data greater than the last valley point
        df_last = df[df.iloc[col_num] >= valley_points[-1]]     
        subset_list.append(df_last)       

    return subset_list



