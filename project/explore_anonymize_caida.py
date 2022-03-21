"""
Explore datasets and anonymize datasets
Input: original datasets
Output: anonymized datasets

"""

import os, sys, glob
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from src.utils import get_proj_root
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
random_seed = 222




# anonymize datasets
print("    Anonymize datasets ...     ")

def get_prefix_number(num):
    """
    Getting the prefix for the new name 
    e.g, '0' for 12 so it becomes '012'
    """
    prefix = '0'
    if num < 10:
        prefix = '00'
    elif num >= 100:
        prefix = ''
    else:
        prefix = '0'
    
    return prefix











