import os, sys, glob
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from src.utils import get_proj_root
import pandas as pd
import numpy as np
import detect_modality, find_thresholds, create_synthetic_data
import explore_anonymize_caida
import warnings

warnings.filterwarnings("ignore")
random_seed = 222

proj_root = get_proj_root()
path_proj = proj_root 
path_proj_pu = proj_root / 'project'
# path_caida = proj_root / 'pu_caida' / 'caida_datasets'
input_dir = path_proj / 'caida_processed'
# thres_dir = path_proj_pu / 'caida_processed'
current_dir = path_proj 

#Creating a folder if it does not exist
output_dir = path_proj_pu / 'caida_processed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)




if __name__ == '__main__':


    # go to the original caida datasets
    os.chdir(input_dir)
    # collect the caida_data in the csv format
    caida_data = glob.glob('*.csv')
    print("caida data: ", caida_data)
    # sort the dataset name to make it easier to track
    caida_sorted = sorted(caida_data)
    print("caida sorted: ", caida_sorted)
    # example use only the first dataset
    datasets = caida_sorted[0]
    print("datasets: ", datasets)
    anon_caida_name = []

    # anonymize dataset
    datact = 0
    
    for cd in caida_sorted:

        os.chdir(input_dir)


        print("---------------------------------------------------")
        print("    Collecting datasets ....   ")

        print("    Datasets: ")
        print("    " + (str(cd)))
        print("---------------------------------------------------")      
        # create prefix for new name
        dprefix = explore_anonymize_caida.get_prefix_number(datact)
        dname = 'dataset_'
        dname = dname + str(dprefix) + str(datact) + '.csv'
        print("new client name: ", dname)
        anon_df = pd.read_csv(cd)
        
        # write data to csv with new name
        os.chdir(input_dir)
        print("dir where anon_df is put: ", str(input_dir))
        # recall that input_dir = path_proj / 'caida_processed'
      
        '''
        UNCOMMENT WHEN input_dir IS EMPTY
        '''
        # anon_df.to_csv(dname, encoding='utf-8', index=False, header=False)
        anon_caida_name.append(dname)
        datact += 1
        
    print("anon caida name list" , anon_caida_name)

    # dataset_newname = anon_caida_name[0]
    

    
    for client in anon_caida_name:
        os.chdir(input_dir)

        print("client: ", client)
        df = pd.read_csv(client, names=['id','time', 'rtt'])
        print("df: \n", df.head(5))
        df = df[['time', 'rtt', 'id']]
        print("df new: \n", df.head(5))
        col_num = 2
        col_name_str = 'rtt'
        client_valleys = detect_modality.find_lowest_points_between_peaks(\
                  df=df,col_name_str=col_name_str,min_peak_fraction=0.1)
        print("client_valleys: ", client_valleys)
        # subsets = detect_modality.divide_data_based_on_peaks_no_header(df, col_num, client_valleys)
        subsets = detect_modality.divide_data_based_on_peaks(df, col_name_str, client_valleys)
        print("Client: " + str(client[:-4]) + " " + str(len(subsets)) + " subsets")

        subset_count = 1

        for df_subset in subsets:
            print("subset: "+ str(subset_count))

            subset_num = 's' + str(subset_count)
            os.chdir(output_dir)
            # output_dir = path_proj_pu / 'caida_processed'
        
            #Creating a folder if it does not exist
            subset_dir_name = str(client[:-4])
            
            subset_dir = output_dir / subset_dir_name
            if not os.path.exists(subset_dir):
                os.makedirs(subset_dir)
            
            
            # find threshold
            # iqr = find_thresholds.find_iqr(df_subset, 'RTT')
            iqr = find_thresholds.find_iqr(df_subset, 'rtt')
            # mean1sd = find_thresholds.find_mean_plus_sd_thres_no_avg(df_subset, 'RTT')
            mean1sd = find_thresholds.find_mean_plus_sd_thres_no_avg(df_subset, 'rtt')
            thres = mean1sd
            thres_list = [thres]


            print("the dir where threshold csv is stored: ", str(subset_dir))
            os.chdir(subset_dir)
            find_thresholds.record_threshold(df_name_str=subset_num, dataset_name_str=client[:-4], client_list=[client], thres_list=thres_list)

            
            print("the dir where the subset is stored: ", str(subset_dir))
            
            subset_name =  str(client[:-4]) + '-'+ str(subset_num) + '.csv'
            print("subset name: ", subset_name)
  
            
            df_subset.to_csv(subset_name, encoding='utf-8', index=False, header=False)
            subset_count += 1



        


