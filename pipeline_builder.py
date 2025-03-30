import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import pandas as pd
import warnings
import time
from tsml_eval.publications.y2023.tsc_bakeoff.run_experiments import _set_bakeoff_classifier

from basic_func import dataset_provider, dataset_overview, apply_TSC_algos
from apply_dca import apply_label_errors, visualize_acc_decr, visualize_trace_M

DATASET_NAME = "ElectricDevices"    #should be in DS_list
CLASSIFIER_NAME = "MR-Hydra"        #should be in cl_ names
REDUCTION_F = 10                    #only for large datasets
RANDOM_S = 0                        #Random Seed for everything except the DCA
DCA= "LabelErrors"                  #
DoE_PARAM = {"random_seed":0,"start":0,"stop":10,"step":5}  #stop = max 90% of test_set_size, step=1-10 
EXP_FOLD = "simulation_results/"                            #respect folder structure
SAVE_FILES = True                                           #export files and figures in the respective directorys
VIS_DATA = False                                            # Visualizes DataDistribution before applying DCA


def run_single_pipeline(ds_name, cl_name, reduction_f,random_s, dca, doe_params, exp_fold, save_files, vis_data=VIS_DATA):
    #CONSTANTS
    TEST_SET_RATIO = "default_benchmark"
    STOP_PERCENTAGE = 0.9

    current_ds, current_meta = dataset_provider(name=ds_name, reduction_factor=reduction_f, test_set_ratio=TEST_SET_RATIO,
                                                random_state=random_s)
    if vis_data == True:
        x_t, y_t = dataset_overview(train_test_dct=current_ds["y_train_small"] , dataset_name=ds_name)
    else: x_t, y_t = None,None
    
    current_cl = _set_bakeoff_classifier(cl_name, random_state=random_s, n_jobs=1)
    cl_dict = {cl_name: current_cl}
    df_, trace_M_= apply_label_errors(train_test_df=current_ds, cl_dict=cl_dict, ds_=ds_name, stop=doe_params["stop"],
                                    stop_percentage=STOP_PERCENTAGE,  step=doe_params["step"])
    res_dict = {"df_":df_,
                "trace_M_":trace_M_,
                "current_ds":current_ds,
                "current_meta":current_meta,
                "x_t_y_t":(x_t,y_t)}
    return res_dict



if __name__ == "__main__":
    res_obj = run_single_pipeline(ds_name=DATASET_NAME, cl_name=CLASSIFIER_NAME, reduction_f=REDUCTION_F,random_s=RANDOM_S, dca=DCA,
                        doe_params=DoE_PARAM, exp_fold=EXP_FOLD, save_files=SAVE_FILES, vis_data=VIS_DATA)
    print(res_obj)