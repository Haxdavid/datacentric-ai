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

class CurrentExp:
    def __init__(self, dataset_name, classifier_name, reduction_factor=1, random_seed=0, test_set_ratio="default_benchmark", exp_fold="simulation_results/", save_files=True, doe_param=None):
        # Initialization
        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.reduction_factor = reduction_factor
        self.random_seed = random_seed
        self.test_set_ratio = test_set_ratio
        self.exp_fold = exp_fold
        self.save_files = save_files
        self.doe_param = doe_param if doe_param else {"random_seed": 0, "start": 0, "stop": 10, "step": 1}

        # Dataset and classifier related variables
        self.current_ds = None
        self.current_cl = None
        self.current_meta = None
        self.x_t = None
        self.y_t = None
        self.df_ = None
        self.pred_ = None
        self.trace_m_ = None

    def dataset_provider(self):
        self.current_ds, self.current_meta = dataset_provider(
            name=self.dataset_name, 
            reduction_factor=self.reduction_factor, 
            test_set_ratio=self.test_set_ratio, 
            random_state=self.random_seed
        )

    def dataset_overview(self):
        if self.current_ds is None:
            self.dataset_provider()
        
        self.x_t, self.y_t = dataset_overview(
            train_test_dct=self.current_ds["y_train_small"], 
            dataset_name=self.dataset_name
        )

    def set_classifier(self):
        self.current_cl = _set_bakeoff_classifier(
            self.classifier_name, 
            random_state=self.random_seed, 
            n_jobs=1
        )

    def apply_label_errors(self, stop_percentage=0.9):
        cl_dict = {self.classifier_name: self.current_cl}
        self.df_, self.trace_m_ = apply_label_errors(
            train_test_df=self.current_ds, 
            cl_dict=cl_dict, 
            ds_=self.dataset_name, 
            stop=self.doe_param["stop"], 
            stop_percentage=stop_percentage, 
            step=self.doe_param["step"]
        )

    def run_experiment(self):
        self.dataset_provider()
        self.set_classifier()
        self.apply_label_errors()

        if self.save_files:
            self.save_results()

    def save_results(self):
        pass


