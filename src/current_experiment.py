import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict
from datetime import datetime
from matplotlib import gridspec
from src.utils import logger
from tsml_eval.publications.y2023.tsc_bakeoff.run_experiments import _set_bakeoff_classifier

from src.basic_func import dataset_provider,dataset_overview, overview_of_bakeoff_cl
from src.apply_dca import apply_label_errors
from src.classifierWrapper import BakeoffClassifier
from src.visualizations import visualize_acc_decr, visualize_trace_M

class Experiment:
    def __init__(self, config: Dict[str, Any], base_path: str, results_root: str):
        self.config = config
        self.base_path = base_path
        self.results_root = results_root
        self.random_seed = config["RANDOM_S"]
        self.dataset_name = config["DATASET_NAME"]
        self.clf_name = config["CLASSIFIER_NAME"]
        self.doe_params = config["DCA"]["DoE_param"]
        self.reduction_factor = config["REDUCTION_F"]
        self.strategy = config["DCA"]["type"]

        self.dataset = None
        self.meta_ = None
        self.classifier = None
        self.x_t = None
        self.y_t = None
        self.df_ = None
        self.pred_ = None
        self.trace_m_ = None


        self.dataset, self.meta_ = dataset_provider(name=self.dataset_name, reduction_factor=self.reduction_factor,
                                                    base_path=self.base_path, random_state=self.random_seed)
        self.classifier = BakeoffClassifier(self.clf_name, random_state=self.random_seed)
        logger.info(f"Initializing Exp with dataset: {self.dataset_name}, classifier: {self.clf_name}, strategy: {self.strategy}")
        logger.info(f"and configuration with DCA-type{self.strategy}, DoE_param: {self.doe_params}")


    def dataset_overview(self) -> None:
        self.x_t, self.y_t = dataset_overview(
            train_test_dct=self.dataset["y_train_small"], 
            dataset_name=self.dataset_name)
    
    def apply_dca(self):
        cl_dict = {self.clf_name : self.classifier}
        self.df_, self.trace_m_ = apply_label_errors(
            train_test_df=self.dataset, 
            cl_dict=cl_dict, 
            ds_=self.dataset_name, 
            doe_param=self.doe_params)

    def run_experiment(self):
        logger.info(f"Run Experiment")
        self.apply_dca()

    def acc_decr(self, save_fig=False):
        visualize_acc_decr(self.df_, save_fig=save_fig, cl_ = self.clf_name, ds_= self.dataset_name)

    def trace_M(self, save_fig=False):
        visualize_trace_M(self.trace_m_, cl_=self.clf_name, ds_ = self.dataset_name, save_fig=save_fig)








# class CurrentExp:
#     def __init__(self, dataset_name, classifier_name, reduction_factor=1, random_seed=0, test_set_ratio="default_benchmark", exp_fold="simulation_results/", save_files=True, doe_param=None):
#         # Initialization
#         self.dataset_name = dataset_name
#         self.classifier_name = classifier_name
#         self.reduction_factor = reduction_factor
#         self.random_seed = random_seed
#         self.test_set_ratio = test_set_ratio
#         self.exp_fold = exp_fold
#         self.save_files = save_files
#         self.doe_param = doe_param if doe_param else {"le_strategy":"leV1","random_seed":0,"start":0,"stop":10,"step":1}

#         # Dataset and classifier related variables
#         self.current_ds = None
#         self.current_cl = None
#         self.current_meta = None
#         self.x_t = None
#         self.y_t = None
#         self.df_ = None
#         self.pred_ = None
#         self.trace_m_ = None

#     def dataset_provider(self):
#         self.current_ds, self.current_meta = dataset_provider(
#             name=self.dataset_name, 
#             reduction_factor=self.reduction_factor, 
#             test_set_ratio=self.test_set_ratio, 
#             random_state=self.random_seed
#         )

#     def dataset_overview(self):
#         if self.current_ds is None:
#             self.dataset_provider()
        
#         self.x_t, self.y_t = dataset_overview(
#             train_test_dct=self.current_ds["y_train_small"], 
#             dataset_name=self.dataset_name
#         )

#     def set_classifier(self):
#         self.current_cl = _set_bakeoff_classifier(
#             self.classifier_name, 
#             random_state=self.random_seed, 
#             n_jobs=1
#         )

#     def apply_label_errors(self, stop_percentage=0.9):
#         cl_dict = {self.classifier_name: self.current_cl}
#         self.df_, self.trace_m_ = apply_label_errors(
#             train_test_df=self.current_ds, 
#             cl_dict=cl_dict, 
#             ds_=self.dataset_name, 
#             stop=self.doe_param["stop"], 
#             stop_percentage=stop_percentage, 
#             step=self.doe_param["step"]
#         )

#     def run_experiment(self):
#         self.dataset_provider()
#         self.set_classifier()
#         self.apply_label_errors()

#         if self.save_files:
#             self.save_results()

#     def save_results(self):
#         pass


