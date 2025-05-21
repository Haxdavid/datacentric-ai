import site
import os
import sys
import ssl
import sktime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score
from aeon.datasets.tsc_datasets import univariate, univariate_equal_length
from tsml_eval.publications.y2023.tsc_bakeoff.set_bakeoff_classifier import bakeoff_classifiers

from src.utils import setup_logger
logger = setup_logger("Basic_Func_Logger")
logger.info("Custom-named logger active.")

###BASIC FUNCTIONS
###(0) pre_functions 
### 1. dataset_provider
### 2. dataset_overview
### 3. apply_TSC_algos

CURRENT_TSC_DATASETS = univariate_equal_length
CLASSIFIERS_CATEGORIZED = {
    "distance_based": ["1NN-DTW", "GRAIL"],
    "feature_based": ["Catch22", "FreshPRINCE", "TSFresh", "Signatures"],
    "shapelet_based": ["STC", "RDST", "RSF", "MrSQM"],
    "interval_based": ["R-STSF", "RISE", "TSF", "CIF", "STSF", "DrCIF", "QUANT"],
    "dictionary_based": ["BOSS", "cBOSS", "TDE", "WEASEL", "WEASEL_V2"],
    "convolution_based": ["ROCKET", "MiniROCKET", "MultiROCKET", "Arsenal", "Hydra", "MR-Hydra"],
    "deep_learning": ["CNN", "ResNet", "InceptionTime", "H-InceptionTime", "LITETime"],
    "hybrid": ["HC1", "HC2", "RIST"]
}


def check_for_valid_dataset(ds_name_, ds_list):
    if ds_name_ in ds_list:
        return True
    else:
        return False

def check_for_valid_cl(cl_name_, cl_list):
    if cl_name_ in cl_list:
        return True
    else:
        return False
    
def overview_of_bakeoff_cl(show_all_possible_names=False):
    print(CLASSIFIERS_CATEGORIZED)
    if show_all_possible_names:
        print("classifier names can be written in several different naming conventions.")
        print("Here is a list of possible alternative names for the accesible bakeoff classifiers")
        for cl_names_ in bakeoff_classifiers:
            print(cl_names_)


def dataset_provider(name="FaceAll", reduction_factor=20, test_set_ratio="default_benchmark", random_state=42,
                    current_ds=CURRENT_TSC_DATASETS):
    """
    RECEIVE: dataset name to load the dataset from 112UCRFolds/...
             reduction factor to randomy reduce the data. min=1 max=?
             test_set_ratio to specify the train test split if default_benchmark split is not wanted.
             random_state only relevant if manual train test split is applied. 
             current_ds: ? 
    RETURN: train_test_dct with 8 columns(keys), meta_ 
    """

    #TODO overwork architecture. add predefined resample. connect with load_from_tsl_file to receive exact <name>0.TRAIN.ts ?
    if name not in current_ds:
        raise ValueError(f"Dataset {name} is not available in the dataset list.")
 
    # Differentiate between benchmark split & manual split  
    if test_set_ratio == "default_benchmark":
        X_train, y_train, meta_ = load_classification(name=name, split="train" , return_metadata=True,
                                  extract_path="/Users/david/Documents/Studium D&E/Applied AI/David_MA/112UCRFolds") 
        X_test, y_test = load_classification(name=name, split="test", return_metadata=False,
                                  extract_path="/Users/david/Documents/Studium D&E/Applied AI/David_MA/112UCRFolds")
    else:
        X_, y_, meta_ = load_classification(name=name, return_metadata=True,
                                  extract_path="/Users/david/Documents/Studium D&E/Applied AI/David_MA/112UCRFolds") 
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=test_set_ratio, random_state=random_state)


    # Shuffle indices and select a smaller subset
    if reduction_factor != 1:
        np.random.seed(random_state)
        train_indices = np.random.permutation(len(X_train))
        test_indices = np.random.permutation(len(X_test))
        reduced_train_indices = train_indices[:len(X_train) // reduction_factor]
        reduced_test_indices = test_indices[:len(X_test) // reduction_factor]

        X_train_small, y_train_small = X_train[reduced_train_indices], y_train[reduced_train_indices]
        X_test_small, y_test_small = X_test[reduced_test_indices], y_test[reduced_test_indices]
        #---> Different return types ! 
    else:
        X_train_small, y_train_small, X_test_small, y_test_small = X_train, y_train, X_test, y_test


    train_test_dct = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
                        "X_train_small": X_train_small,"y_train_small": y_train_small, "X_test_small": X_test_small,"y_test_small": y_test_small }
    for name, array in train_test_dct.items():
        print(f"{name:<20}: {array.shape}")

    return train_test_dct, meta_


def dataset_overview(train_test_dct, dataset_name="FaceAll", top_gap=0.18):
    """
    RECEIVE: train_test_array with the intern format (DataFrame with the columns X_train, X_test, y_train, y_test
            and their reduced identity (X_train_small)), list of classifiers
    RETURN: x_ticks, y_ticks for the plot
    """       
    y_df = pd.DataFrame(train_test_dct, columns=["Label"])

    # Create the count plot
    plt.figure(figsize=(7, 2.8), dpi=120)
    ax = sns.countplot(data=y_df, x=y_df["Label"], hue=y_df["Label"] ,palette='muted') #order=Y_data['activity_label'].value_counts().index)

    # Add annotations
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), 
                    textcoords='offset points')

    # Customizing plot aesthetics
    ax.set_title(f'label distribution [{dataset_name}]', fontsize=18, weight='bold')
    ax.set_xlabel('Labels', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    max_count = y_df.value_counts().max()  # Find the max count
    y_lim = plt.ylim(0, max_count * (1+top_gap))  # Increase the y-axis limit by top_gap%
    x_ticks= plt.xticks(fontsize=10)
    y_ticks = plt.yticks(fontsize=12)

    return x_ticks, y_ticks


def apply_TSC_algos(train_test_dct, classifiers, exclude_classifiers=[" "]):
    """
    RECEIVE: train_test_dct with the intern format (Dict or DataFrame with the columns X_train, X_test, y_train, y_test
            and their reduced identity (X_train_small)), DICT of classifiers. potential of exclude classifiers
            present in classifier dict.
    RETURN: pred_dict: a nested dict with classifier names as keys for several prediction dicts containing
            multiple performance metrics and y_pred,y_pred_prob arrays.
    SUPPORTS: singe OR multiple classifiers at once. Pipeline is constructed for simple classifier usage
    """
    DEBUG = False
    pred_dict = {} #{"alg_name": {"accuracy": 0.0, "y_true": [0,0,0] "y_pred": [0,0,0], "y_pred_prob": [0,0,0]}}
    # Looping through the classifiers
    if not isinstance(classifiers, dict):
        raise TypeError("Classifiers should be a dictionary with names as keys and classifier objects as values.")
   
    for name, clf in classifiers.items():
        if name not in exclude_classifiers:
            start_time = time.time()
            print(f"\n\nClassifier: {type(clf).__name__}")
            clf.fit(train_test_dct["X_train_small"],train_test_dct["y_train_small"])
            train_time = time.time() - start_time
            y_pred = clf.predict(train_test_dct["X_test_small"])
            y_pred_prob = clf.predict_proba(train_test_dct["X_test_small"])
            acc_score = accuracy_score(train_test_dct["y_test_small"], y_pred)
            #nll = log_los(train_test_array["y_test_small"], y_pred)
            #balanced_acc
            #AUROC=
            eval_time = time.time() - start_time - train_time
            #print("---------------------------- "+ f"Train time={train_time:.2f}s, Eval Time={eval_time:.2f}s")
            logger.info("------------------------"+ f"Train time={train_time:.2f}s, Eval Time={eval_time:.2f}s")
            pred_dict[name] = {"accuracy":acc_score,"y_train":train_test_dct["y_train_small"], "y_pred":y_pred,"y_pred_prob":y_pred_prob}
        else:
            pass
        
    print("\n" + f'{"Algorithm":<34}{"Accuracy"}')   
    for name, pred in pred_dict.items():
        acc = pred["accuracy"]
        print(f"{name:<33} {acc:.4f}")

    if DEBUG:
        return pred_dict, train_test_dct
    return pred_dict    