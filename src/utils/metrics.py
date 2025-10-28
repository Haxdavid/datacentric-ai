import os
import numpy as np
import pandas as pd



def auc_calculator(df_: pd.DataFrame) -> float:
    """
    Calculate the Area Under the Curve (AUC) for the given DataFrame.
    The DataFrame should contain 'accuracy' and 'LE_relative' columns.
    """
    if 'accuracy' not in df_.columns or 'LE_relative' not in df_.columns:
        raise ValueError("DataFrame must contain 'accuracy' and 'LE_relative' columns.")
    if len(df_) < 2:
        raise ValueError("DataFrame must contain at least two points to compute AUC.")
    
    # Ensure the DataFrame is sorted by 'LE_relative'
    df_ = df_.sort_values(by='LE_relative')
    le_range = df_['LE_relative'].iloc[-1] - df_['LE_relative'].iloc[0]
    if le_range == 0:
        raise ValueError("LE_relative values must span a range to compute normalized AUC.")
       
    # Calculate AUC using the trapezoidal rule
    auc = np.trapz(df_['accuracy'], df_['LE_relative'])
    initial_accuracy = df_['accuracy'].iloc[0]
    normalized_auc = auc / (initial_accuracy * le_range)
    #print(f"Area Under the Curve (AUC): {auc:.5f}")
    #print(f"Normalized Area Under the Curve (AUC): {normalized_auc:.5f}")
    return round(normalized_auc, 5)


def original_accuracy(df_: pd.DataFrame) -> float:
    """
    Calculate the original accuracy from the given DataFrame.
    The DataFrame should contain 'accuracy' and 'LE_relative' columns.
    """
    original_acc = df_['accuracy'].iloc[0]
    #print(f"Original Accuracy: {original_acc:.5f}")
    return round(original_acc, 5)


def acc_robustness_calculator(df_: pd.DataFrame) -> float:
    """
    Calculate the robustness of accuracy for the given DataFrame.
    The DataFrame should contain 'accuracy' and 'LE_relative' columns.
    """
    if 'accuracy' not in df_.columns or 'LE_relative' not in df_.columns:
        raise ValueError("DataFrame must contain 'accuracy' and 'LE_relative' columns.")
    
    accuracies = df_['accuracy'].tolist()

    # First acc_
    acc_0 = accuracies[0]
    if acc_0 == 0:
        raise ZeroDivisionError("Clean accuracy (acc_0) cannot be zero.")
    #TODO: Maybe add a check of equidistant LE_relative values

    #Sum of following relative accuracies to the clean accuracy divided by the number of perturbations
    perturbed_accuracies = accuracies[1:]
    robustness = sum(acc / acc_0 for acc in perturbed_accuracies) / len(perturbed_accuracies)
    #print(f"Robustness of Accuracy: {robustness:.5f}")
    return round(robustness, 5)


def early_degradation_point(df_ : pd.DataFrame) -> float:
    """
    Calculate the early degradation point from the DataFrame.
    The DataFrame should contain 'accuracy' and 'LE_relative' columns.
    """
    DEGRADATION_THRESHOLD = 0.9  # Threshold for early degradation

    if 'accuracy' not in df_.columns or 'LE_relative' not in df_.columns:
        raise ValueError("DataFrame must contain 'accuracy' and 'LE_relative' columns.")

    # Find the first point where accuracy drops below 0.9 or clean data acc
    acc_0 = df_['accuracy'].iloc[0]
    acc_degradation_relative = acc_0 * DEGRADATION_THRESHOLD
    early_degradation = df_[df_['accuracy'] < acc_degradation_relative]
    if early_degradation.empty:
        print("No early degradation point found.")
        print("No early degradation point found: accuracy never dropped below threshold.")
        return np.nan

    ed_point = early_degradation['LE_relative'].iloc[0]
    #print(f"Early Degradation Point: {ed_point:.5f}")
    return round(ed_point, 5)


def average_train_time(df_: pd.DataFrame) -> float:
    """
    Calculate the average training time from the given DataFrame.
    """
    if 'train_time' not in df_.columns:
        raise ValueError("DataFrame must contain a 'train_time' column.")
    
    avg_time = df_['train_time'].mean()
    return round(avg_time, 5)


def average_eval_time(df_: pd.DataFrame) -> float:
    """
    Calculate the average evaluation time from the given DataFrame.
    """
    if 'eval_time' not in df_.columns:
        raise ValueError("DataFrame must contain an 'eval_time' column.")
    
    avg_time = df_['eval_time'].mean()
    return round(avg_time, 5)




METRIC_FUNCTIONS = {
    "initial_accuracy": original_accuracy,
    "auc_score": auc_calculator,
    "acc_robustness": acc_robustness_calculator,
    "early_degradation" : early_degradation_point,
    "avg_train_time" : average_train_time,
    "avg_eval_time": average_eval_time,
}