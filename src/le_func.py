import os
import re
import numpy as np
import pandas as pd
import warnings
from typing import List, Optional
from src.utils import setup_logger
logger = setup_logger("Apply_LE_Logger") # Use module-specific logger

def create_label_flip_trajectory(train_array: np.ndarray, 
                            p_vector: List[float], 
                            seed: Optional[int] = None) -> pd.DataFrame:
    """
    Flips labels in the train_set according to a specified probability distribution (p_vector).
    
    Parameters:
    - train_array: np.ndarray, the training set labels.
    - p_vector: list or np.ndarray, probabilities for each class.
    - seed: int, optional random seed for reproducibility.
    
    Returns:
    - pd.DataFrame with flip logs.
    """

    # --- Input validation --- #
    unique_classes = np.unique(train_array)
    total_instances = len(train_array)
    if not isinstance(train_array, np.ndarray):
        raise TypeError(f"`train_array` must be a numpy array, got {type(train_array).__name__}.")
    if not isinstance(p_vector, list):
        raise TypeError(f"`p_vector` must be a list of floats, got {type(p_vector).__name__}.")
    if not all(isinstance(p, float) for p in p_vector):
        raise TypeError("All elements of `p_vector` must be floats.")
    if len(unique_classes) != len(p_vector):
        logger.warning(f"Number of unique classes ({len(unique_classes)}) does not match length of p_vector ({len(p_vector)}).")
    if seed is not None:
        np.random.seed(seed)

 
    
    # Tracking structures
    flipped_indices = set()
    log = []
    # class_members[class_label] = set of original (not-yet-flipped) indices
    class_members = {cls: set(np.where(train_array == cls)[0]) for cls in unique_classes}
    received_instances = {cls: 0 for cls in unique_classes}  # Track flips *to* each class

    flip_counter = 0


    #DEBUG

    p_vector_norm = p_vector / sum(p_vector)
    logger.info(f"sum of p_vector before normalizing = {sum(p_vector)} // sum after = {sum(p_vector_norm)}")

    # Label-Flipping Mechanism
    while len(flipped_indices) < total_instances:
        # --- Choose source class based on p_vector ---
        source_class = np.random.choice(unique_classes, p=p_vector_norm)

        # --- Skip if no unflipped indices remain ---
        candidate_indices = class_members[source_class] - flipped_indices
        if not candidate_indices:
            continue

        # --- Choose target class randomly, different from source ---
        target_choices = [cls for cls in unique_classes if cls != source_class]
        target_class = np.random.choice(target_choices)

        # --- Pre-check: will current flip make source_class empty? ---
        if len(candidate_indices) == 1 and received_instances[source_class] == 0:
            logger.warning(f"Skipping flip from {source_class} to {target_class} as it would empty the source class.")
            # This is the last original instance in the source class, and nothing has been flipped to it
            continue  # skip and try again

        # --- Proceed with flipping mechanism ---
        flip_index = np.random.choice(list(candidate_indices))

        # Update tracking
        flipped_indices.add(flip_index)
        class_members[source_class].remove(flip_index)
        received_instances[target_class] += 1

        # Log the flip
        flip_counter += 1
        log.append({
            "instances": flip_counter,
            "where": flip_index,
            "from": source_class,
            "to": target_class
        })

    return pd.DataFrame(log)





def check_for_le_trajectory(le_trajectory_dir: str,
                             leV: str,
                             dataset: str,
                             train_arr: np.ndarray,
                             p_vector: List[float],
                             random_seed: int) -> pd.DataFrame:
    """
    Check if a label error trajectory file exists for the given dataset and version.
    
    Parameters:
    - le_trajectory_dir: str, directory where label error trajectories are stored.
    - leV: str, version of the label error trajectory.
    - dataset: str, name of the dataset.
    - train_arr: array, y_train
    - p_vector: list, floats of probabilities for each class
    - random_seed: integer, seed for randomization
    
    Returns:
    - bool: True if the file exists, False otherwise.
    """
    # Build directory and file path
    #full_path = os.path.join(le_trajectory_dir, dataset, leV) #already receiving full path
    full_path = le_trajectory_dir
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, f"{dataset}_le_traj.parquet")

     # Check if file exists
    if os.path.exists(file_path):
        logger.info(f"Loading existing trajectory from {file_path}")
        le_trajectory = pd.read_parquet(file_path)
    
    else:
        logger.info(f"No trajectory found. Creating new trajectory at {file_path}")
        le_trajectory = create_label_flip_trajectory(train_array=train_arr,
                                                     p_vector=p_vector,
                                                     seed=random_seed)
        le_trajectory.to_parquet(file_path, index=False)

    return le_trajectory



def reconstruct_state_y_train(initial_y_train: np.ndarray,
                            flip_trajectory: pd.DataFrame,
                            k: int) -> np.ndarray:
    """
    Parameters:
    ----------
    initial_y_train : np.ndarray
        Original labels (before any flips). This should be a 1D array of class names or label identifiers.
    
    flip_trajectory : pd.DataFrame
        DataFrame describing the flip operations. Must contain the columns:
        - 'instances': flip number (1-based)
        - 'where': index in y_train to flip
        - 'from': original class label
        - 'to': new class label
    
    k : int
        Number of flips to apply (0 ≤ k ≤ len(flip_trajectory)).
        - If k = 0 → returns the original `initial_y_train`.
        - If k = len(flip_trajectory) → returns the fully flipped dataset.
        - If k is in between → returns the state after the first `k` flips.

    Returns:
    -------
    np.ndarray
        A new NumPy array representing the labels after `k` flips.
    """
    
    y = initial_y_train.copy()
    for row in flip_trajectory.iloc[:k].itertuples():
        y[row.where] = row.to
    return y



def full_flip_handler(step_, error_increasement, error_stop):
    """
    Handles the case where the step exceeds the error_increasement and adjusts the step accordingly.
    
    Parameters:
    - step_: int, current step value.
    - error_increasement: int, increment value for error.
    - error_stop: int, maximum value for error.
    
    Returns:
    - Adjusted step_, error_increasement, and error_stop values.
    """
    if step_  > error_stop + 1:  # 168 > 169
        # Adjust step to not exceed the error_stop
        error_increasement = error_stop - step_ + 1
        logger.warning(f"Error Increasement {error_increasement} adjusted to not exceed error stop {error_stop}.")
    
    return step_, error_increasement, error_stop 