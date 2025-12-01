import os
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from src.utils.utilizations import setup_logger
logger = setup_logger("Apply_LE_Logger") # Use module-specific logger



def init_le_params(le_strategy, p_vector, train_test_df):
    label_names = np.unique(train_test_df["y_train_small"], return_counts=False)
    logger.info(f"label_names: {label_names}") 
    if le_strategy in ["default", "V1", "leV1"]:
        p_vector_temp = [np.round(1/label_names.size, 4) for label in label_names]
        p_vector_dict = {label:np.round(1/label_names.size, 4) for label in label_names }
        logger.info("Current Label Error Strategy: DEFAULT: leV1")
        logger.info(f"The p_vector for the current_experiment: {p_vector_temp}")
        return "leV1", p_vector_temp, p_vector_dict
    elif le_strategy == "V2" or le_strategy =="leV2":
        if p_vector is None:
            raise ValueError("p_vector is not provided. If you want to use LE strategy V2 ensure your p_vector is valid")
        elif len(p_vector) != label_names.size:
            raise ValueError("p_vector does not match the number of classes for the current dataset choice")
        p_vector_dict = {label:np.round(p_value, 4) for label, p_value in zip(label_names, p_vector)}
        logger.info(f"The p_vector for the current_experiment: {p_vector_dict}")
        return "leV2", p_vector, p_vector_dict
    elif le_strategy == "V3" or le_strategy == "leV3":
        if p_vector is None:
            raise ValueError("p_vector is not provided. If you want to use LE strategy V3 ensure your p_vector is valid")
        else:
            p_vector = np.array(p_vector)
            logger.info(f"p_vector: {p_vector}")      
            if p_vector.shape[0] != label_names.size:
                raise ValueError("p_vector has to be a noise matrix CxC")
        p_vector_dict = {label:row for label, row in zip (label_names, p_vector)}
        logger.info(f"The p_vector for the current_experiment is a matrix, with the probs:\n{p_vector_dict}")
        return "leV3", p_vector, p_vector_dict
        
    
    else:
        raise ValueError(f"Unknown le_strategy: {le_strategy}. Please choose a valid strategy ('default', 'V1', 'leV1', 'V2', 'leV2', 'leV3').")


def percentage_to_instance_converter(doe_param, train_test_df):
    """Convert percentage-based DOE parameters to instance-based parameters.
       This function ensures:
        - the step size is a valid integer and does not exceed the number of instances.
        - the stop value is less than or equal to 99% of the total instances.
       !IMPORTANT!: The function logic ensures that the requested relative label error is always MET. Which means that
        the stop value is one step ABOVE the LOWER_THRESHOLD. ONLY IF this exceeds the UPPER_THRESHOLD this Condition is not fulfilled!

       RETURNS: doe_param with updated 'stop' and 'step' values based on the number of instances.
    """
    UPPER_THRESHOLD = 0.99                        #this value will not be exceeded unless 100% is explicitly requested.
    LOWER_THRESHOLD = doe_param["stop"] * 1/100  #This value will be exceeded if not exactly met.
    # READ IMPORTANT NOTE ABOVE concerning the handling of different inputs & when changing THRESHOLDS


    doe_param = doe_param.copy()
    try:
        instances_no = train_test_df["y_train_small"].shape[0]
    except:
        instances_no = len(train_test_df["y_train_small"])

    # for start=0
    percentage_start = doe_param["start"]
    percentage_stop = doe_param["stop"]
    percentage_step = doe_param["step"]

    # Validate types
    if not all(isinstance(v, (int, np.integer)) for v in [percentage_start, percentage_stop, percentage_step]):
        raise TypeError(
            "DOE percentage parameters ('start', 'stop', 'step') must all be integers. "
            f"Received types: start={type(percentage_start)}, stop={type(percentage_stop)}, step={type(percentage_step)}"
        )

    # Validate start value
    if percentage_start != 0:
        raise ValueError(
            "In the current pipeline implementation, 'start' must always be 0. "
            "Non-zero start values are not supported; existing configuration results will be skipped accordingly."
        )
    
    no_perc_steps = int(percentage_stop/percentage_step) #should be integer because 2 --> 29 should be invalid 
    
    requested_instance_step = instances_no * percentage_step/100
    instances_step = int(np.round(instances_no * percentage_step/100))
    transformed_percentage_step = np.round(instances_step/instances_no * 100, 4)

    logger.info("Converting percentage-based DOE parameters to instance-based parameters")
    logger.info(f"requested_instance_step = {requested_instance_step} will be transformed into {instances_step}")
    logger.info(f"requested_percentage_step = {percentage_step} % || transformed into {transformed_percentage_step} %")
    if instances_step == 0:
        instances_step = 1
        logger.info("requested instances per step < 0.5 --> rounded up to 1 because its smallest possible increment")

    max_steps = int(instances_no/instances_step)  # example: 390 / 8 = 48.75 --> 48


    while no_perc_steps * instances_step > instances_no * LOWER_THRESHOLD:                #Ensure we start below the LOWER_threshold
        no_perc_steps -= 1
    while no_perc_steps * instances_step < instances_no * LOWER_THRESHOLD:  #Ensure we increase back up again to reach the first instance above the LOWER_THRESHOLD
        no_perc_steps += 1
        logger.info(f"requested_number_of_percentage_steps = {no_perc_steps -1} was increased by one")
    while max_steps * instances_step > instances_no * UPPER_THRESHOLD:  #Ensure we do not exceed a threshold
        max_steps -= 1

    # Ensure that the number of percentage steps does not exceed the maximum allowed steps
    if not no_perc_steps <= max_steps: 
        no_perc_steps = max_steps
        logger.info("Converting percentage-based DOE parameters to instance-based parameters")
        print("Cap reached")


    instances_stop = instances_step*no_perc_steps # > lower Threshold but <= upper threshold 
    #100 PERCCENT CASE 
    # Pipeline has to handle this CASE appropriately to ensure to NOT exceed the numbers with the given stepsize
    if percentage_stop == 100:
        instances_stop = instances_no
        logger.info("stop value is 100% --> set to number of instances")

    doe_param["stop"]=instances_stop
    doe_param["step"]=instances_step

    return doe_param


def create_label_flip_trajectory(train_array: np.ndarray, 
                            p_vector: Union[List[float], np.ndarray], 
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
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    total_instances = len(train_array)
    if not isinstance(train_array, np.ndarray):
        raise TypeError(f"`train_array` must be a numpy array, got {type(train_array).__name__}.")
    if not isinstance(p_vector, list):
        if not isinstance(p_vector, np.ndarray):
            raise TypeError(f"`p_vector` must be a list or matrix of floats, got {type(p_vector).__name__}.")
    if seed is not None:
        np.random.seed(seed)

  
    noise_matrix_mode = False
    if p_vector.ndim == 2:
        noise_matrix_mode = True

    # Tracking structures
    flipped_indices = set()
    log = []
    # class_members[class_label] = set of original (not-yet-flipped) indices
    class_members = {cls: set(np.where(train_array == cls)[0]) for cls in unique_classes}
    received_instances = {cls: 0 for cls in unique_classes}  # Track flips *to* each class

    flip_counter = 0


    #DEBUG
    if noise_matrix_mode:
        # Normalize row-wise for safety
        noise_matrix = p_vector.astype(float)
        noise_matrix_classes = noise_matrix.sum(axis=1)
        noise_matrix_class_norm = noise_matrix_classes / noise_matrix_classes.sum()
        logger.info("Using noise matrix mode: row-normalized matrix activated.")
        logger.info(f"noise_matrix_class_norm: {noise_matrix_class_norm}")

    else: 
        p_vector_norm = np.array(p_vector) / sum(p_vector)
        logger.info(f"sum of p_vector before normalizing = {sum(p_vector)} "
                f"// sum after = {sum(p_vector_norm)}")

    # Label-Flipping Mechanism
    while len(flipped_indices) < total_instances:
        # --- Choose source class based on p_vector ---
        if noise_matrix_mode:
            source_class = np.random.choice(unique_classes, p=noise_matrix_class_norm)

        else:
            source_class = np.random.choice(unique_classes, p=p_vector_norm)

        # --- Skip if no unflipped indices remain ---
        candidate_indices = class_members[source_class] - flipped_indices
        if not candidate_indices:
            continue

        # --- Choose target class randomly, different from source ---
        if noise_matrix_mode:
            src_idx = class_to_idx[source_class]
            row = noise_matrix[src_idx].copy()
            class_prob = row / row.sum()
            target_class = np.random.choice(unique_classes, p= class_prob)
            
        else:
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
                             p_vector: Union[List[float], np.ndarray],
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
    if random_seed == 0:
        filename = f"{dataset}_le_traj.parquet"
    else:
        filename = f"{dataset}_le_traj{random_seed}.parquet"

    file_path = os.path.join(full_path, filename)


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