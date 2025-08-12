import os
import re
import random
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Union

from src.basic_func import apply_TSC_algos
from src.utils import setup_logger
from src.classifierWrapper import BakeoffClassifier, assign_GPU
from src.le_func import reconstruct_state_y_train, check_for_le_trajectory, full_flip_handler


logger = setup_logger("Apply_DCA_Logger") # Use module-specific logger
logger.info("apply_dca.py logger active")
logging.getLogger("matplotlib").setLevel(logging.WARNING)



###----------------------------------HELPER FUNC--------------------------------------###
def check_for_results(target_directory, filename_list , leV, randomS, start, stop, step):
    """ Check if currenct_classifer/current_dataset directory exists in current_directory
        Check if DesignOfExperiment parameter are already present
        ---> Justify equality for: LabelErrorVersion;RandomS;Start;Stop;Step
        - Exact match (leV, randomS, start, stop, step). (Case1)
        - Partial match (same leV&randomS, different stop or step) (Case2)
          |-- (historic_stop >= stop) & (historic_step <= step) --> load_historic_res --> continue exp (Case2.1)
                                     |-- (historic_step > step) --> run whole exp from start to stop (OR run just the missing ones?..) (Case2.2)
          |-- (historic_stop < stop) & (historic_step <= step) --> load_historic_res UNTIL the current stop. Cut the rest. (Case2.1)
                                     |-- (historic_step > step) --> run the whole exp from start to stop (Case2.3)
    """
    historic_steps = []
    partial_matches = []
    coarse_matches = []
    pattern = re.compile(rf"{leV}_{randomS}_(\d+)_(\d+)_(\d+)")
    logger.info(f"Searching inside {target_directory} for results")
    logger.info(f"Looking for files matching pattern: {pattern.pattern} with start={start}, stop={stop}, step={step}")
    
    # List all subdirectories
    try:
        existing_dirs = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]
        logger.info(f"📁 Found directories: {existing_dirs}")
    except FileNotFoundError:
        logger.info(f"❌ Directory does not exist. No previous experiments found.")
        return {"status": "no_results_present"}
    
    k=0

    for dir_name in existing_dirs:
        match = pattern.fullmatch(dir_name)
        if match:
            file_start, file_stop, file_step = map(int, match.groups())
            dir_path = os.path.join(target_directory, dir_name)
            k += 1

            # ✅ Case 1: Exact match found
            if file_start == start and file_stop == stop and file_step == step:
                logger.info(f"✅ Exact match found: {dir_name}")
                return {
                    "status":"exact_match",
                    "load_existing": dir_path, # changed full_path --> dir_path
                    "latest_stop": stop
                }

            # ✅ Case 2: Partial match (same start but different stop or smaller step (or both))
            if file_start == start and file_step <= step:
                partial_matches.append((dir_path, file_stop, file_step))
                logger.info(f"🟡 Partial Match found: {dir_name}")
            

            # Case 3: Coarse partial match (same start but coarser step)
            if file_step > step:
                historic_steps.append(file_step)
                if file_start == start:
                    coarse_matches.append((dir_path, file_stop, file_step))
                    logger.info(f"🟡 Coarse Match found: {dir_name}")
            

    # Print out the historic_steps for clarity
    if len(historic_steps) == k != 0:
        print("All historic Steps are to coarse. Historic steps: {}".format(historic_steps))
        print("Cannot load any file which meets the requested stepsize of:{} ".format(step))


    # CASE 2: Process Partial Matches 
    # This CASE will only be reached if historic_step is finer or equal to the required step
    if partial_matches:
        # Find the closest valid stop that allows continuation
        closest_file = None
        closest_stop = None
        closest_step = None

        for dir_path, hist_stop, hist_step in partial_matches:
            if hist_stop < stop:  # Case2.1 experiment can be continued from historic_stop
                if closest_stop is None or hist_stop > closest_stop:  # Get latest valid historic_stop
                    closest_file, closest_stop, closest_step = dir_path, hist_stop, hist_step
                    logger.info("Succesfully loaded closest_file with same or finer stepsize")
            else: #Case2.2 experiment was executed further. Load and Trim
                closest_file, closest_stop, closest_step = dir_path, hist_stop, hist_step
                logger.info("Closest stop is higher than requested. Loaded trimmed historic file")
                return {
                    "status":"load_and_trim",    
                    "load_existing": dir_path, #changed closest_file ---> dir_path
                    "latest_stop": closest_stop,
                }

        if closest_file:  # Case2.1 If we found a valid partial match to continue
            logger.info(f"Continuing from {closest_stop} to {stop} with step {step}.")
            return {
                "status":"load_and_continue",
                "load_existing":closest_file,
                "latest_stop": closest_stop,
            }

    #CASE3 Neither Exact Match nor partial Match found
    if coarse_matches:
        # Find the closest valid stop that allows continuation
        closest_file = None
        closest_stop = None
        closest_step = None

        for dir_path, hist_stop, hist_step in coarse_matches:          
            if hist_stop < stop: # Case3.1 experiment can be continued from historic stop, but missing lines can be skipped and processed
                if closest_stop  is None or hist_stop > closest_stop : # get the latest.
                    closest_file, closest_stop, closest_step = dir_path, hist_stop, hist_step
                    logger.warning("update closest file with COARSER stepsize")
            else: #Case3.2 experiment was executed further. Only executing between already existing steps
                closest_file, closest_stop, closest_step = dir_path, hist_stop, hist_step
                logger.warning("closest stop is higher then requested. BUT stepsize is to coarse")
                return {
                    "status": "load_and_trim_and_fill_between",
                    "load_existing": dir_path,
                    "latest_stop": closest_stop
                }
            
    # Case3.1 If we found a valid coarser partial match to continue
        if closest_file:  
            logger.info(f"Continuing from {closest_stop} to {stop} with step {step} and fill between historic step {closest_step}")
            return {
                "status":"load_and_fill_between_and_continue",
                "load_existing":closest_file,
                "latest_stop": closest_stop,
            }



    #CASE4 Neither exact nor partial match nor Coarser Match found.        
    else:
        print("results are not present with the current experiment parameters")
        print("There is [1] no matching labelerror Version and [2] no matching randomSeed or [3] no experiment at all")
        return {"status":"no_results_present"}  #If any file with DesignOfExperiment Parameters is not already present


#historic
def save_history_df(RES_PATH, df):
    """
    Save the structured experiment results:
    - metrics.json: stores step, LE_instances, LE_relative, accuracy
    - y_train_history.npy, y_pred.npy, y_pred_prob.npy: store corresponding array-like results
    """
    os.makedirs(RES_PATH, exist_ok=True)
    logger.info("HISTORIC_SAVING_USED!")

    # Save metrics (as list of dicts for easier loading)
    metrics_cols = ["step", "LE_instances", "LE_relative", "accuracy"]
    if "train_time" in df.columns and "eval_time" in df.columns:
        metrics_cols += ["train_time", "eval_time"]
    metrics_records = df[metrics_cols].to_dict(orient="records")
    with open(os.path.join(RES_PATH, "metrics.json"), "w") as f:
        json.dump(metrics_records, f, indent=4)


    df["y_train_history"] = df["y_train_history"].apply(lambda x: [int(i) for i in x])
    df["y_pred"] = df["y_pred"].apply(lambda x: [int(i) for i in x])
    np.save(os.path.join(RES_PATH, "y_train_history.npy"), df["y_train_history"].tolist())
    np.save(os.path.join(RES_PATH, "y_pred.npy"), df["y_pred"].tolist())
    np.save(os.path.join(RES_PATH, "y_pred_prob.npy"), df["y_pred_prob"].tolist())

    print(f"✅ Results saved in: {RES_PATH}")


def save_history_df_compressed(RES_PATH, df):
    """
    Save the structured experiment results:
    - metrics.json: stores step, LE_instances, LE_relative, accuracy
    - y_train_history.npy, y_pred.npy, y_pred_prob.npy: store corresponding array-like results
    """
    os.makedirs(RES_PATH, exist_ok=True)

    # Save metrics (as list of dicts for easier loading)
    metrics_cols = ["step", "LE_instances", "LE_relative", "accuracy"]
    if "train_time" in df.columns and "eval_time" in df.columns:
        metrics_cols += ["train_time", "eval_time"]
    metrics_records = df[metrics_cols].to_dict(orient="records")
    with open(os.path.join(RES_PATH, "metrics.json"), "w") as f:
        json.dump(metrics_records, f, indent=4)


    #df["y_train_history"] = df["y_train_history"].apply(lambda x: [int(i) for i in x])  RAUS
    df["y_pred"] = df["y_pred"].apply(lambda x: [int(i) for i in x])

    # Save array-like columns as .npz
    y_pred_prob_array = np.array(df["y_pred_prob"].tolist(), dtype=np.float16)
    y_train_array = np.array(df["y_train_history"].tolist(), dtype=np.uint8)  # or uint16 if needed
    ### !!!!y_train array will already be there in the new implementation logic!!!
    ### Dont forget do delete the saving of y_train_history below!!!

    y_pred_array = np.array(df["y_pred"].tolist(), dtype=np.uint8) #uint8 up to 256 classes

    #np.savez_compressed(os.path.join(RES_PATH, "y_train_history.npz"), y_train=y_train_array)  RAUS
    np.savez_compressed(os.path.join(RES_PATH, "y_pred.npz"), y_pred=y_pred_array)
    np.savez_compressed(os.path.join(RES_PATH, "y_pred_prob.npz"), y_pred_prob=y_pred_prob_array)

    logger.info(f"✅ Results saved in: {RES_PATH}")


def load_history_df_safely(load_path, le_trajectory, initial_y_train):
    # Load metrics
    with open(os.path.join(load_path, "metrics.json"), "r") as f:
        metrics = json.load(f)
    df = pd.DataFrame(metrics)
    if "train_time" not in df.columns:
        df["train_time"] = np.nan
        df["eval_time"] = np.nan

    df["step"] = df["step"].astype(int)
    df["LE_instances"] = df["LE_instances"].astype(int)
    # --- y_train_history calculation ---
    #For each instance in df["LE_instances"] we need to calculate the corresponding y_train_history
    df["y_train_history"] = [reconstruct_state_y_train(initial_y_train, le_trajectory, k=int(row["LE_instances"])).astype(str)
        for _, row in df.iterrows()
    ]

    # --- y_pred ---
    y_pred_path_npz = os.path.join(load_path, "y_pred.npz")
    y_pred_path_npy = os.path.join(load_path, "y_pred.npy")
    if os.path.exists(y_pred_path_npz):
        y_pred_array = np.load(y_pred_path_npz)["y_pred"]
    elif os.path.exists(y_pred_path_npy):
        y_pred_array = np.load(y_pred_path_npy, allow_pickle=True)
    else:
        raise FileNotFoundError("y_pred file not found")


    # --- y_pred_prob ---
    y_prob_path_npz = os.path.join(load_path, "y_pred_prob.npz")
    y_prob_path_npy = os.path.join(load_path, "y_pred_prob.npy")
    if os.path.exists(y_prob_path_npz):
        y_pred_prob_array = np.load(y_prob_path_npz)["y_pred_prob"]
    elif os.path.exists(y_prob_path_npy):
        y_pred_prob_array = np.load(y_prob_path_npy, allow_pickle=True)
    else:
        raise FileNotFoundError("y_pred_prob file not found")

    df["y_pred"] = [np.array(x, dtype=str) for x in y_pred_array]
    df["y_pred_prob"] = list(y_pred_prob_array)

    logger.info(f"✅ Results loaded from: {load_path}")
    return df

###historic
def load_history_df(load_path):
    with open(os.path.join(load_path, "metrics.json"), "r") as f:
        metrics = json.load(f)

    y_train_history = np.load(os.path.join(load_path, "y_train_history.npy"), allow_pickle=True)
    y_pred = np.load(os.path.join(load_path, "y_pred.npy"), allow_pickle=True)
    y_pred_prob = np.load(os.path.join(load_path, "y_pred_prob.npy"), allow_pickle=True)

    # Reconstruct DataFrame
    df = pd.DataFrame(metrics)
    # df["y_train_history"] = list(y_train_history)
    # df["y_pred"] = list(y_pred)
    # df["y_pred_prob"] = list(y_pred_prob)

    # df["y_train_history"] = df["y_train_history"].apply(lambda x: [str(i) for i in x])
    # df["y_pred"] = df["y_pred"].apply(lambda x: [str(i) for i in x])

    df["y_train_history"] = [np.array(x, dtype=str) for x in y_train_history]
    df["y_pred"] = [np.array(x, dtype=str) for x in y_pred]
    df["y_pred_prob"] = list(y_pred_prob)  # unchanged

    return df    

###historic
def load_results_json(RES_PATH):
    df = pd.read_csv(RES_PATH)
    df["y_train_history"] = df["y_train_history"].apply(json.loads)  # ✅ Convert back to list
    df["y_pred"] = df["y_pred"].apply(json.loads)
    df["y_pred_prob"] = df["y_pred_prob"].apply(json.loads)
    return df


def load_trace_m(df_temp):
    y_train_initial=np.array(df_temp["y_train_history"].iloc[0])
    y_train_last=np.array(df_temp["y_train_history"].iloc[-1])
    label_names = np.unique(y_train_initial, return_counts=False) 
    LE_trace_matrix = np.zeros((label_names.shape[0],label_names.shape[0]))
    check_dif_ = y_train_initial != y_train_last
    dif_idx = np.where(check_dif_)[0]
    for idx in dif_idx:
        LE_trace_matrix[int(y_train_initial[idx])-1, int(y_train_last[idx])-1] +=1

    return LE_trace_matrix

###historic
def ensure_json_serializable(value):
    """Convert numpy arrays or lists to JSON strings if necessary."""
    if isinstance(value, str):  # Already a JSON string, return as is
        return value
    elif isinstance(value, np.ndarray):  # Convert numpy array to list, then serialize
        return json.dumps(value.tolist())
    elif isinstance(value, list):  # Convert list to JSON if it's not already
        return json.dumps(value)
    else:
        return json.dumps([value])  # Convert single values into a list and serialize

###historic
def recursive_convert_to_serializable(obj):
    """Recursively convert ndarrays to lists inside nested structures."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [recursive_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: recursive_convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj  # base case: keep the object as is if it's not list/dict/array

###historic
def safe_json_dumps(obj):
    """Convert to JSON-safe object and dump as JSON string."""
    return json.dumps(recursive_convert_to_serializable(obj))
   

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
    
    else:
        raise ValueError(f"Unknown le_strategy: {le_strategy}. Please choose a valid strategy ('default', 'V1', 'leV1', 'V2', 'leV2').")
    #TODO ADD SAFETY MECHANISM IF one p_i of one or more classes == 0 --> remaining_classes are not selectable


def perform_label_flip(flip_trajectory, le_instance, y_train,  error_rel, error_p_incr):
    """
    Perform label flips based on the provided flip trajectory.
    This function modifies the y_train array in place and updates the error relative.
    """
    flip_row = flip_trajectory[flip_trajectory["instances"] == le_instance].iloc[0]
    flip_index = flip_row["where"]
    new_label = flip_row["to"]
    old_label = flip_row["from"]

    #Validate that flip_trajectory & y_train are congruent
    if y_train[flip_index] != old_label:
        logging.error(f"Flip mismatch at instance {le_instance}: expected label '{old_label}' at index {flip_index}, "
                      f"but found '{y_train[flip_index]}' in y_train.")
        raise ValueError(f"Label mismatch at index {flip_index}: expected '{old_label}', got '{y_train[flip_index]}'.")

    # Perform flip
    y_train[flip_index] = new_label
    error_rel += error_p_incr
    logger.info(f"changed label {old_label} to {new_label} at index {flip_index} of the data")

    return y_train, error_rel


def perform_label_flips_historic(history_df, source_dict, y_train, label_names_, label_counts_, le_params, res_path, float_p, error_rel, error_p_incr):  
    #VARIANTE1: PICK RANDOMLY ACROSS classes. Each class has an equal chance until empty.
    #VARIANTE2: CLASS HIERARCHY/HETEROGENITY: Define a p_vector with classwise probabilites of flipping UNTIL EMPTY?
    #VARIANTE2.1: MINORITY CLASS FIRST/MAJORITY CLASS FIRST: Special case of CLASS HIERARCHY  
    #VARIANTE3: PICK RANDOM LABELS. Each class has the chance according to their number of instances
    #CURRENT CHOICE: VARIANTE1
    #IMPLEMENTED: VARIANTE1, VARIANTE2 
    non_empty_classes = [cls for cls in source_dict if source_dict[cls]]
    class_count_one = (label_counts_ == 1)
    class_count_one_dict = dict(zip(label_names_, class_count_one))

    #logger.info(f"Current Source dict: {source_dict}")
    if not non_empty_classes:  # Stop early if all classes are empty
        logger.error("No more instances left to process.......return INVALID")
        return "INVALID"
    
    try:           
        selected_class = random.choices(non_empty_classes, weights=le_params[1], k=1)[0]
        if class_count_one_dict[selected_class] == True:
            logger.warning(f"Class {selected_class} has only one instance left. It will not be considered for the current label_flip_mechanism!")
            if error_rel >= 0.88:
                logger.warning(f"Error relative is already high: {error_rel}. Flipping class: {selected_class} which has only one instance left is now valid")
            else: 
                while class_count_one_dict[selected_class] == True:
                    logger.info("Trying another approach to select a non-one-instance-left class")
                    selected_class = random.choices(non_empty_classes, weights=le_params[1], k=1)[0]
                
    ###Perform a ONE TIME clearance of empy classes when loading and continuing the exp
    ### if the number of classes does not match the population (randon.choices())
    except:
        empty_classes = [cls for cls in source_dict if not source_dict[cls]]
        logger.warning(f"Some classes are empty: {empty_classes}")
        logger.info(f"Current length of non empty clasces: {len(non_empty_classes)}")
        logger.info(f"Current length of le_params: {len(le_params[1])}")
        for cls in empty_classes:
            logger.warning(f"Class {cls} is now empty and will be removed from le_params!")
            p_value_to_remove = le_params[2][cls]
            p_index_to_remove = le_params[1].index(p_value_to_remove)
            le_params[1].pop(p_index_to_remove)
        selected_class = random.choices(non_empty_classes, weights=le_params[1], k=1)[0]
        if class_count_one_dict[selected_class] == True:
            logger.warning(f"Class {selected_class} has only one instance left. It will not be considered for the current label_flip_mechanism!")
            while class_count_one_dict[selected_class] == True:
                logger.info("Trying another approach to select a non-one-instance-left class")
                selected_class = random.choices(non_empty_classes, weights=le_params[1], k=1)[0]

    remaining_labels = [label for label in label_names_ if label != selected_class]
    rlc2 = np.random.choice(remaining_labels, size=1)[0]
    removed_instance_idx = source_dict[selected_class].pop(random.randint(0, len(source_dict[selected_class]) - 1))
    if not source_dict[selected_class]:  
        logger.warning(f"Class {selected_class} is now empty and will be removed from le_params!")
        p_value_to_remove = le_params[2][selected_class]
        p_index_to_remove = le_params[1].index(p_value_to_remove)
        le_params[1].pop(p_index_to_remove)
        
        if sum(le_params[1]) == 0:
            logger.error("ERROR WARNING There are no classes left with probability > 0.  --> stop execution!")
            save_history_df_compressed(res_path, df=history_df)
            return "INVALID"

    y_train[removed_instance_idx] = rlc2
    error_rel += error_p_incr
    error_rel = round(error_rel, float_p)
    logger.info(f"changed label {selected_class} to {rlc2} at index {removed_instance_idx} of the data")
    return history_df, source_dict, y_train, le_params, error_rel



def missing_step_calculator(
    train_test_df: pd.DataFrame,
    history_df: pd.DataFrame,
    le_trajectory: pd.DataFrame,
    cl_dict: Dict[str, object],
    start: int,
    stop: int,
    step: int,
    float_prec: int,
    hist_df_column_validation: List["str"]
) -> pd.DataFrame:
    """
    Calculate and fill in missing label error steps in the experiment history.

    Parameters:
    - history_df: DataFrame containing existing experiment results.
    - le_trajectory: DataFrame describing the label flip trajectory.
    - cl_dict: Dictionary with classifier name as key and classifier object as value.
    - start: Integer, starting number of label errors (inclusive).
    - stop: Integer, ending number of label errors (inclusive).
    - step: Integer, the step size between error counts.
    - float_prec: Integer, number of decimal places to round floating point results.

    Returns:
    - A DataFrame with missing steps filled in.
    """
    requested_steps = set(range(start, stop, step))
    if stop not in requested_steps:
        requested_steps.add(stop)
    existing_steps = set(history_df["LE_instances"].tolist())
    missing_steps = sorted(list(requested_steps - existing_steps))  #sets are needed for subtraction
    logger.info(f"Missing steps to compute: {missing_steps}")
    df_length = len(history_df)
    y_train_initial = history_df["y_train_history"].iloc[0]
    cl_ = next(iter(cl_dict)) 

    # Apply label errors for each missing step
    for i_, step_ in enumerate(missing_steps):
        y_train = reconstruct_state_y_train(y_train_initial, le_trajectory, k=step_)
        logger.info(f"y_train for step_: {step_}  will be calculated")
        

        # Fit Classifier and make prediction
        train_test_df["y_train_small"] = y_train  # Update the train_test_df with the modified y_train
        res_ = apply_TSC_algos(train_test_dct=train_test_df, classifiers=cl_dict)

        error_relative = step_ / y_train_initial.shape[0]
        step_val = history_df["step"].max() + 1
        logger.info(f"current iteration: {i_} at step_val {step_val} for current LE_instances: {step_} error_relative: {error_relative}")
        new_row = pd.DataFrame([{
            "step": int(step_val),
            "LE_instances": int(step_),
            "LE_relative": round(error_relative, float_prec),
            "accuracy": round(res_[cl_]["accuracy"], float_prec),
            "train_time": res_[cl_]["train_time"],
            "eval_time": res_[cl_]["eval_time"],
            "y_train_history": y_train.copy(),
            "y_pred": res_[cl_]["y_pred"],
            "y_pred_prob": res_[cl_]["y_pred_prob"],
        }])

        missing_cols = [col for col in hist_df_column_validation if col not in history_df.columns]
        if missing_cols:
            raise ValueError(f"The following expected columns are missing from history_df: {missing_cols}")

        history_df = pd.concat([history_df, new_row], ignore_index=True)
        history_df = history_df.sort_values(by="LE_instances")#.reset_index(drop=True)


        logger.info(f"Iteration finished succesfully \n")

    history_df = history_df.astype({"step": "int32","LE_instances": "int32","LE_relative": "float64", "accuracy": "float64"})
    return history_df



def missing_step_calculator_historic(history_df, train_test_df,le_params, res_path, float_p, cl_dict, start, stop, step):
    FLOAT_PREC = float_p
    RES_PATH = res_path
    DEBUG_TRAIN_TEST = []
    existing_steps = set(history_df["LE_instances"].tolist())
    requested_steps = set(range(start, stop + 1, step))
    missing_steps = sorted(list(requested_steps - existing_steps))  
    logger.info(f"Missing steps to compute: {missing_steps}")

    new_rows = []
    all_y_trains = list(history_df["y_train_history"])
    step_to_y_train = dict(zip(history_df["LE_instances"], all_y_trains))
    y_train_initial = np.array(all_y_trains[0])
    error_perc_incr = np.round(1/len(y_train_initial), FLOAT_PREC)

    # Iterate over all missing steps
    for step_ in missing_steps: #step_ = 3
        prev_step = max([s for s in existing_steps if s <= step_], default=0)
        y_train = np.array(step_to_y_train[prev_step]) #prev y_train [1,1,1,1,1,2,2,2,2] y_train_ahead [2,2,2,1,1,2,2,2,2]
        label_names,label_counts = np.unique(y_train, return_counts=True)
        print("y_train:  ", y_train)
        print("y_train_initial:  ", y_train_initial)
        print("label_names:   ", label_names)
    
        source_dict = {cls: np.where((y_train == cls) & (y_train_initial == cls))[0].tolist() for cls in label_names}
        error_relative = np.round(1/y_train.shape[0] * step_ , FLOAT_PREC)

        # Apply exactly the number of flips needed to go from prev_step to step_
        flips_needed = step_ - prev_step   #eg   prev_step=1, step_=2, flips_needed=1
        for _ in range(flips_needed):
        # Perform label flip — reuse your flip logic here
            label_flip_res = perform_label_flips_historic(history_df = history_df,
                                source_dict=source_dict,
                                y_train=y_train,
                                label_names_ = label_names,
                                label_counts_= label_counts,
                                le_params=le_params,
                                res_path=RES_PATH,
                                float_p=FLOAT_PREC,
                                error_rel=error_relative,
                                error_p_incr=error_perc_incr)
            if label_flip_res == "INVALID":
                break
            else:
                existing_steps.add(step_)
                history_df, source_dict, y_train, le_params, error_relative = label_flip_res
                step_to_y_train[step_]=y_train


        train_test_df["y_train_small"] = y_train
        # DEBUG      DEBUG_TRAIN_TEST.append(train_test_df.copy())
        res_ = apply_TSC_algos(train_test_dct=train_test_df, classifiers=cl_dict)
        cl_ = next(iter(cl_dict)) 
        new_row = {
        "step": int(len(history_df) + len(new_rows)),
        "LE_instances": step_,
        "LE_relative": round(step_ / len(y_train), FLOAT_PREC),
        "accuracy": np.round(res_[cl_]["accuracy"], FLOAT_PREC),
        "y_train_history": y_train.copy(),
        "y_pred": res_[cl_]["y_pred"],
        "y_pred_prob": res_[cl_]["y_pred_prob"]
        }
        new_rows.append(new_row)
        logger.info("current iteration: {}   current LE_step: {} error_relative: {}".format(len(new_rows), step_, error_relative))

    # Merge old and new dataframes
    new_df = pd.DataFrame(new_rows)
    history_df = pd.concat([history_df, new_df], ignore_index=True)
    history_df = history_df.sort_values("LE_instances").reset_index(drop=True)
    return history_df                   #, DEBUG_TRAIN_TEST


def percentage_to_instance_converter(doe_param, train_test_df):
    """Convert percentage-based DOE parameters to instance-based parameters.
       This function ensures that the step size is a valid integer and does not exceed the number of instances.
       It also ensures that the stop value is less than or equal to 99% of the total instances.
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


def apply_label_errors(train_test_df, cl_dict, ds_="ds_0", doe_param=None, exp_folder=None):
    """train_test_df should be a pd.DataFrame with the columns X_train, X_test, y_train, y_test
       and their reduced identity (X_train_small). cl_dict should be a dict out of classifier names
       and their respective instances.
       RETURNS: history_df      ---> with column structure:
                                    [step || LE_instances || LE_relative || accuracy || y_train_hist || y_pred || y_pred_prob]
                LE_trace_matrix ---> np.array (dim=2, dtype=int) with label flip history
       STORES: result files in EXP_PATH
                1.: algorithm data,parameters,train&prediction time                             Konfigurationsfile (yaml)
                2.: data parameters,current_dataset, current split, random_seed, etc...?                           (yaml)
                3.: performance metrics acc, bal_acc, NLL, AUROC, F1Sc                      Aufbereitung (visualize(csv))
                4: y_train_hist, y_pred, y_pred_proba, ...                                              Results_file(csv)   
                EXP_PATH: os.path.join(RESULT_DIRECTORY/cl_/ds_/filename_)))
                where filename_ consists of: ds_restype_randomS_start_stop_step          
    """
    #CONSTANTS
    RANDOM_S = 0 #seed for classifier architecture & LE_trajectory
    FLOAT_PREC=6
    METRICS= "metrics.json"
    Y_TRAIN_HIST= "y_train_history.npz"  ###--> historic
    Y_PRED= "y_pred.npz"
    Y_PRED_PROB="y_pred_prob.npz"
    PARAM_MODE="percentage"
    RESULT_DIRECTORY = "simulation_results/"
    LE_TRAJECTORY_DIR = "simulation_results/label_errors/" 
    HIST_DF_COLS= ["step", "LE_instances", "LE_relative", "accuracy", "train_time", "eval_time",
                   "y_train_history", "y_pred","y_pred_prob"]
    DEBUG = False
    GPU_USAGE = True

    #DOE_PARAMS
    if doe_param is None:
        doe_param = {"le_strategy":"leV1","p_vec":None, "random_seed":0,"start":0,"stop":26,"step":1}
    if PARAM_MODE == "percentage":
        doe_param = percentage_to_instance_converter(doe_param, train_test_df)
    if GPU_USAGE == True:
        assign_GPU()
    le_strategy=doe_param["le_strategy"]
    p_vector=doe_param["p_vec"]
    random_seed=doe_param["random_seed"]
    start=doe_param["start"]
    stop=doe_param["stop"]
    step=doe_param["step"]
    if start + step >= stop:
        logger.error("Not Enough range to perform ONE Flip. Adjust parameters!")
        raise ValueError(f"Not Enough range to perform ONE Flip. Start: {start} , stop: {stop}")
    np.random.seed(random_seed)

    #__init__ LE_PARAMS, CLASSIFIER, DATASET, DIRECTORY----
    le_params = init_le_params(le_strategy, p_vector, train_test_df) # returns le_string and p_vector
    cl_ = next(iter(cl_dict))                   #get the name of the cl_ instance
    dataset_name = ds_                          
    directory_extension = le_params[0]+ "_" + str(random_seed)+ "_" + str(start)+ "_" + str(stop)+ "_" + str(step)
    if exp_folder is not None:
        RESULT_DIRECTORY = exp_folder    
    EXP_PATH = os.path.join(RESULT_DIRECTORY,cl_, dataset_name) #directory with RES_FOLDERS
    RES_PATH = os.path.join(EXP_PATH, directory_extension)      #RES_FOLDER
    LE_TRAJECTORY_PATH = os.path.join(LE_TRAJECTORY_DIR, dataset_name, le_params[0])
    os.makedirs(EXP_PATH, exist_ok=True)

    #CHECK FOR RESULTS. Case logic is heaviely defined in check_for_results()
    existing_results= check_for_results(target_directory=EXP_PATH , filename_list=[METRICS,Y_TRAIN_HIST,Y_PRED,Y_PRED_PROB],
                                        leV=le_params[0], randomS=random_seed, start=start, stop=stop, step=step) 
    
    le_trajectory = check_for_le_trajectory(le_trajectory_dir=LE_TRAJECTORY_PATH,
                                            leV=le_params[0],
                                            dataset=dataset_name,
                                            train_arr=train_test_df["y_train_small"],
                                            p_vector=le_params[1],
                                            random_seed=RANDOM_S,
                                            )    

    #C1 Results are already there (complete)
    if existing_results["status"]=="exact_match":
        history_df = load_history_df_safely(RES_PATH, le_trajectory=le_trajectory, initial_y_train=train_test_df["y_train_small"])
        LE_trace_matrix = load_trace_m(df_temp=history_df)
        return history_df, LE_trace_matrix
    
    #C1.1 Results are already FURTHER calculated then requested and have to be trimmed
    if existing_results["status"]=="load_and_trim":
        historic_path = existing_results["load_existing"]
        history_df = load_history_df_safely(historic_path, le_trajectory=le_trajectory, initial_y_train=train_test_df["y_train_small"])
        trimmed_df = history_df[history_df["LE_instances"] <= stop]
        LE_trace_matrix = load_trace_m(df_temp=trimmed_df)
        return trimmed_df, LE_trace_matrix
    
    #C2 Results are partialy calculated and have to be loaded and continued
    if existing_results["status"]=="load_and_continue":
        historic_path = existing_results["load_existing"]
        history_df = load_history_df_safely(historic_path, le_trajectory=le_trajectory, initial_y_train=train_test_df["y_train_small"])
        latest_stop = existing_results["latest_stop"]
        trimmed_df = history_df[history_df["LE_instances"] <= latest_stop]  ###NEWLINE with TRIMMING
        logger.info("Trimmed history_df to the latest stop: %s", latest_stop)
        history_df = trimmed_df                 #OVERWRITE history df to ensure correct continuation
        y_train = np.array(trimmed_df["y_train_history"].iloc[-1])
        train_test_df["y_train_small"]=y_train #traing_test_df points on the correct object
        y_train_initial=np.array(trimmed_df["y_train_history"][0])
  

        label_names, label_counts = np.unique(y_train_initial, return_counts=True) 
        LE_trace_matrix = load_trace_m(df_temp=history_df)
        start = latest_stop + step
        if start >= stop:
            logger.error("Not Enough range to perform ONE Flip. Adjust parameters!")
            raise ValueError(f"Not Enough range to perform ONE Flip. Start: {start} , stop: {stop}. Please adjust your parameters.")
        error_relative = trimmed_df["LE_relative"].iloc[-1]

    #C3 Results are to coarse
    #C3.1 Results are fully calculated BUT are to coarse |--> finer
    #C3.2 Results are partialy calculated AND are to coarse |--> finer & further
    if existing_results["status"]=="load_and_fill_between_and_continue":
        existing_results["status"]= "load_and_trim_and_fill_between"
    if existing_results["status"]=="load_and_trim_and_fill_between":
        historic_path = existing_results["load_existing"]
        history_df = load_history_df_safely(historic_path, le_trajectory=le_trajectory, initial_y_train=train_test_df["y_train_small"])
        history_df = missing_step_calculator(train_test_df=train_test_df,
                                            history_df=history_df,
                                            le_trajectory=le_trajectory,
                                            cl_dict=cl_dict,
                                            start=start,
                                            stop=stop,
                                            step=step,
                                            float_prec=FLOAT_PREC,
                                            hist_df_column_validation=HIST_DF_COLS)

        LE_trace_matrix = load_trace_m(df_temp=history_df)
        save_history_df_compressed(RES_PATH, df=history_df)
        return history_df, LE_trace_matrix


    #C4 No results present. Start experiment from scratch
    if existing_results["status"] =="no_results_present":
        y_train, y_test = train_test_df["y_train_small"], train_test_df["y_test_small"]  #---> UNNECESSARY Y TEST
        #Initialize resulting DataFrame
        history_df = pd.DataFrame(columns= HIST_DF_COLS)
        error_relative = 0
        #Initialize all present label names & their respective number, Initialize Trace Matrix for label tracing
        #Provide subscriptable Source_dict to ensure correct non-duplicate label flipping
        label_names, label_counts = np.unique(y_train, return_counts=True) 
        LE_trace_matrix = np.zeros((label_names.shape[0],label_names.shape[0]))


    #Case-Fusion. FOR ALL cases in which there are no results or no far enough results execute the Main-Loop 
    #START EXPERIMENT
    error_start = start
    error_increasement = step
    #error_perc_incr = np.round(1/y_train.shape[0], FLOAT_PREC)
    error_perc_incr = 1/ y_train.shape[0]
    error_stop = stop
    add_row = True


    #for every step between start and stop, intrude a label flip and fit the cl_
    instance_steps_to_iterate = list(range(error_start, error_stop, error_increasement))
    if instance_steps_to_iterate[-1] != error_stop:
        instance_steps_to_iterate.append(error_stop)
    
    last_increasement = instance_steps_to_iterate[-1] - instance_steps_to_iterate[-2]  # last step increasement

    for i_, step_ in enumerate(instance_steps_to_iterate):
        if step_ >= 1 and add_row == True:
            if step_ == error_stop:
                error_increasement = last_increasement
            for e_i_ in range(step_ - error_increasement + 1, step_ + 1, 1):
                # step = 10, error_increasement = 5 -> e_i_ = 6, 7, 8, 9, 10
                # step = 12, last_increasement = 2 -> e_i_ = 11, 12
                label_flip_res = perform_label_flip(flip_trajectory=le_trajectory,
                                                    le_instance=e_i_,
                                                    y_train=y_train,
                                                    error_rel=error_relative,
                                                    error_p_incr=error_perc_incr)
                
                if label_flip_res == "INVALID":
                    logger.error("Label flip resulted in an invalid state. Stopping execution.")
                    add_row = False
                    break
                else:
                    y_train, error_relative = label_flip_res


            label_names, label_counts = np.unique(y_train, return_counts=True)
            logger.info("current class balance distribution: %s" ,dict(zip(label_names, label_counts)))
                    

        # Fit Classifier and make prediction
        train_test_df["y_train_small"] = y_train  # Update the train_test_df with the modified y_train
        res_ = apply_TSC_algos(train_test_dct=train_test_df, classifiers=cl_dict)

        step_val = 1 if history_df.empty else history_df["step"].max() + 1
        logger.info(f"current iteration: {i_} at step_val {step_val} for current LE_instances: {step_} error_relative: {error_relative}")
        # Prepare new row for history_df        
        if add_row:
            new_row = pd.DataFrame([{
            "step": int(step_val),
            "LE_instances": int(step_),
            "LE_relative": error_relative,
            "accuracy": np.round(res_[cl_]["accuracy"], FLOAT_PREC),
            "train_time": res_[cl_]["train_time"],
            "eval_time": res_[cl_]["eval_time"],
            "y_train_history": y_train.copy(),
            "y_pred": res_[cl_]["y_pred"],
            "y_pred_prob": res_[cl_]["y_pred_prob"]
        }])

            history_df = pd.concat([history_df, new_row], ignore_index=True)

            #history_df.loc[i_+ len_df] = [int(i_+len_df), int(step_), error_relative, np.round(res_[cl_]["accuracy"],FLOAT_PREC),
             #                     res_[cl_]["train_time"], res_[cl_]["eval_time"],
              #                  y_train.copy() ,res_[cl_]["y_pred"], res_[cl_]["y_pred_prob"]]
    
        logger.info(f"Iteration finished succesfully \n")

    history_df = history_df.astype({"step": "int32","LE_instances": "int32","LE_relative": "float64", "accuracy": "float64"})
    history_df = history_df.sort_values(by="LE_instances")#.reset_index(drop=True)
    history_df_to_store = history_df.copy()
    save_history_df_compressed(RES_PATH, df=history_df_to_store)
    

    return history_df, LE_trace_matrix
    
