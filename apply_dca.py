import os
import re
import random
import numpy as np
import pandas as pd
import json
import logging
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from basic_func import apply_TSC_algos
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def check_for_results(target_directory, filename_list,leV, randomS, start, stop, step):
    """ Check if currenct_classifer/current_dataset directory exists in current_directory
        Check if DesignOfExperiment parameter are already present
        ---> Justify equality for: RandomS;Start;Stop;Step
        - Exact match (randomS, start, stop, step). (Case1)
        - Partial match (same randomS, different stop or step) (Case2)
          |-- (historic_stop >= stop) & (historic_step <= step) --> load_historic_res --> continue exp (Case2.1)
                                     |-- (historic_step > step) --> run whole exp from start to stop (OR run just the missing ones?..) (Case2.2)
          |-- (historic_stop < stop) & (historic_step <= step) --> load_historic_res UNTIL the current stop. Cut the rest. (Case2.1)
                                     |-- (historic_step > step) --> run the whole exp from start to stop (Case2.3)
    """
    historic_steps = []
    partial_matches = []
    pattern = re.compile(rf"{leV}_{randomS}_(\d+)_(\d+)_(\d+)")
    print("searching for {} in {}".format(filename_list, target_directory))

    # Get all existing files in the target directory
    try:
        existing_files = os.listdir(target_directory)
        print("Potential Files in the current directory: ", existing_files)
    except FileNotFoundError:
        print("❌ Directory does not exist. No previous experiments found.")
        return False
    
    k=0
    for file_ in existing_files:
        full_path = os.path.join(target_directory, file_)

        # Check if filename matches the pattern
        match = pattern.search(file_)
    
        if match:
            file_start, file_stop, file_step = map(int, match.groups())
            k += 1

            # ✅ Case 1: Exact match found
            if file_start == start and file_stop == stop and file_step == step:
                print(f"✅ Exact match found: {file_}")
                return {
                    "status":"exact_match",
                    "load_existing": full_path,
                    "latest_stop": stop
                }

            # ✅ Case 2: Partial match (same start but different stop or smaller step (or both))
            if file_start == start and file_step <= step:  
                partial_matches.append((full_path, file_stop, file_step))
                print("Partial Match found: ", partial_matches)
            
            if file_step > step:
                historic_steps.append(file_step)

    # Print out the historic_steps for clarity
    if len(historic_steps) == k != 0:
        print("All historic Steps are to coarse. Historic steps: {}".format(historic_steps))
        print("Cannot load any file which meets the requested stepsize of:{} ".format(step))


    # CASE 2: Process Partial Matches 
    # This point will only be reached if historic_step is finer or equal to the required step
    if partial_matches:
        # Find the closest valid stop that allows continuation
        closest_file = None
        closest_stop = None
        closest_step = None

        for file_, hist_stop, hist_step in partial_matches:
            if hist_stop < stop:  # Case2.1 experiment can be continued from historic_stop
                if closest_stop is None or hist_stop > closest_stop:  # Get latest valid historic_stop
                    closest_file, closest_stop, closest_step = file_, hist_stop, hist_step
                    print("update closest_file")
            else: #Case2.2 experiment was executed further. Load and Trim
                closest_file, closest_stop, closest_step = file_, hist_stop, hist_step
                print("closest stop is higher than requested. Loaded trimmed historic file")
                return {
                    "status":"load_and_trim",    
                    "load_existing": closest_file,
                    "latest_stop": closest_stop,
                }

        if closest_file:  # Case2.1 If we found a valid partial match to continue
            print(f"Continuing from {closest_stop} to {stop} with step {step}.")
            return {
                "status":"load_and_continue",
                "load_existing":closest_file,
                "latest_stop": closest_stop,
            }

    #CASE3 Neither Exact Match nor partial Match found
    else:
        print("results are not present with the current experiment parameters")
        print("There is [1] no matching labelerror Version and [2] no matching randomSeed or [3] no experiment at all")
        return {"status":"no_results_present"}  #If any file with DesignOfExperiment Parameters is not already present


def load_results(RES_PATH):
    df = pd.read_csv(RES_PATH)
    df["y_train_history"] = df["y_train_history"].apply(json.loads)  # ✅ Convert back to list
    df["y_pred"] = df["y_pred"].apply(json.loads)
    df["y_pred_prob"] = df["y_pred_prob"].apply(json.loads)
    return df


def load_trace_m(df_temp):
    y_train_initial=np.array(df_temp["y_train_history"][0])
    y_train_last=np.array(df_temp["y_train_history"].iloc[-1])
    label_names = np.unique(y_train_initial, return_counts=False) 
    LE_trace_matrix = np.zeros((label_names.shape[0],label_names.shape[0]))
    check_dif_ = y_train_initial != y_train_last
    dif_idx = np.where(check_dif_)[0]
    for idx in dif_idx:
        LE_trace_matrix[int(y_train_initial[idx])-1, int(y_train_last[idx])-1] +=1

    return LE_trace_matrix

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


def safe_json_dumps(obj):
    """Convert to JSON-safe object and dump as JSON string."""
    return json.dumps(recursive_convert_to_serializable(obj))
   

def init_le_params(le_strategy, p_vector, train_test_df):
    label_names = np.unique(train_test_df["y_train_small"], return_counts=False)
    print("label_names: ", label_names) 
    if le_strategy in ["default", "V1", "leV1"]:
        p_vector_temp = [np.round(1/label_names.size, 4) for label in label_names]
        p_vector_dict = {label:np.round(1/label_names.size, 4) for label in label_names }
        print("Current Label Errors Strategy: DEFAULT: leV1")
        print("The p_vector for the current_experiment: "+ str(p_vector_temp))
        return "leV1", p_vector_temp, p_vector_dict
    elif le_strategy == "V2" or le_strategy =="leV2":
        if p_vector is None:
            raise ValueError("p_vector is not provided. If you want to use LE strategy V2 ensure your p_vector is valid")
        elif len(p_vector) != label_names.size:
            raise ValueError("p_vector does not match the number of classes for the current dataset choice")
        p_vector_dict = {label:np.round(p_value, 4) for label, p_value in zip(label_names, p_vector)}
        print("The p_vector for the current_experiment: "+ str(p_vector_dict))
        return "leV2", p_vector, p_vector_dict
    
    else:
        raise ValueError(f"Unknown le_strategy: {le_strategy}. Please choose a valid strategy ('default', 'V1', 'leV1', 'V2', 'leV2').")
    #TODO ADD SAFETY MECHANISM IF one p_i of one or more classes == 0 --> remaining_classes are not selectable


def apply_label_errors(train_test_df, cl_dict, ds_="ds_0", doe_param= {"le_strategy":"leV1","random_seed":0,"start":0,"stop":26,"step":1},
                        exp_folder=None, p_vector=None):
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
                EXP_PATH: os.path.join(directory_current/cl_/ds_/filename_)))
                where filename_ consists of: ds_restype_randomS_start_stop_step          
    """
    RANDOM_S = 0
    FLOAT_PREC=4
    STOP_PERC=0.9

    le_strategy=doe_param["le_strategy"]
    random_seed=doe_param["random_seed"]
    start=doe_param["start"]
    stop=doe_param["stop"]
    step=doe_param["step"]

    le_params = init_le_params(le_strategy, p_vector, train_test_df) # returns le_string and p_vector
    cl_ = next(iter(cl_dict))                   #get the name of the cl_ instance
    dataset_name = ds_                          
    directory_current = "simulation_results/"
    filename_ = le_params[0]+ "_" + str(random_seed)+ "_" + str(start)+ "_" + str(stop)+ "_" + str(step) +".csv"
    filename_res = ds_ + "_res_" + filename_
    if exp_folder is not None:
        directory_current = exp_folder
    EXP_PATH = os.path.join(directory_current,cl_, dataset_name)
    RES_PATH = os.path.join(EXP_PATH, filename_res)
    os.makedirs(EXP_PATH, exist_ok=True)

    #CHECK FOR RESULTS
    existing_results= check_for_results(target_directory=EXP_PATH , filename_list=[filename_res], leV=le_params[0],
                                        randomS=random_seed, start=start, stop=stop, step=step) 
    
    #C1 Results are already there (complete)
    if existing_results["status"]=="exact_match":
        history_df = load_results(RES_PATH)
        LE_trace_matrix = load_trace_m(df_temp=history_df)
        return history_df, LE_trace_matrix
    
    #C1.1 Results are already FURTHER calculated then requested and have to be trimmed
    if existing_results["status"]=="load_and_trim":
        historic_file = existing_results["load_existing"]
        history_df = load_results(historic_file)
        trimmed_df = history_df[history_df["LE_instances"] <= stop]
        LE_trace_matrix = load_trace_m(df_temp=trimmed_df)
        return trimmed_df, LE_trace_matrix
    
    #C2 Results are partialy calculated and have to be loaded and continued
    if existing_results["status"]=="load_and_continue":
        historic_file = existing_results["load_existing"]
        history_df = load_results(historic_file)
        latest_stop = existing_results["latest_stop"]
        y_train = np.array(history_df["y_train_history"].iloc[-1])
        train_test_df["y_train_small"]=y_train #traing_test_df points on the correct object
        y_train_initial=np.array(history_df["y_train_history"][0])
        label_names, label_counts = np.unique(y_train_initial, return_counts=True) 
        source_dict = {cls: np.where((y_train == cls) & (y_train_initial == cls))[0].tolist() for cls in label_names}
        print(source_dict)
        LE_trace_matrix = load_trace_m(df_temp=history_df)
        start = latest_stop + step
        error_relative = history_df["LE_relative"].iloc[-1]

    #C3 No results present. Start experiment from scratch
    if existing_results["status"] =="no_results_present":
        y_train, y_test = train_test_df["y_train_small"], train_test_df["y_test_small"]  #---> UNNECESSARY Y TEST
        #Initialize resulting DataFrame
        history_df = pd.DataFrame(columns=["step", "LE_instances", "LE_relative", "accuracy", "y_train_history", "y_pred","y_pred_prob"])
        error_relative = 0
        #Initialize all present label names & their respective number, Initialize Trace Matrix for label tracing
        #Provide subscriptable Source_dict to ensure correct non-duplicate label flipping
        label_names, label_counts = np.unique(y_train, return_counts=True) 
        LE_trace_matrix = np.zeros((label_names.shape[0],label_names.shape[0]))
        source_dict = {cls: np.where(y_train == cls)[0].tolist() for cls in label_names}


    #Case-Fusion. After all Cases are handled appropriatly, execute the Main-Loop
    #START EXPERIMENT
    np.random.seed(random_seed)
    error_start = start
    error_increasement = step
    error_perc_incr = np.round(1/y_train.shape[0], FLOAT_PREC)
    error_stop = stop
    error_stop_perc = STOP_PERC
    add_row = True

    history_df = history_df.astype({"step": "int64","LE_instances": "int64","LE_relative": "float64", "accuracy": "float64"})
    history_df_json = history_df.copy(deep=True)   #Initialize JSON seriesable DF variant 
    # Convert existing non-JSON columns into JSON before appending new results
    history_df_json["y_train_history"] = history_df_json["y_train_history"].apply(ensure_json_serializable)
    history_df_json["y_pred"] = history_df_json["y_pred"].apply(ensure_json_serializable)
    history_df_json["y_pred_prob"] = history_df_json["y_pred_prob"].apply(ensure_json_serializable)

    #for every step between start and stop, intrude a label flip and fit the cl_
    iteration_start = history_df.shape[0]
    for i_, step_ in enumerate(range(error_start, error_stop + 1, error_increasement),start=iteration_start):
        if step_ >= 1 and add_row == True:
            for e_i_ in range(1, error_increasement + 1 , 1):    
                if error_relative >= error_stop_perc: ###---break criteria error_relative
                    print(str(error_relative) +"error threshold exceeds the current limit of {0} stop at iteration {1} ".format(error_stop_perc, step_))
                    add_row=False
                    break
                else:   
                    #PERFORM LABEL FLIP
                    #VARIANTE1: PICK RANDOMLY ACROSS classes. Each class has an equal chance until empty.
                    #VARIANTE2: CLASS HIERARCHY/HETEROGENITY: Define a p_vector with classwise probabilites of flipping UNTIL EMPTY?
                    #VARIANTE2.1: MINORITY CLASS FIRST/MAJORITY CLASS FIRST: Special case of CLASS HIERARCHY  
                    #VARIANTE3: PICK RANDOM LABELS. Each class has the chance according to their number of instances
                    #CURRENT CHOICE: VARIANTE1
                    #IMPLEMENTED: VARIANTE1, VARIANTE2 
                    non_empty_classes = [cls for cls in source_dict if source_dict[cls]]
                    
                    if not non_empty_classes:  # Stop early if all classes are empty
                        print("No more instances left to process.")
                        break

                    
                    #selected_class, le_params = try_class_selection()       ‚ 
                    try:           
                        selected_class = random.choices(non_empty_classes, weights=le_params[1], k=1)[0]

                    ###Perform a ONE TIME clearance of empy classes when loading and continuing the exp
                    ### if the number of classes does not match the population (randon.choices())
                    except:
                        empty_classes = [cls for cls in source_dict if not source_dict[cls]]
                        for cls in empty_classes:
                            print(f"Class {cls} is now empty and will be removed from le_params!")
                            p_value_to_remove = le_params[2][cls]
                            p_index_to_remove = le_params[1].index(p_value_to_remove)
                            le_params[1].pop(p_index_to_remove)
                        selected_class = random.choices(non_empty_classes, weights=le_params[1], k=1)[0]

                    remaining_labels = [label for label in label_names if label != selected_class]
                    rlc2 = np.random.choice(remaining_labels, size=1)[0]
                    removed_instance_idx = source_dict[selected_class].pop(random.randint(0, len(source_dict[selected_class]) - 1))
                    if not source_dict[selected_class]:  
                        print(f"Class {selected_class} is now empty and will be removed from le_params!")
                        p_value_to_remove = le_params[2][selected_class]
                        p_index_to_remove = le_params[1].index(p_value_to_remove)
                        le_params[1].pop(p_index_to_remove)
                        
                        if sum(le_params[1]) == 0:
                            print("WARNING There are no classes left with probability > 0.  --> stop execution!")
                            history_df_json.to_csv(RES_PATH, index=False)
                            break
                    y_train[removed_instance_idx] = rlc2

                    LE_trace_matrix[np.where(label_names == selected_class)[0][0], np.where(label_names== rlc2)[0][0]] +=1
                    error_relative += error_perc_incr
                    error_relative = round(error_relative, FLOAT_PREC)
                    label_names, label_counts = np.unique(y_train, return_counts=True)
                    print(f"changed label {selected_class} to {rlc2} at index {removed_instance_idx} of the data")
                #print(each substep for error increment)           
            print("current class balance distribution: {}".format(dict(zip(label_names, label_counts))))


        #fit classifier and make prediction  
        res_ = apply_TSC_algos(train_test_dct=train_test_df, classifiers=cl_dict)
        print("current iteration: {}   current LE_step: {} error_relative: {}".format(i_, step_, error_relative))
        if add_row:
            history_df.loc[i_] = [int(i_), int(step_), error_relative, np.round(res_[cl_]["accuracy"],FLOAT_PREC),
                                    y_train.copy() ,res_[cl_]["y_pred"], res_[cl_]["y_pred_prob"]]
            history_df_json.loc[i_] = [int(i_), int(step_), error_relative, np.round(res_[cl_]["accuracy"],FLOAT_PREC),
                                    json.dumps(y_train.copy().tolist()),json.dumps(res_[cl_]["y_pred"].tolist()),
                                    json.dumps(res_[cl_]["y_pred_prob"].tolist())]
            # store lists as JSON instead of raw strings to improves data integrity for several storing & loading options.
            # res should be stored in /David_MA/dca/label_er/<current_cl>/<current_ds>/results.csv # maybe add conf_matrices. delta?
    history_df = history_df.astype({"step": "int64","LE_instances": "int64","LE_relative": "float64", "accuracy": "float64"})
    history_df_json.to_csv(RES_PATH, index=False)

    return history_df, LE_trace_matrix
    

def visualize_acc_decr(df_acc_inst_rel, w_=6, h_=4, dpi_=150, first="instances", second="relative",
                      cl_="cl_0", ds_="ds_0", filename_="acc_decr", save_fig=False, exp_folder=None):
    """
    VISUALIZE accuracy decrease of one SINGLE dataset/algorithm/DCA -combination.
    RECEIVE: df_acc_inst_rel: DataFrame with the column structure accuracy;LE_instances;LE_relative
             w_, h_, dpi: weight, height and dpi of the figure
             first: first axis type of the visualization. Options: [instances], [relative]
             second: second axis type of the visualization
    RETURNS: Nothing
    STORES: IF save_fig == True: stores figure in exp_path (directory_current/cl_/ds_+filename)
    """
    acc_decr=df_acc_inst_rel["accuracy"]
    LE_instances=df_acc_inst_rel["LE_instances"]
    LE_relative=df_acc_inst_rel["LE_relative"]
    colors = ['tab:blue', 'tab:orange']
    directory_current = "simulation_results/"
    dataset_name = ds_
    if exp_folder is not None:
        directory_current = exp_folder
    EXP_PATH = os.path.join(directory_current,cl_, dataset_name, (dataset_name + "_" + filename_))

    fig, ax1 = plt.subplots(figsize=(w_, h_), dpi=dpi_)
    fig.suptitle('Impact of Label Errors on Model Accuracy', fontsize=12, fontweight='bold')

    # Plot the number of instances (left y-axis)
    if first == "relative" :
        x_ = LE_relative
        x_label = "Label Errors (Relative)"

    else:
        x_ = LE_instances
        x_label = "Label Errors (Instances)"


    #Initialize first plot
    ax1.set_ylabel('Accuracy' )
    ax1.set_xlabel(x_label)
    ax1.set_xlim(x_.min(), x_.max()+0.02*x_.max())
    ax1.plot(x_ ,acc_decr, color=colors[1], label=cl_)
    ax1.tick_params(axis='y') #labelcolor=colors[0])
    ax1.grid(visible=True, linestyle='--', alpha=0.6, linewidth=0.5)

    def inst2rel(x):
        return df_acc_inst_rel.loc[(df_acc_inst_rel["LE_instances"]==x),"LE_relative"].iloc[0]
    def rel2inst(x):
        return df_acc_inst_rel.loc[(df_acc_inst_rel["LE_relative"]==x),"LE_instances"].iloc[0]

    ###CHECK for second axis
    if not second == None:
        print("CURRENTLY IN DEV MODE")
        if second == "relative":
            x2_ = LE_relative
            x_label = "Label Errors (Relative)"

        elif second =="instances":
            x2_ = LE_instances
            x_label = "Label Errors (Instances)"

        #ax2 = ax1.twiny()
        ax2 = ax1.secondary_xaxis('top', functions=(inst2rel, rel2inst))
        #ax2.plot(x2_, acc_decr, color=colors[0], label=cl_)
        #ax2.tick_params(axis="x")
        ax2.set_xlabel(x_label)

       

    ### Independent of a second axis should be present: 
    # Finalize Figure aesthetics and saveplot
    ax1.legend(loc='upper right')
    fig.tight_layout()
    if save_fig==True:
        os.makedirs(os.path.join(directory_current,cl_, dataset_name), exist_ok=True)
        fig.savefig(fname=(EXP_PATH))


    plt.show()


# Compute row sums
def visualize_trace_M(trace_M, cl_="cl_0", ds_="ds_0",dpi=200, filename_="trace_M", save_fig=False, exp_folder=None):
    """
    VISUALIZE trace Matrix of one SINGLE dataset/algorithm/DCA -combination.
    RECEIVE: trace_Matrix: DataFrame with the column structure: Original label, New label
    RETURNS: Nothing
    STORES: IF save_fig == True: stores figure in exp_path (directory_current/cl_/ds_/filename_+"_"+paramStr)
            paramStr = randomS_+start_+stop_+step_
    """

        
    colors = ['tab:blue', 'tab:orange'] #TODO
    directory_current = "simulation_results/"
    if exp_folder is not None:
        directory_current = exp_folder
    EXP_PATH = os.path.join(directory_current,cl_, (ds_+ "_"+ filename_))


    row_sums = trace_M.sum(axis=1)

    # Create figure with GridSpec
    fig = plt.figure(figsize=(5.5, 3), dpi=200)
    gs = gridspec.GridSpec(1, 2, width_ratios=[14,1], wspace=-0.4)  # Adjust width_ratios for heatmap and side plot

    # Create heatmap
    ax0 = plt.subplot(gs[0])
    sns.heatmap(trace_M, annot=True, linewidths=0, square=True,cbar=False, cmap='Oranges', ax=ax0)
    ax0.set_ylabel('Original Label')
    ax0.set_xlabel('New Label')
    ax0.set_title("Label_Flip_Trace_"+ds_+ "_"+ cl_, size=9)

    # Create side bar aggregation plot of manipulated original labels 
    ax1 = plt.subplot(gs[1], sharey=ax0)  # Share y-axis with heatmap
    bar_container = ax1.barh(np.arange(len(row_sums)), row_sums, height=1, color='darkorange', align="edge", alpha=0.6, edgecolor="darkorange")  # Horizontal bar chart
    for bar, value in zip(bar_container, row_sums):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{value:.0f}', 
                va='center', ha='left', fontsize=8)
    #ax1.set_xticks([])  # Hide x-axis ticks
    #ax1.set_yticks([])  # Hide y-axis labels for clarity
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['right'].set_visible(False)
    ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0)

    ax1.tick_params(axis="y", which="both", length=0, labelsize=0, color="white", grid_color="white")
    ax1.set_ylabel(r'$\sum$', rotation=0, fontsize=10, labelpad=5, va='center')
    fig.tight_layout()
    plt.show()


