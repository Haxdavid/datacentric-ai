import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from basic_func import apply_TSC_algos



def check_for_results(filename_list):
    """ Check if currenct_classifer/current_dataset directory exists in current_directory
        Check if DesignOfExperiment parameter are already present
        ---> Justify equality for: RandomS;Start;Stop;Step """
     
    exist_all = []
    for file_ in filename_list:
        if os.path.exists(file_):
            print(f"File exists: {file_}")
            exist_all.append(True)
        else:
            print(f"File NOT found: {file_}")
            exist_all.append(False)
    if all(exist_all): 
        return True
    else:
        print("results are not present with the current experiment parameters")
        return False #If any file with DesignOfExperiment Parameters is not already present
   
def load_results(RES_PATH):
    df = pd.read_csv(RES_PATH)
    df["y_train_history"] = df["y_train_history"].apply(json.loads)  # ✅ Convert back to list
    df["y_pred"] = df["y_pred"].apply(json.loads)
    df["y_pred_prob"] = df["y_pred_prob"].apply(json.loads)
    return df

def load_trace_m(df):
    trace_m = df
    return trace_m


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
   

   
 
def apply_label_errors(train_test_df, cl_dict, ds_="ds_0", start=0, stop=10, step=1, stop_percentage=0.25, float_prec=4, exp_folder=None):
    """train_test_df should be a pd.DataFrame with the columns X_train, X_test, y_train, y_test
       and their reduced identity (X_train_small). cl_dict should be a dict out of classifier names
       and their respective instances.
       RETURNS: history_df      ---> with column structure: [step || LE_instances || LE_relative || accuracy]
                res_            ---> pred_dict with : [acc, y_pred, y_pred_prob]
                LE_trace_matrix ---> np.array (dim=2, dtype=int) with label flip history
       STORES: result files in EXP_PATH
                1.: algorithm data,parameters,train&prediction time                         Konfigurationsfile (yaml)
                2.: data parameters,current_dataset, current split, random_seed, etc...?                       (yaml)
                3.: performance metrics acc, bal_acc, NLL, AUROC, F1Sc                      Aufbereitung (visualize(csv))
                4: y, y_pred, y_pred_proba, ...                                             Ergebnisse (csv)   
                EXP_PATH: os.path.join(directory_current/cl_/ds_/filename_)))
                where filename_ consists of: ds_restype_randomS_start_stop_step          
       """
    RANDOM_S = 0
    cl_ = next(iter(cl_dict))
    dataset_name = ds_
    directory_current = "simulation_results/"
    randomS=0
    filename_ = str(randomS)+ "_" + str(start)+ "_" + str(stop)+ "_" + str(step) +".csv"
    filename_res = ds_ + "_res_" + filename_
    filename_pred = ds_ + "_pred_" + filename_
    filename_trace = ds_ + "_trace_" + filename_
    if exp_folder is not None:
        directory_current = exp_folder
    EXP_PATH = os.path.join(directory_current,cl_, dataset_name)
    RES_PATH = os.path.join(EXP_PATH, filename_res)
    PRED_PATH = os.path.join(EXP_PATH, filename_pred)
    TRACE_PATH = os.path.join(EXP_PATH, filename_trace)
    os.makedirs(EXP_PATH, exist_ok=True)


    #TODO Add LE-Tracing (Which class flips to which class how often, class^2 relations)
    if not check_for_results(filename_list=[RES_PATH, PRED_PATH, TRACE_PATH]):
        X_train, y_train = train_test_df["X_train_small"], train_test_df["y_train_small"] #---> UNNECESSSARY ecept y_train
        X_test, y_test = train_test_df["X_test_small"], train_test_df["y_test_small"]  #---> UNNECESSARY
        error_start = start
        error_increasement = step
        error_perc_incr = np.round(1/y_train.shape[0],float_prec)
        error_stop = stop
        error_stop_perc = stop_percentage
        add_row = True
        black_list= []
        y_train_history = []
        history_df = pd.DataFrame(columns=["step", "LE_instances", "LE_relative", "accuracy", "y_train_history", "y_pred","y_pred_prob"])
        history_df = history_df.astype({"step": "int64","LE_instances": "int64","LE_relative": "float64", "accuracy": "float64"})
                                        #"y_train_history" : , "y_pred":, "y_pred_prob":})
        results_df = pd.DataFrame(columns=["iteration","pred_dict"])
        error_relative = 0
        #n_classes (in training data) 
        label_names, label_counts = np.unique(y_train, return_counts=True)
        LE_trace_matrix = np.zeros((label_names.shape[0],label_names.shape[0]))
        
        #for every step between start and stop, intrude a label flip and fit the cl_
        np.random.seed(RANDOM_S)
        for i_, step_ in enumerate(range(error_start, error_stop, error_increasement)):
            y_train_history_current_step = []  #List of y_trains for every substep in following loop (list of lists)
            ##############print("current iteration: {}   current LE_step: {} error_relative: {}".format(i_, step_, error_relative))
            if step_ >= 1 and add_row == True:
                for e_i_ in range(1, error_increasement + 1 , 1):    
                    if error_relative >= error_stop_perc: ###---break criteria error_relative
                        print(str(error_relative) +"error threshold exceeds the current limit of {0} stop at iteration {1} ".format(error_stop_perc, step_))
                        add_row=False
                        break
                    else:
                        #random_label_choice1 & random_label_choice2 AND Ensure that: rlc2 =/= rlc1
                        #implementation_question: pick randomly accross the whole indexes OR fairly between label_names ?
                        #  
                        rlc1, rlc2 = np.random.choice(label_names, size=2, replace=False)
                        
                        #list all indexes where rcl1 is present
                        idx_l1 = np.where(y_train==rlc1)[0]
                        ##TODO Verify that there are potential instances present from the current class choice which are not in the blacklist
                        ## IF SO: change the class_pick 

                        #convert_label(rc_idx_l1--->random_label2) if rc_idx_l1 is not used AND append rc_idx_l1 to the black_list
                        rc_idx_l1 = np.random.choice(idx_l1)
                        while_limit = 20
                        while rc_idx_l1 in black_list:
                            print("PICKED the same instance again. CHANGED PICK SUCCESFULLY")
                            rc_idx_l1 = np.random.choice(idx_l1)
                            while_limit -=1
                            if while_limit <= 0:
                                print("MAX_LIMIT_REACHED. It seems that there is no flipable label left in class {}".format(rlc1))
                                rlc1, rlc2 = np.random.choice(label_names, size=2, replace=False)
                                idx_l1 = np.where(y_train==rlc1)[0]
                                rc_idx_l1 = np.random.choice(idx_l1)
                                while_limit = 20
                                

                        y_train[rc_idx_l1] = rlc2
                        black_list.append(rc_idx_l1)
                        LE_trace_matrix[np.where(label_names == rlc1)[0][0], np.where(label_names== rlc2)[0][0]] +=1
                        y_train_history_current_step.append(y_train.copy().tolist())
                        error_relative += error_perc_incr
                        error_relative = round(error_relative, float_prec)
                        label_names, label_counts = np.unique(y_train, return_counts=True)
                        print(f"changed label {rlc1} to {rlc2} at index {rc_idx_l1} of the data") 
             
                print("current class balance distribution: {}".format(dict(zip(label_names, label_counts))))

            elif step < 1 and add_row == True:
                y_train_history_current_step.append(y_train)   
                pass


            #fit classifier and make prediction  
            res_ = apply_TSC_algos(train_test_dct=train_test_df, classifiers=cl_dict)
            print("current iteration: {}   current LE_step: {} error_relative: {}".format(i_, step_, error_relative))
            if add_row:
                history_df.loc[i_] = [int(i_), int(step_), error_relative, np.round(res_[cl_]["accuracy"],float_prec),
                                      json.dumps(y_train_history_current_step),json.dumps(res_[cl_]["y_pred"].tolist()),
                                      json.dumps(res_[cl_]["y_pred_prob"].tolist())]
                # store lists as JSON instead of raw strings to improves data integrity for several storing & loading options.
                # res should be stored in /David_MA/dca/label_er/<current_cl>/<current_ds>/results.csv # maybe add conf_matrices. delta?
        history_df = history_df.astype({"step": "int64","LE_instances": "int64","LE_relative": "float64", "accuracy": "float64"})
        history_df.to_csv(RES_PATH, index=False)
        ###DELETION#####results_df.to_csv(PRED_PATH, index=False)
        ###DELETION#####np.savetxt(TRACE_PATH, LE_trace_matrix, delimiter=",", fmt="%d") 


    else: #if results are already present
        #(load_present_results())
        history_df = load_results(RES_PATH)
        #LE_trace_matrix = np.loadtxt(TRACE_PATH, delimiter=",", dtype=int)
        LE_trace_matrix = load_trace_m(history_df)
        LE_trace_matrix = LE_trace_matrix.astype(int) 
        pred_ = history_df.loc[:,["y_pred", "y_pred_prob"]]


    return history_df, history_df.loc[:,["y_pred", "y_pred_prob"]], LE_trace_matrix
    





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
    ax1.set_xlim(x_.min(), x_.max()+0.05*x_.max())
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


