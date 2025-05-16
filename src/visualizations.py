import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib import gridspec



def visualize_acc_decr(df_acc_inst_rel, w_=6, h_=4, dpi_=150, first="relative", second=None,
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



def visualize_acc_decr_multiple(multiple_df, vis_names_, w_=6, h_=4, dpi_=150, ds_="ds_0", filename_="acc_decr", save_fig=False, exp_folder=None):
    """
    VISUALIZE accuracy decrease of multiple algorithms and/or algorithm instances and one SINGLE dataset/DCA -combination.
    RECEIVE: multiple_df: DataFrames with the column structure accuracy;LE_instances;LE_relative
             w_, h_, dpi: weight, height and dpi of the figure
    RETURNS: Nothing
    STORES: IF save_fig == True: stores figure in exp_path (directory_current/ds_+filename)
    """
    
    COLOR_MAP = plt.get_cmap('tab10').colors
    
    color_names = COLOR_MAP
    base_names = sorted(list(set(name.split('_rs')[0] for name in vis_names_)))
    base_color_map = {base: color_names[idx % len(color_names)] for idx, base in enumerate(base_names)}


    directory_current = "simulation_results/"
    dataset_name = ds_
    if exp_folder is not None:
        directory_current = exp_folder
    EXP_PATH = os.path.join(directory_current + (dataset_name + "_" + filename_ + ".png"))

    fig, ax1 = plt.subplots(figsize=(w_, h_), dpi=dpi_)
    fig.suptitle('Impact of Label Errors on Model Accuracy', fontsize=12, fontweight='bold')
    x_label = "Label Errors (Relative)"

    #Initialize first plot
    y_label = "Accuracy "+ ds_
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    x_max=0.2
    y_max = 0.6

    for data_, names_ in zip(multiple_df, vis_names_):
        acc_decr=data_["accuracy"]
        LE_relative=data_["LE_relative"]
        x_ = LE_relative
        

        # Safe split: handle names with or without '_rs'
        if '_rs' in names_:
            base_name = names_.split('_rs')[0]
            seed = int(names_.split('_rs')[-1])
        else:
            base_name = names_
            seed = 0  # Treat no seed as 'seed 0'

        base_color = base_color_map[base_name]
        alpha = 1.0 if seed == 0 else (0.6 if seed == 1 else 0.4)

        ax1.plot(x_ ,acc_decr, label=names_, color=base_color, alpha=alpha)
        if x_.max() > x_max: x_max = x_.max() 
        if acc_decr.max() > y_max: y_max = acc_decr.max()

    ax1.set_xlim(x_.min(), x_.max()+0.025*x_.max())
    ax1.set_ylim(top=y_max+0.05*y_max)
    ax1.tick_params(axis='y') #labelcolor=colors[0])
    ax1.grid(visible=True, linestyle='--', alpha=0.6, linewidth=0.5)

    handles, labels = ax1.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])  # sort by label
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)

    ### Independent of a second axis should be present: 
    # Finalize Figure aesthetics and saveplot
    ax1.legend(sorted_handles, sorted_labels, loc='upper right')
    fig.tight_layout()
    if save_fig==True:
        fig.savefig(fname=(EXP_PATH))


    plt.show()
