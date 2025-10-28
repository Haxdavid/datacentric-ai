import panel as pn
import holoviews as hv
from holoviews import dim
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
#from IPython.display import display
hv.extension('bokeh')
#pn.extension('ipywidgets', 'tabulator',"bokeh", sizing_mode='stretch_width') --> for Jupyter Notebook


#nested_super_df = pd.read_pickle("simulation_results/nested_super_df.pkl")
nested_super_df = pd.read_pickle("simulation_results/df_nested_interp")
datasets = nested_super_df['dataset'].unique().tolist()
classifiers = nested_super_df['classifier'].unique().tolist()
dataset_colors = dict(zip(datasets, cc.glasbey[:len(datasets)]))  # Or use other palettes





#### ------------------------------------------------- Fourth Plot ------------------------------------------------- ####


compare_clf_x = pn.widgets.Select(name="Classifier X", options=classifiers, value=classifiers[0], sizing_mode='stretch_width',
                                  styles={'font-size': '16pt'})
compare_clf_y = pn.widgets.Select(name="Classifier Y", options=classifiers, value=classifiers[1], sizing_mode='stretch_width',
                                   styles={'font-size': '16pt'})
le_slider = pn.widgets.FloatSlider(name="Label Error (%)", start=0.0, end=0.9, step=0.02, value=0.10, sizing_mode='stretch_width',
                                    styles={'font-size': '16pt'})
# Placeholder for multi-select, will be populated after first update
dataset_selector = pn.widgets.MultiChoice(name="Datasets",sizing_mode='stretch_width',
                                    styles={'font-size': '20pt'})





scatter_pane = pn.pane.HoloViews(hv.Text(0.5, 0.5, "Loading..."), sizing_mode='stretch_width', height=500)
@pn.depends(compare_clf_x, compare_clf_y, le_slider)

def update_classifier_comparison(event=None):
    clf_x = compare_clf_x.value
    clf_y = compare_clf_y.value
    le_val = le_slider.value

    df_filtered = nested_super_df[
        (nested_super_df['classifier'].isin([clf_x, clf_y])) &
        (nested_super_df['dataset'] != "All")
    ]
    df_le = df_filtered[df_filtered['LE_relative'] <= le_val]



    pivot_df = df_le.pivot_table(
        index=['dataset', 'LE_relative'],
        columns='classifier', values='accuracy'
        ).dropna(subset=[clf_x, clf_y]).reset_index()

    if pivot_df.empty:
        scatter_pane.object = hv.Text(0.5, 0.5, "No data for this classifier pair at this LE level.")
        return
    

    # Initialize dataset color map if needed
    prechoice = ["BirdChicken", "DiatomSizeReduction", "FaceFour"]
    unique_datasets = pivot_df['dataset'].unique()
    if not dataset_selector.options:
        dataset_selector.options = list(unique_datasets)
        dataset_selector.value = prechoice     #list(unique_datasets)[-6:-4]

    # Apply dataset filter
    visible_datasets = dataset_selector.value
    pivot_df = pivot_df[pivot_df['dataset'].isin(visible_datasets)]


     # Assign color to each dataset
    color_map = dict(zip(unique_datasets, cc.glasbey[:len(unique_datasets)]))



    # Build plot: lines and scatter points per dataset
    elements = []
    for ds in visible_datasets:
        ds_data = pivot_df[pivot_df['dataset'] == ds].sort_values('LE_relative')
        if not ds_data.empty:
            line = hv.Curve(
                (ds_data[clf_x], ds_data[clf_y]),
                label=ds
            ).opts(
                color=color_map[ds],
                line_width=2,
                tools=[],
            )

            points = hv.Scatter(
                ds_data,
                kdims=[clf_x],
                vdims=[clf_y, 'LE_relative']
            ).opts(
                color=color_map[ds],
                size=8,
                tools=['hover'],
                hover_tooltips=[
                    ("Dataset", ds),
                    ("LE", "@LE_relative"),
                    (f"{clf_x} Acc", f"@{{{clf_x}}}{{0.0000}}"),
                    (f"{clf_y} Acc", f"@{{{clf_y}}}{{0.0000}}")
                ],
            )

            elements.append(line * points)

    

    # Determine who wins
    pivot_df["winner"] = np.select(
        [pivot_df[clf_x] > pivot_df[clf_y], pivot_df[clf_x] < pivot_df[clf_y]],
        [clf_x, clf_y],
        default="Tie"
    )

    color_map = {clf_x: "green", clf_y: "red", "Tie": "gray"}
    pivot_df["color"] = pivot_df["winner"].map(color_map)

    # Count wins
    win_count_x = (pivot_df["winner"] == clf_x).sum()
    win_count_y = (pivot_df["winner"] == clf_y).sum()
    tie_count = (pivot_df["winner"] == "Tie").sum()

    # Dashed lines representing average performance of each classifier
    mean_x = pivot_df[clf_y].mean()
    avg_line_x = hv.Curve([(0, mean_x), (mean_x, mean_x)]).opts(
        color=color_map[clf_y],line_dash='dashed',line_width=2,alpha=0.7)
    
    mean_y = pivot_df[clf_x].mean()
    avg_line_y = hv.Curve([(mean_y, 0), (mean_y, mean_y)]).opts(
        color=color_map[clf_x],line_dash='dashed',line_width=2,alpha=0.7)

    diag = hv.Curve([(0, 0), (1, 1)]).opts(color='gray',line_dash='dashed')
        
    # Info box text
    winner_label_y = f"{clf_y} wins here"
    winner_label_x = f"{clf_x} wins here" 
    info_text_y = f"{winner_label_y}\n[{win_count_y}W, {tie_count}T, {win_count_x}L]"
    info_text_x = f"{winner_label_x}\n[{win_count_x}W, {tie_count}T, {win_count_y}L]"


    info_box_y = hv.Text(0.03, 0.95, info_text_y).opts(
        text_align='left',
        text_font_size='15pt',
        bgcolor='white',
        text_color=color_map[clf_y],
        padding=0.06
    )

    info_box_x = hv.Text(0.80, 0.07, info_text_x).opts(
        text_align='left',
        text_font_size='15pt',
        bgcolor='white',
        text_color=color_map[clf_x],
        padding=0.06
    )


    overlay = hv.Overlay(elements + [diag]).opts(
        xlabel=f'{clf_x} Accuracy',
        ylabel=f'{clf_y} Accuracy',
        xlim=(0, 1.03),
        ylim=(0, 1.03),
        responsive=True,
        active_tools=[],
        title=f'{clf_x} vs {clf_y} (History up to LE={le_val:.2f})',
        legend_position='right',
        fontscale=1.5,
        height=400
    )

    scatter_pane.object = overlay


update_classifier_comparison()

# === Attach watchers ===
compare_clf_x.param.watch(update_classifier_comparison, 'value')
compare_clf_y.param.watch(update_classifier_comparison, 'value')
le_slider.param.watch(update_classifier_comparison, 'value')
dataset_selector.param.watch(update_classifier_comparison, 'value')


# === Layout ===
layout = pn.Column(
    pn.Tabs(
        ("CL_Comparison History", pn.Column(
            pn.Row(compare_clf_x, compare_clf_y, le_slider),
            pn.Row(dataset_selector),
            pn.Row(scatter_pane)
            )),
        ("Next_Tab", pn.Column()),
    ),
    sizing_mode='stretch_both',
    styles={'border': '1px solid lightgray', 'padding': '10px'}
)


pn.serve(layout, show=True, port=5017, title="Label Error Impact Dashboard")
