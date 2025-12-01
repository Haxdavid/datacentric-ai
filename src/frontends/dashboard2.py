import panel as pn
import holoviews as hv
from holoviews import dim
import seaborn as sns
import numpy as np
import pandas as pd
import colorcet as cc
import os
#from IPython.display import display
hv.extension('bokeh')
#pn.extension('ipywidgets', 'tabulator',"bokeh", sizing_mode='stretch_width') --> for Jupyter Notebook

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "simulation_results", "df_nested_interp_ext_13_70_50.pkl")

nested_super_df = pd.read_pickle(DATA_PATH)
all_datasets = nested_super_df[nested_super_df['dataset'] != "All"]['dataset'].unique().tolist()
classifiers = nested_super_df['classifier'].unique().tolist()





#### ------------------------------------------------- Fourth Plot ------------------------------------------------- ####


compare_clf_x = pn.widgets.Select(name="Classifier X", options=classifiers, value=classifiers[0], sizing_mode='stretch_width',
                                  styles={'font-size': '16pt'})
compare_clf_y = pn.widgets.Select(name="Classifier Y", options=classifiers, value=classifiers[1], sizing_mode='stretch_width',
                                   styles={'font-size': '16pt'})
le_slider = pn.widgets.FloatSlider(name="Label Error (%)", start=0.0, end=1.0, step=0.02, value=0.10, sizing_mode='stretch_width',
                                    styles={'font-size': '16pt'})
# Placeholder for multi-select, will be populated after first update
dataset_selector = pn.widgets.MultiChoice(name="Datasets",sizing_mode='stretch_width',
                                    styles={'font-size': '20pt'})


prechoice = ["BirdChicken", "DiatomSizeReduction", "FaceFour"]
dataset_selector.options = all_datasets
dataset_selector.value = [ds for ds in prechoice if ds in all_datasets]



scatter_pane = pn.pane.HoloViews(hv.Text(0.5, 0.5, "Loading..."), sizing_mode='stretch_width', height=500)
@pn.depends(compare_clf_x, compare_clf_y, le_slider, dataset_selector)

def update_classifier_comparison(event=None):
    clf_x = compare_clf_x.value
    clf_y = compare_clf_y.value
    le_val = le_slider.value
    visible_datasets = dataset_selector.value

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


    # Apply dataset filter
    pivot_df = pivot_df[pivot_df['dataset'].isin(visible_datasets)]

     # Assign color to each dataset
    palette = sns.color_palette("tab10", n_colors=len(visible_datasets))
    color_map = dict(zip(visible_datasets, palette.as_hex()))
    



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

    diag = hv.Curve([(0, 0), (1, 1)]).opts(color='gray',line_dash='dashed')
        
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
