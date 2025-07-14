import panel as pn
import holoviews as hv
from holoviews import dim
import numpy as np
import pandas as pd
import seaborn as sns
#from IPython.display import display
hv.extension('bokeh')
#pn.extension('ipywidgets', 'tabulator',"bokeh", sizing_mode='stretch_width') --> for Jupyter Notebook

nested_df = pd.read_pickle("simulation_results/nested_df_agg.pkl")
nested_super_df = pd.read_pickle("simulation_results/nested_super_df.pkl")
nested_super_df_extended_cat = pd.read_pickle("simulation_results/nested_super_df_extended_cat.pkl")
datasets = nested_df['dataset'].unique().tolist()
classifiers = nested_df['classifier'].unique().tolist()


# Define consistent colors for datasets
dataset_colors = dict(zip(datasets, sns.color_palette("tab10", len(datasets)).as_hex()))

# Define consistent line styles for classifiers
linestyles = ['solid', 'dashed', 'dotted']
classifier_styles = dict(zip(classifiers, linestyles))


# Panel widgets
dataset_select = pn.widgets.MultiChoice(name="Datasets",
                                        options=datasets,
                                        value=[datasets[0]],
                                        sizing_mode='stretch_width',
                                        styles={'font-size': '20pt'})
classifier_select = pn.widgets.MultiChoice(name="Classifiers",
                                            options=classifiers,
                                            value=[classifiers[0]],
                                            sizing_mode='stretch_width',
                                            styles={'font-size': '20pt'})

@pn.depends(dataset_select, classifier_select)
def acc_degr_plot(selected_datasets, selected_classifiers):
    if not selected_datasets or not selected_classifiers:
        return hv.Text(0.1, 0.5, "Select at least one dataset and classifier")

    overlays = []
    for ds in selected_datasets:
        for clf in selected_classifiers:
            truth_search= (nested_df.loc[:,"dataset"]==ds) & (nested_df.loc[:,"classifier"]==clf)
            res_df = nested_df.loc[truth_search, "acc_drop_df"].iloc[0] #only one row
            x = res_df["LE_relative"]
            y = res_df["accuracy"]
            label = f'{ds} - {clf}'
            curve = hv.Curve((x, y), 'Label Error (%)', 'Accuracy', label=label).opts(
                color=dataset_colors[ds],
                line_dash=classifier_styles[clf],
                line_width=2,
                tools=['hover', "pan"], 
                active_tools=[], 
                muted_alpha=0.1
            )
            overlays.append(curve)

    return hv.Overlay(overlays).opts(
        responsive=True,
        height=350,
        show_legend=True,
        legend_position='right',
        title="Accuracy Drop vs Label Error",
        fontscale=1.5
    )




#### ------------------------------------------------- Second Plot ------------------------------------------------- ####
compare_clf_x = pn.widgets.Select(name="Classifier X", options=classifiers, value=classifiers[0], sizing_mode='stretch_width',
                                  styles={'font-size': '16pt'})
compare_clf_y = pn.widgets.Select(name="Classifier Y", options=classifiers, value=classifiers[1], sizing_mode='stretch_width',
                                   styles={'font-size': '16pt'})
le_slider = pn.widgets.FloatSlider(name="Label Error (%)", start=0.0, end=0.9, step=0.02, value=0.0, sizing_mode='stretch_width',
                                    styles={'font-size': '16pt'})


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
    df_le = df_filtered[np.isclose(df_filtered['LE_relative'], le_val)]

    pivot_df = (
        df_le
        .pivot(index='dataset', columns='classifier', values='accuracy')
        .dropna(subset=[clf_x, clf_y])
        .reset_index()
    )

    if pivot_df.empty:
        scatter_pane.object = hv.Text(0.5, 0.5, "No data for this classifier pair at this LE level.")
        return
    
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


    scatter = hv.Scatter(
        data=pivot_df,
        kdims=[clf_x],
        vdims=[clf_y, 'dataset', 'color']
    ).opts(
        xlabel=f'{clf_x} Accuracy',
        ylabel=f'{clf_y} Accuracy',
        size=8,
        color = 'color',
        responsive=True,
        tools=['hover'],  # Removed 'wheel_zoom' and 'pan'
        hover_tooltips=[
        ("Dataset", "@dataset"),
        (f"{clf_x} Acc", f"@{{{clf_x}}}{{0.0000}}"),
        (f"{clf_y} Acc", f"@{{{clf_y}}}{{0.0000}}")],
        active_tools=[], 
        height=350,
        xlim=(0, 1.03),
        ylim=(0, 1.03),
        title=f'{clf_x} vs {clf_y} at LE = {le_val:.2f}',
        fontscale=1.5
    )


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


    scatter_pane.object = (scatter * diag * info_box_x * info_box_y * avg_line_x * avg_line_y).opts(legend_position='right')

update_classifier_comparison()

compare_clf_x.param.watch(update_classifier_comparison, 'value')
compare_clf_y.param.watch(update_classifier_comparison, 'value')
le_slider.param.watch(update_classifier_comparison, 'value')



#### ------------------------------------------------- Third Plot ------------------------------------------------- ####
property_select = pn.widgets.Select(
    name="Dataset Property", 
    options=["Type", "no_classes_cat", "Length_cat", "train_size_cat"], 
    value="Type",
    sizing_mode='stretch_width'
)

property_scatter_pane = pn.pane.HoloViews(hv.Text(0.5, 0.5, "Loading..."), height=500, sizing_mode='stretch_width')

def update_property_scatter(event=None):
    property_name = property_select.value
    category_order = list(nested_super_df_extended_cat[property_name].cat.categories)
    le_val = le_slider.value
    df = nested_super_df_extended_cat[
        (np.isclose(nested_super_df_extended_cat['LE_relative'], le_val)) &
        (nested_super_df_extended_cat['dataset'] != "All")
    ]

    if df.empty:
        return hv.Text(0.5, 0.5, "No data available for this LE level.")

    scatter = hv.Scatter(
        data=df,
        kdims=[hv.Dimension(property_name, values=list(df[property_name].cat.categories))],
        vdims=['accuracy', 'classifier', 'dataset']
    ).opts(
        color='classifier',
        tools=['hover'],
        hover_tooltips=[
        ("Dataset", "@dataset"),
        ("Classifier", "@classifier"),
        ("Accuracy", "@accuracy{0.0000}")],
        active_tools=[],  
        size=8,
        xlabel=property_name,
        ylabel='Accuracy',
        responsive=True,
        ylim=(0, 1.03),
        height=350,
        fontscale=1.5,
        jitter=0.2,  # optional: makes categorical dots more readable
        title=f"Accuracy vs {property_name} at LE = {le_val:.2f}"
    )

    property_scatter_pane.object = (scatter).opts(legend_position='right')

update_property_scatter()
property_select.param.watch(update_property_scatter, 'value')
le_slider.param.watch(update_property_scatter, 'value')
    

# Layout
# dashboard = pn.Column(
#     pn.pane.Markdown("### Classifier Comparison Plot", styles={'font-size': '18pt'}),
#     pn.Row(compare_clf_x, compare_clf_y, le_slider, styles={'padding-left': '40px', 'padding-right': '40px'}),
#     scatter_pane ,
#     pn.layout.Divider(),
#     pn.Row(dataset_select, classifier_select, styles={'padding-left': '40px', 'padding-right': '40px','font-size': '50pt'}),
#     acc_degr_plot,
#     margin=(10, 10),
#     sizing_mode='stretch_both',
#     styles={'border': '1px solid lightgray', 'padding': '10px'}
# )

dashboard = pn.Column(
    pn.Tabs(
        ("Accuracy vs Label Error", pn.Column(
            pn.Row(dataset_select, classifier_select),
            acc_degr_plot  
        )),
        ("Classifier Comparison", pn.Column(
            pn.Row(compare_clf_x, compare_clf_y, le_slider),
            pn.Row(scatter_pane)
        )),
        ("Dataset property influence", pn.Column(
            pn.Row(le_slider, property_select),
            pn.Row(property_scatter_pane)
        )),
#        ("Classifier Comparison", pn.Column(
#            pn.Row(compare_clf_x, compare_clf_y, le_slider, property_select),
#            pn.Row(scatter_pane, property_scatter_pane)
#        )),

    ),
    sizing_mode='stretch_both',
    styles={'border': '1px solid lightgray', 'padding': '10px'}
)

#dashboard.servable()
pn.serve(dashboard, show=True, port=5015, title="Label Error Impact Dashboard")
#dashboard.show()