import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import List, Callable, Dict, Union, Any, Tuple
from pandas.api.types import CategoricalDtype


from src.utils.metrics import original_accuracy, auc_calculator, average_train_time, average_eval_time
from src.utils.metrics import acc_robustness_calculator, early_degradation_point
from src.utils.utilizations import get_frames_and_names
from src.current_experiment import Experiment


###----------------------------------METRIC FUNC--------------------------------------###
### Metric_Mapper also in metrics.py
METRIC_FUNCTIONS = {
    "initial_accuracy": original_accuracy,
    "auc_score": auc_calculator,
    "acc_robustness": acc_robustness_calculator,
    "early_degradation" : early_degradation_point,
    "avg_train_time" : average_train_time,
    "avg_eval_time": average_eval_time,
}

DATASET_PROPERTIES = ["no_classes", "Type", "Length", "train_size"] #test_size

###----------------------------------CATEGORIZER--------------------------------------###
### Maps numerical values into binned Category Spans
number_of_class_categories = ["2", "3-5", "6-10", "11+"]
length_categories = ["1-199", "200-499", "500-999", "1000+"]
train_set_size_categories = ["1-99", "100-299", "300-699", "700+"]
# "rel_training_size" average traininging instances per class (220 instances for 4 classes = 55 instances per class) 
#  "class_imbalance" 0.25, 0.5, 0.75, 1.0 (equal class distribution) 


def get_dataset_properties(dataset_: str,
                     ds_source: str = "./112UCRFolds/datasetTable.json",
                     return_fields: Union[str, List[str]] = "no_classes"
                    ) -> Union[Any, tuple]:
    """
    Retrieve selected metadata for a dataset from the dataset source.

    Parameters:
    - dataset_ (str): The name of the dataset.
    - ds_source (str): Path to the JSON file containing dataset metadata.
    - return_fields (Union[str, List[str]]): Field(s) to return. Options: 'no_classes', 'Type', 'Length'
        'train_size', 'test_size'.

    Returns:
    - Union[Any, tuple]: The requested value(s). Single value if one field is requested, tuple otherwise.

    Raises:
    - ValueError: If the dataset is not found or requested field is invalid.
    """
    df_source = pd.read_json(ds_source)
    relevant_row = df_source[df_source.Dataset == dataset_]

    if relevant_row.empty:
        raise ValueError(f"Dataset '{dataset_}' not found in source.")

    field_map = {
        "no_classes": "Number_of_classes",
        "Type": "Type",
        "Length": "Length", 
        "train_size": "Train_size",
        "test_size": "Test_size",
    }

    if isinstance(return_fields, str):
        return_fields = [return_fields]

    results = []
    for field in return_fields:
        if field not in field_map:
            raise ValueError(f"Invalid return field: {field}")
        value = relevant_row[field_map[field]].values[0]
        results.append(value)

    return results[0] if len(results) == 1 else tuple(results)


def aggregate_accuracy_curves(
    master_df, 
    classifier, 
    x_common=np.linspace(0, 0.9, 46), 
    agg_func=np.mean,
    group_by=None
):
    """
    Aggregates accuracy degradation curves for a given classifier.

    Parameters:
    - master_df: pandas DataFrame with ['dataset', 'classifier', 'acc_drop_df']
    - classifier: str, filter on this classifier
    - x_common: array-like of LE_relative values to interpolate on (e.g. np.linspace(0, 0.9, 10))
    - agg_func: function like np.mean, np.median
    - group_by: optional, function or column name to group datasets by meta-feature

    Returns:
    - pd.DataFrame with aggregated accuracy curves
    """
    clf_df = master_df[master_df['classifier'] == classifier]

    # Return linear interpolation of accuracy degradation for common x values
    def interpolate_curve(df):
        x = df['LE_relative'].values
        y = df['accuracy'].values
        interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        return interp_func(x_common)

    # Optional grouping by meta-feature
    if group_by is not None:
        grouped = clf_df.groupby(group_by)
        result = {}
        for group_value, group_df in grouped:
            curves = []
            for _, row in group_df.iterrows():
                curve = interpolate_curve(row['acc_drop_df'])
                curves.append(curve)
            agg_curve = agg_func(np.vstack(curves), axis=0)
            result[group_value] = agg_curve
        return pd.DataFrame(result, index=x_common).reset_index().rename(columns={'index': 'LE_relative'})
    
    # No grouping
    curves = []
    for _, row in clf_df.iterrows():
        curve = interpolate_curve(row['acc_drop_df'])
        curves.append(curve)

    agg_curve = agg_func(np.vstack(curves), axis=0)
    


    return pd.DataFrame({
        'LE_relative': x_common,
        'accuracy': agg_curve
    })


def unpack_and_interpolate(nested_df, x_common=np.linspace(0, 0.9, 46)):
    """
    Unpacks and interpolates the nested acc_drop_df for each dataset/classifier.

    Parameters:
    - nested_df: DataFrame with columns ['dataset', 'classifier', 'acc_drop_df']
    - x_common: array of LE_relative values to interpolate to

    Returns:
    - Flat DataFrame with columns ['dataset', 'classifier', 'LE_relative', 'accuracy']
    """
    rows = []

    for _, row in nested_df.iterrows():
        dataset = row['dataset']
        classifier = row['classifier']
        acc_df = row['acc_drop_df']

        x = acc_df['LE_relative'].values
        y = acc_df['accuracy'].values

        # Create interpolation function
        interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')

        # Interpolate at fixed points
        y_interp = interp_func(x_common)

        # Collect rows
        for le, acc in zip(x_common, y_interp):
            rows.append({
                'dataset': dataset,
                'classifier': classifier,
                'LE_relative': round(le, 4),
                'accuracy': round(acc, 4)
            })

    # Create final flat DataFrame
    return pd.DataFrame(rows)


def generate_master_df(exp_dict: Dict[str, Tuple[dict, Experiment]],
                     metric_functions: Dict[str, Callable[[pd.DataFrame], float]],
                     dataset_properties: List[str]
                    ) -> pd.DataFrame:
    """
    Evaluate multiple metrics for a list of DataFrames and return a summary DataFrame.

    Parameters:
    - exp_dict (dict): Dictionary mapping experiment names to a list with:
        [0] exp_conf: dict, experiment configuration
        [1] exp_obj: Experiment object containing attributes:
                     [*Attribute_List*]
    - metric_functions (Dict[str, Callable]): Dictionary mapping metric names (str)
      to functions that take a DataFrame and return a float.
    - dataset_properties (List[str]): List of dataset properties to include in the results.

    Returns:
    - pd.DataFrame: DataFrame with columns:
      ['Classifier_name', 'ds_name', <metric_1>, <metric_2>, <...>, <property_1>, <property_2>, ...]
      where each row contains results for one classifier-dataset pair.
    """

    results = []
    dfs_, clfs_, ds_names, clfs_seed_ = get_frames_and_names(exp_dict = exp_dict)

    for df_, clf_name, ds_name in zip(dfs_, clfs_, ds_names):
        result_entry = {
            "Classifier_name": clf_name,
            "ds_name": ds_name
        }

        for metric_name, metric_func in metric_functions.items():
            try:
                score = metric_func(df_)
            except Exception as e:
                print(f"Failed to calculate {metric_name} for {clf_name} on {ds_name}: {e}")
                score = None
                
            result_entry[metric_name] = score
        
        property_values = get_dataset_properties(ds_name, return_fields=dataset_properties)
        for property_name, property in zip(dataset_properties, property_values):
            result_entry[property_name] = property
   
        results.append(result_entry)

    return pd.DataFrame(results)


def create_nested_df_from_exp_dict(exp_dict: dict) -> pd.DataFrame:
    """
    Create a nested DataFrame from the experimental dictionary.
    
    Parameters:
    exp_dict (dict): Dictionary containing experimental results.
    
    Returns:
    pd.DataFrame: Nested DataFrame with experiment results.
    """
    rows = []
    dfs_, clfs_, ds_names, clfs_seed_ = get_frames_and_names(exp_dict = exp_dict)
    
    rows = []
    for df, clf_name, ds_name in zip(dfs_, clfs_, ds_names):
        rows.append({
            "dataset": ds_name,
            "classifier": clf_name,
            "acc_drop_df": df.loc[:,"step":"accuracy"]  # this is a full DataFrame
        })

    return pd.DataFrame(rows)


def extend_nested_df_with_properties(nested_df: pd.DataFrame, dataset_properties: List[str]) -> pd.DataFrame:

    """
    Extend the nested DataFrame with additional dataset properties.

    Parameters:
    - nested_df: DataFrame with columns ['dataset', 'classifier', 'LE_relative', 'accuracy', <other columns>]
    - dataset_properties: List of properties to retrieve for each dataset

    Returns:
    - Extended DataFrame with additional columns for each property
    """

    # Get unique datasets
    unique_datasets = nested_df["dataset"].drop_duplicates()

    # Build a mapping DataFrame of dataset -> properties
    property_records = []
    for dataset_name in unique_datasets:
        property_values = get_dataset_properties(dataset_name, return_fields=dataset_properties)
        record = {"dataset": dataset_name}
        record.update(dict(zip(dataset_properties, property_values)))
        property_records.append(record)
    
    # SConvert to DataFrame and perform a left inner join
    properties_df = pd.DataFrame(property_records)
    extended_df = nested_df.merge(properties_df, on="dataset", how="left")

    return extended_df


def categorize_dataset_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins or reduces categories for certain dataset properties.
    
    Expects columns: 'no_classes', 'Length', 'train_size', 'Type'
    """
    
    # Bin 'no_classes'
    def bin_no_classes(n):
        if n == 2:
            return "2"
        elif 3 <= n <= 5:
            return "3-5"
        elif 6 <= n <= 10:
            return "6-10"
        else:
            return "11+"
    
    # Bin 'Length'
    def bin_length(n):
        if n < 200:
            return "1-199"
        elif n < 500:
            return "200-499"
        elif n < 1000:
            return "500-999"
        else:
            return "1000+"
    
    # Bin 'train_size'
    def bin_train_size(n):
        if n < 100:
            return "1-99"
        elif n < 300:
            return "100-299"
        elif n < 700:
            return "300-699"
        else:
            return "700+"
    
    # Apply binning
    df["no_classes_cat"] = df["no_classes"].apply(bin_no_classes)
    df["Length_cat"] = df["Length"].apply(bin_length)
    df["train_size_cat"] = df["train_size"].apply(bin_train_size)

    # Optionally drop or replace original columns
    # df.drop(["no_classes", "Length", "train_size"], axis=1, inplace=True)
    # or
    # df["no_classes"] = df["no_classes_binned"]
    # df.drop("no_classes_binned", axis=1, inplace=True)

    # Define categorical orders
    df["no_classes_cat"] = df["no_classes_cat"].astype(
        CategoricalDtype(categories=["2", "3-5", "6-10", "11+"], ordered=True)
    )
    df["Length_cat"] = df["Length_cat"].astype(
        CategoricalDtype(categories=["1-199", "200-499", "500-999", "1000+"], ordered=True)
    )
    df["train_size_cat"] = df["train_size_cat"].astype(
        CategoricalDtype(categories=["1-99", "100-299", "300-699", "700+"], ordered=True)
    )
    df["Type"] = df["Type"].astype(
        CategoricalDtype(categories=['IMAGE', 'SIMULATED', 'SENSOR', 'TRAFFIC',
                                      'DEVICE', 'ECG', 'SPECTRO', 'HAR','EOG'], ordered=True)
    )
    return df


### AGGREGATE results over one classifier and append to nested_df
def aggregate_accuracy_curvers_multiple(nested_df: pd.DataFrame) -> pd.DataFrame:
    for cl_ in nested_df['classifier'].unique():
        agg_df = aggregate_accuracy_curves(master_df=nested_df, classifier=cl_)
        new_row = pd.DataFrame([{
            'dataset': 'All',
            'classifier': cl_,
            'acc_drop_df': agg_df
        }])
        # Append to master_df
        nested_df = pd.concat([nested_df, new_row], ignore_index=True)
    return nested_df