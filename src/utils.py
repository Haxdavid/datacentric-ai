import os
import logging
from itertools import product
import yaml


def setup_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)

logger = setup_logger("UtilsLogger")

RESULTS_DIR = "simulation_results"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.csv")


### Utility to Expand Configurations ###

def load_and_expand_yaml(path: str):
    logger.info(f"Loading and expanding YAML configuration from: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    def ensure_list(val):
        return val if isinstance(val, list) else [val]

    configs = []
    for block in ensure_list(config["EXPERIMENT"]):
        datasets = ensure_list(block["DATASET_NAME"])
        classifiers = ensure_list(block["CLASSIFIER_NAME"])
        reduction_f = ensure_list(block.get("REDUCTION_F", 1))
        seeds = ensure_list(block.get("RANDOM_S", 0))

        strategy_blocks = ensure_list(block["DCA"])

        for strategy in strategy_blocks:
            strategy_type = strategy["type"]
            doe_params = strategy.get("DoE_param", {})


            for dataset_name, classifier_name, seed, red_f in product(
                datasets, classifiers, seeds, reduction_f):
                new_conf = {
                    "DATASET_NAME": dataset_name,
                    "CLASSIFIER_NAME": classifier_name,
                    "REDUCTION_F": red_f,
                    "RANDOM_S":seed,
                    "DCA": {
                        "type": strategy_type,
                        "DoE_param": doe_params
                    },
                }
                configs.append(new_conf)

    logger.info(f"YAML configuration expanded into {len(configs)} configurations")
    return configs


def get_frames_and_names(exp_dict):
    """
    Get the dataframes and classifier names from the experimental dictionary.

    Parameters:
    experimental_dict (dict): Dictionary containing experiment configurations and results.

    Returns:
    tuple: A tuple containing a list of dataframes and a list of classifier names.
    """
    dataframes_ = []
    clf_names_ = []
    ds_names_ = []
    clf_names_with_seeds = []
    for exp_ in exp_dict.keys(): 
        dataframes_.append(exp_dict[exp_][1].df_)
        ds_names_.append(exp_dict[exp_][1].dataset_name)
        clf_names_.append(exp_dict[exp_][1].clf_name)
        clf_name_seed_ = exp_dict[exp_][1].clf_name + "_rs"+ str(exp_dict[exp_][0]["DCA"]["DoE_param"]["random_seed"])
        clf_names_with_seeds.append(clf_name_seed_)


    return dataframes_, clf_names_, ds_names_,  clf_names_with_seeds


