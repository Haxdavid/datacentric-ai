# Influence of Label Noise on Time Series Classification  
### A Data-Centric Framework for Large-Scale Evaluation of Performance Degradation under Label Noise

## 📌 Project Overview
This repository accompanies the Master's thesis *“Empirical Evaluation of a Data-Centric Framework for Time Series Classification: Investigating Machine Learning and Deep
Learning Model Performance across Dataset Characteristics and Algorithm Classes”*.  
It provides a reproducible pipeline for applying **controlled label-noise levels** to benchmark datasets and evaluating how **state-of-the-art Time Series Classification (TSC) algorithms** degrade under noise.

The framework enables a **data-centric perspective** by analyzing:
- Performance degradation across 50 noise levels (0 → 1)  
- Differences in robustness between algorithmic families  
- How dataset properties (training size, series length, class count, …) shape degradation  

The framework is extensible and serves as a foundation for future data-centric perturbation studies.

---

## 🚀 Features
- Systematic injection of **label noise** into TSC benchmark datasets  
- Evaluation of **13+ representative TSC algorithms**  
- **Robustness evaluation** via normalized area-under-curve or through
- **Fine-grained performance trajectories** across all noise levels  
- Support for statistical modelling (e.g., linear mixed-effects models)  
- Reproducible, modular, and dataset-agnostic experiment design 

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11  
- Key libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, `tslearn`, `seaborn`, `aeon`, `tensorflow`, `statsmodels`  
- UCR TSC datasets: https://www.timeseriesclassification.com/dataset.php
- Classifier implementation and Benchmarking on the base of : Bake Off Redux: https://link.springer.com/article/10.1007/s10618-024-01022-1
- Master Thesis (Hax) - link forthcoming



### Installation
```bash
git clone https://github.com/Haxdavid/datacentric-ai
cd datacentric-ai
pip install -r requirements.txt
```
### Setup
- If using the default UCR benchmark structure, ensure that datasets are stored under:
   datasets/112UCRFold0/ 
   with filenames of the form:
   <dataset_name>_TRAIN.ts or <dataset_name>_TEST.ts
- Experimental results, including robustness metrics and noise trajectories, are stored in
   simulation results/
   To change the output directory, modify the corresponding path in src/utils/utilizations.py or override res_path in main.py.


## 📂 Code Structure of this Framework
This Framework is structured as follows:

```
datacentric-ai/
├── benchmark/              # Baseline results and comparison utilities
├── configs/                # matplotlib style configuration for thesis
├── datasets/               # Dataset folder 
|   ├── 112UCRFold0/        # UCR dataset default benchmark split (train/test splits)
|
├── experiments/            # yaml files (experimetal configurations)
├── notebooks/              # Analysis + reproduction notebooks
├── server_scripts/         # scripts for computing experiments on HCP
├── simulation_results/     # Outputs of large-scale simulations
│
├── src/                    # Core source code
│   ├── data_handlers/      # Data loading & preprocessing
│   │   ├── basic_func.py
│   │   └── data_processing.py
│   │
│   ├── dca/                # (Legacy) data-centric augmentation modules
│   │   ├── apply_dca.py    # <---Dominat Function Logic !  
│   │   └── le_func.py
│   │
│   ├── frontends/          # Dashboard frontends
│   │   ├── dashboard.py
│   │   └── dashboard2.py
│   │
│   ├── models/             # Classifier wrappers + TSC-API
│   │   ├── classifierWrapper.py
│   │   └── tsc_algos.py
│   │
│   ├── utils/              # Shared utilities + metrics
│   │   ├── metrics.py
│   │   └── utilizations.py
│   │
│   ├── visuals/            # Visualization utilities
│   │   └── visualizations.py
│   │
│   └── current_experiment.py
│
├── .venv/                  # Local virtual environment (ignored)
├── main.py                 # Legacy entry point
├── requirements.txt
├── requirements_server.txt
└── README.md
```



##  Run Experiments

### Run main.py
Define the experimental space in the yaml configuration file (see example configurations in folder)
```python
if __name__ == "__main__":
    config_path = "experiments/<your_config>yaml"
    base_path = "datasets/112UCRFold0"
    results_path = "simulation_results"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    configs = load_and_expand_yaml(config_path)

    for config in configs:
        experiment = Experiment(config, base_path=base_path, results_root=results_path)
        experiment.run_experiment()

    logger.info("All experiments completed")

```


### Experimental Configuration

Experiments are defined via YAML configuration files.  
A minimal example (`experiment.yaml`) looks as follows:

```yaml
EXPERIMENT:
  - DATASET_NAME: ["Fish", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale"]
    CLASSIFIER_NAME: ["MR-Hydra", "Quant", "Weasel-D", "Arsenal"]
    REDUCTION_F: [1]
    RANDOM_S: 0
    DCA:
      - type: "LabelErrors"
        DoE_param:
          le_strategy: "leV1"
          p_vec: null
          random_seed: 0
          start: 0
          stop: 100
          step: 2
```

This configuration yields:  
**4 datasets × 4 classifiers × 1 random seed × 50 noise levels**,  
using the `"leV1"` label-noise strategy, which generates **uniform synthetic label noise** across the specified range.

---

## Data Post-Processing, Analysis, and Visualization

After running an experiment, results can be explored using the provided notebooks:
- **`classifier_comparison.ipynb`**  
  Main notebook for post-processing, robustness analysis, visualizations, and aggregated comparisons.
- **`run_experiment_example2.ipynb`**  
  Demonstrates the full pipeline, including configuration loading, experiment execution, and structured preprocessing.

---

## Quick-Start: Running a Single Classifier–Dataset Experiment

To become familiar with the framework structure, a simplified pipeline is provided.  
It allows running a **single classifier–dataset combination** and stepping through the experiment logic interactively.

The notebook **`run_experiment_example1.ipynb`** guides the user through the following components:

1. **Load and optionally visualize the dataset**  
2. **Initialize the selected classifier**  
3. **Apply the label-noise process systematically**  
4. **Visualize performance degradation** (accuracy trajectories and trace metrics)

This walkthrough introduces the experiment architecture and demonstrates the modular design of the framework.
E.g. in this Notebook the User can easily run through the subsequent steps following:

### 1️⃣ Define the Input Mask

```python
from src.dca.apply_dca import apply_label_errors

DATASET_NAME = "ProximalPhalanxTW"      #should be in DS_list
CLASSIFIER_NAME = "Quant"               #should be in cl_ names
REDUCTION_F = 1                         #optional. only for large datasets. Should be 1 if DS should be in its original size
RANDOM_S = 0                            #Random Seed for everything except the DCA
DCA= "LabelErrors"                      #DCA Strategy Category --> Determines DoE_PARAM DICT
DoE_PARAM = {"le_strategy":"leV1", "random_seed":0,"start":0,"stop":50,"step":2,"p_vec":None}    #stop = max 90% of test_set_size, step=1-10 
EXP_FOLD = "simulation_results/"        #respect folder structure
BASE_PATH = "datasets/112UCRFold0"               #Relative to root path where UCR FOLDS are contained
```

### 2️⃣ Load the Data

```python
from src.data_handlers.basic_func import dataset_provider,dataset_overview
#Load the current ds and the respective meta_data with the function dataset_provider
#A visualization (hist of labels) can optionally be included where the x,y ticks are returned
current_ds, current_meta = dataset_provider(name=DATASET_NAME, reduction_factor=REDUCTION_F,
                                            test_set_ratio="default_benchmark",
                                            random_state=0, base_path=BASE_PATH)
x_t, y_t = dataset_overview(train_test_dct=current_ds["y_train_small"] , dataset_name=DATASET_NAME)
```

### 3️⃣ Choose TSC Algorithms which should be applied 

```python
from tsml_eval.publications.y2023.tsc_bakeoff.run_experiments import _set_bakeoff_classifier
#Classifiers can be initialized with _set_bakeoff_classifier if the initial results should be as similar as possible
# to the current benchmark paper. The Classifier should be initialized in a dict, with its name as as key.
current_cl = _set_bakeoff_classifier(CLASSIFIER_NAME, random_state=0, n_jobs=1)
cl_dict = {CLASSIFIER_NAME: current_cl}
```
### 4️⃣ Apply Label Noise

```python
from src.dca.apply_dca import apply_label_errors
#Apply label errors for the current DS
df_, trace_M_= apply_label_errors(train_test_df=current_ds,
                                cl_dict=cl_dict,
                                ds_=DATASET_NAME,
                                doe_param=DoE_PARAM,
                                res_path=EXP_FOLD)
```

### Optioal: Immediate visualizations:
Function to load and preprocess TSC datasets.
```python
from src.visuals.visualizations import visualize_acc_decr, visualize_trace_M
#Observe the accuracy decrease when Label Errors get intruded
visualize_acc_decr(df_acc_inst_rel=df_, dpi_=150, first="relative",
                  second=None, w_=4.5, h_=3, cl_="QUANT",
                  ds_="ProxiPhTW", save_fig=False)
visualize_trace_M(trace_M=trace_M_, cl_="Quant", ds_="ProxiPhTW",dpi=200,
                   filename_="trace_M", save_fig=False, exp_folder=None)
```

---

## 📊 Results and Analysis
The benchmarking framework provides insight into how Data centric adaptations (for now only label noise) impact classification performance and ranking stability. Additionaly it is investigated how dataset properties shape the performance degradation.

## 📜 License
This project is not licensed yet

## 📬 Contact
For questions, issues, or contributions, feel free to open an issue or contact the repository maintainer david.hax@uni-bayreuth.de
