# README.md - Time Series Classification Benchmarking and Data-Centric Optimization

## 📌 Project Overview
This repository provides a benchmarking framework for Time Series Classification (TSC) algorithms while incorporating data-centric optimization techniques. The project systematically introduces label errors to evaluate the impact on classification performance.

## 🚀 Features
- **Benchmarking multiple TSC algorithms**
- **Data preprocessing and visualization**
- **Implementation of data-centric optimization techniques**
- **Performance evaluation with label error injection**

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11
- Required libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, `tslearn`, `seaborn`, `aeon`,`tensorflow`
- Dataset: TSC archive datasets https://www.timeseriesclassification.com/dataset.php
- Ground paper: Bake off redux: a review and experimental evaluation of recent time series classification algorithms

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements3.txt
   ```
3. Run the benchmark pipeline:
   ```python
    from reprod_res.py
    import run_multiple_experiments, compare_acc_

    #run your own experiments
    python benchmark_pipeline.py
    current_classifiers = ["1NN-DTW", "GRAIL","Catch22", "FreshPRINCE","RDST", "RSF","R-STSF",
                        "RISE", "TSF","BOSS","WEASEL","ROCKET","Arsenal","CNN", "ResNet", "MR-Hydra"]

    current_dataset_names = ["Chinatown","Beef"]

    run_multiple_experiments(current_classifiers=current_classifiers, current_ds_n=current_dataset_names, res_id=resample_id)

    #compare own experiments and benchmark
    comparison_ArrowHead = compare_acc_(current_clfs=current_classifiers, current_ds="ArrowHead")
   ```

## 📂 Code Structure FOR Data-Centric-Apporaches

### 0 Run Pipeline
Run Pipeline SINGLE CL/DS Combination

import run_single_pipeline(CL, DS, RED, DCA, DoE_PARAM, EXP_FOLD, SAVE_FILES)

run_single_pipeline consists of the following sub-functions which can be imported and used individually
and consecutively:
1. load the data and (optionally) visualize it
2. initializes the current classifier
3. Apply the DCA systematically on the dataset
4. visualizes the performance decrease (accuracy) and the trace_M of the current experiment

```python
from pipeline_functions/pipeline_builders import run_single_pipeline

DATASET_NAME = "ElectricDevices"    #should be in DS_list
CLASSIFIER_NAME = "MR-Hydra"        #should be in cl_ names
REDUCTION_F = 10                    #only for large datasets
DCA= "LabelErrors"                  #
DoE_PARAM = {"random_seed":0,"start":0,"stop":10,"step":5}  #stop = max 90% of test_set_size, step=1-10 
EXP_FOLD = "simulation_results/"                            #respect folder structure
SAVE_FILES = True                                           #export files and figures in the respective directorys

current_pipeline = run_single_pipeline(args=[])
```

### 1️⃣ Load Data
Function to load and preprocess TSC datasets.
```python

from basic_func import dataset_provider,dataset_overview, apply_TSC_algos
#Load the current ds and the respective meta_data with the function dataset_provider
#A visualization (hist of labels) can optionally be included where the x,y ticks are returned
current_ds, current_meta = dataset_provider(name=DATASET_NAME, reduction_factor=REDUCTION_F, test_set_ratio="default_benchmark", random_state=0)
x_t, y_t = dataset_overview(train_test_dct=current_ds["y_train_small"] , dataset_name=DATASET_NAME) 

```

### 2️⃣ Choose TSC Algorithms which should be applied
Applies different TSC algorithms to a dataset.
```python
#Classifiers can be initialized with _set_bakeoff_classifier if the initial results should be as similar as possible
# to the current benchmark paper. The Classifier should be initialized in a dict, with its name as as key.
current_cl = _set_bakeoff_classifier(CLASSIFIER_NAME, random_state=0, n_jobs=1)
cl_dict = {CLASSIFIER_NAME: current_cl}

```

### 3️⃣ Data-Centric <Optimization> - Intruding Label Errors
Function to systemmatically intrude random label errors
```python

from apply_dca import apply_label_erros, visualize_acc_decr, visualize_trace_M
#Apply label errors for the current DS. Define the DoE parameters (random_S, start, stop, step)
#The returned in three objects:
#  history_df      ---> with column structure: [step || LE_instances || LE_relative || accuracy]
#  res_            ---> pred_dict with : [acc, y_pred, y_pred_prob]
#  LE_trace_matrix ---> np.array (dim=2, dtype=int) with label flip history
#For the Storage inspect the function.
df_, res_, trace_m_= apply_label_erros(train_test_df=current_ds, cl_dict=cl_dict, ds_=DS_NAME, stop=300, stop_percentage=0.7,  step=5)
```

### 4️⃣ Performance Analysis
Plots accuracy against label noise.
```python
visualize_acc_decr(df_acc_inst_rel=df_, dpi_=150, first="relative", second=None, w_=4.5, h_=3, cl_=CLASSIFIER_NAME, ds_=DATASET_NAME, save_fig=True)
```

```python
visualize_trace_M(trace_M=trace_m_, cl_=CLASSIFIER_NAME, ds_=DATASET_NAME, dpi=200, filename_="trace_M", save_fig=True)
```

---

## 📊 Results and Analysis
- The benchmarking framework provides insight into how Data centric adaptations (e.g. label errors) impact classification performance.
- By systematically introducing errors, we can analyze algorithm robustness and data quality sensitivity.
- The plots visualize the degradation of accuracy with increasing noise levels.

## 📜 License
This project is licensed under the AAI License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact
For questions, issues, or contributions, feel free to open an issue or contact the repository maintainer david.hax@uni-bayreuth.de
