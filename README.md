# Influence of Label Noise on Time Series Classification  
### A Data-Centric Framework for Large-Scale Robustness Evaluation

## 📌 Project Overview
This repository accompanies the Master's thesis *“Empirical Evaluation of a Data-Centric Framework for Time Series Classification”*.  
It provides a reproducible pipeline for applying **controlled label-noise levels** to benchmark datasets and evaluating how **state-of-the-art Time Series Classification (TSC) algorithms** degrade under noise.

The framework enables a **data-centric perspective** by analyzing:
- Performance degradation across 50 noise levels (0 → 1)  
- Differences in robustness between algorithmic families  
- How dataset properties (training size, series length, class count, …) shape degradation  
- Full performance trajectories rather than single-point evaluations  

The framework is extensible and serves as a foundation for future data-centric perturbation studies.

---

## 🚀 Features
- Systematic injection of **label noise** into TSC benchmark datasets  
- Evaluation of **13+ representative TSC algorithms**  
- **Robustness evaluation** via normalized area-under-curve  
- **Fine-grained performance trajectories** across noise levels  
- Support for statistical modelling (e.g., linear mixed-effects models)  
- Fully reproducible and dataset-agnostic design  

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11  
- Key libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, `tslearn`, `seaborn`, `aeon`, `tensorflow`, `statsmodels`  
- UCR TSC datasets: https://www.timeseriesclassification.com/dataset.php
- Classifier implementation and Benchmarking on the base of : Bake Off Redux Paper: https://link.springer.com/article/10.1007/s10618-024-01022-1



### Installation
```bash
git clone https://github.com/Haxdavid/datacentric-ai
cd datacentric-ai
pip install -r requirements.txt
```
### Setup
- If you want the use the framework with the default benchmark split of the 112UCRFolds ensure that:
   datasets are **correctly stored** in the folder datasets/112UCRFold0 with:
   <name>_TRAIN.ts or <name>_TEST.ts
- current results for evaluation as well as trajectories are stored in /simulation results. If you want to change the result directory change it src/utils/utilizations system wide or define the res_path in main.py


### 📂 Code Structure FOR Data-Centric-Apporaches
```
datacentric-ai/
├── benchmark/              # Baseline results and comparison utilities
├── configs/                # Experiment configuration files
├── datasets/               # Dataset folder 
|   ├── 112UCRFold0/        # UCR dataset default benchmark split (train/test splits)
|
├── experiments/            # Label-noise experiment pipeline
├── notebooks/              # Analysis + reproduction notebooks
├── scripts/                # Helper scripts
├── simulation_results/     # Outputs of large-scale simulations
│
├── src/                    # Core source code
│   ├── data_handlers/      # Data loading & preprocessing
│   │   ├── basic_func.py
│   │   └── data_processing.py
│   │
│   ├── dca/                # (Legacy) data-centric augmentation modules
│   │   ├── apply_dca.py
│   │   └── le_func.py
│   │
│   ├── frontends/          # Dashboard frontends
│   │   ├── dashboard.py
│   │   └── dashboard2.py
│   │
│   ├── models/             # Classifier wrappers + training API
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









# Influence of Data Centric Optimization on Time Series Classification algorithms 

## 📌 Project Overview
This repository provides a framework for for applying systematic Label Noise Levels to a given Set of
Time Series Classification (TSC) algorithms on a huge variety of published datasets. The project enables a data centric perspective ......


## 🚀 Features
- **Benchmarking multiple TSC algorithms**
- **Data preprocessing and visualization**
- **Implementation of data-centric optimization techniques**
- **Performance evaluation/algorithm robustness while applying DCA**

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11
- Required libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, `tslearn`, `seaborn`, `aeon`,`tensorflow`
- Dataset: TSC archive datasets https://www.timeseriesclassification.com/dataset.php
- Base paper: Bake off redux: a review and experimental evaluation of recent time series classification algorithms
- Related Thesis: https://thesis_link by David R.T. Hax

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Haxdavid/datacentric-ai
   cd https://github.com/Haxdavid/datacentric-ai
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements3.txt
   ```
3. Observe or Run the benchmarks and compare own results to benchmark results
   with reprod_res.ipynb
   

## 📂 Code Structure FOR Data-Centric-Apporaches

### 0 Run Pipeline
Run Pipeline SINGLE CL/DS Combination

run_single_pipeline consists of the following sub-functions which can be imported and used individually
and consecutively:
1. load the data and (optionally) visualize it
2. initializes the current classifier
3. Apply the DCA systematically on the dataset
4. visualizes the performance decrease (accuracy) and the trace_M of the current experiment

```python
from pipeline_builder import run_single_pipeline

DATASET_NAME = "ElectricDevices"    #should be in DS_list
CLASSIFIER_NAME = "MR-Hydra"        #should be in cl_ names
REDUCTION_F = 10                    #only for large datasets
RANDOM_S = 0                        #Random Seed for everything except the DCA
DCA= "LabelErrors"                  #DCA Strategy Category --> Determines DoE_PARAM DICT
DoE_PARAM = {"random_seed":0,"start":0,"stop":10,"step":5}  #stop = max 90% of test_set_size, step=1-10 
EXP_FOLD = "simulation_results/"                            #respect folder structure
SAVE_FILES = True                                           #export files and figures in the respective directorys
VIS_DATA = False                                            # Visualizes DataDistribution before applying DCA

current_pipeline_res = run_single_pipeline(*exp_args=[])
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

from apply_dca import apply_label_errors, visualize_acc_decr, visualize_trace_M
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
