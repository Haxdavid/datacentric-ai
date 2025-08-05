from tsml_eval.publications.y2023.tsc_bakeoff.set_bakeoff_classifier import (
    _set_bakeoff_classifier,
)
from tsml_eval.utils.experiments import assign_gpu
from src.utils import logger
import os
import multiprocessing

def assign_GPU():
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        try:
            gpu_id = assign_gpu(set_environ=True)
            logger.info(f"Assigned GPU {gpu_id}")
        except Exception as e:
            logger.warning(f"Could not assign GPU: {e}")


def get_n_jobs(max_local_cores=6):
    # Check if we're on a SLURM cluster
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        # Local execution: don't overload machine
        return min(max_local_cores, multiprocessing.cpu_count())


### Classifier Wrapper ###
class BakeoffClassifier:
    def __init__(self, name: str, random_state: int = 0):
        self.name = name
        self.random_state = random_state
        self.num_jobs = get_n_jobs(max_local_cores=6)
        logger.info(f"Initializing BakeoffClassifier with name: {self.name}, random_state: {self.random_state}")
        logger.info(f"Using {self.num_jobs} jobs for classifier training and prediction")
        self.model = _set_bakeoff_classifier(
            name, random_state=self.random_state, n_jobs=self.num_jobs
        )

    def __name__(self):
        return type(self.model).__name__
    
    def __repr__(self):
        logger.info(f"Building Classifier: BakeoffClassifier with name: {self.name}")
        return f"BakeoffClassifier(name='{self.name}')"

    def fit(self, X, y):
        #logger.info(f"Fitting classifier: {self.name}")
        self.model.fit(X, y)
        #logger.info(f"Classifier {self.name} fitted successfully")

    def predict(self, X):
        #logger.info(f"Making predictions with classifier: {self.name}")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)