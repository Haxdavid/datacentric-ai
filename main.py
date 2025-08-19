import os 
from src.utils import load_and_expand_yaml, logger, RESULTS_DIR
from src.current_experiment import Experiment

print(__name__)

if __name__ == "__main__":
    config_path = "experiments/experiment_Car15.yaml"
    base_path = "112UCRFolds"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    configs = load_and_expand_yaml(config_path)

    for config in configs:
        experiment = Experiment(config, base_path=base_path, results_root=RESULTS_DIR)
        experiment.run_experiment()

    logger.info("All experiments completed")
