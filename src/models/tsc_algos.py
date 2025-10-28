import time
import numpy as np
from sklearn.metrics import accuracy_score
from src.utils.utilizations import setup_logger
logger = setup_logger("TSC_algo_Logger")
logger.info("Custom-named logger active.")



def apply_TSC_algos(train_test_dct, classifiers, exclude_classifiers=[" "]):
    """
    RECEIVE: train_test_dct with the intern format (Dict or DataFrame with the columns X_train, X_test, y_train, y_test
            and their reduced identity (X_train_small)), DICT of classifiers. potential of exclude classifiers
            present in classifier dict.
    RETURN: pred_dict: a nested dict with classifier names as keys for several prediction dicts containing
            multiple performance metrics and y_pred,y_pred_prob arrays.
    SUPPORTS: singe OR multiple classifiers at once. Pipeline is constructed for simple classifier usage
    """
    DEBUG = False
    pred_dict = {} #{"alg_name": {"accuracy": 0.0, "y_true": [0,0,0] "y_pred": [0,0,0], "y_pred_prob": [0,0,0]}}
    # Looping through the classifiers
    if not isinstance(classifiers, dict):
        raise TypeError("Classifiers should be a dictionary with names as keys and classifier objects as values.")
   
    for name, clf in classifiers.items():
        if name not in exclude_classifiers:
            start_time = time.time()
            #Representation of the classifier
            clf.__repr__()
            clf.fit(train_test_dct["X_train_small"],train_test_dct["y_train_small"])
            train_time = time.time() - start_time
            y_pred = clf.predict(train_test_dct["X_test_small"])
            y_pred_prob = clf.predict_proba(train_test_dct["X_test_small"])
            acc_score = accuracy_score(train_test_dct["y_test_small"], y_pred)
            #nll = log_los(train_test_array["y_test_small"], y_pred)
            #balanced_acc
            #AUROC=
            eval_time = time.time() - start_time - train_time
            #print("---------------------------- "+ f"Train time={train_time:.2f}s, Eval Time={eval_time:.2f}s")
            logger.info("------------------------"+ f"Train time={train_time:.2f}s, Eval Time={eval_time:.2f}s")
            pred_dict[name] = {"accuracy":acc_score,"y_train":train_test_dct["y_train_small"], "y_pred":y_pred,"y_pred_prob":y_pred_prob}
            pred_dict[name] = {
                "accuracy": acc_score,
                "y_train": train_test_dct["y_train_small"],
                "y_pred": y_pred,
                "y_pred_prob": y_pred_prob,
                "train_time": round(float(np.float32(train_time)),4),
                "eval_time": round(float(np.float32(eval_time)),4)
            }        
        else:
            pass
        
    logger.info(f'{"Algorithm":<34}{"Accuracy"}')   
    for name, pred in pred_dict.items():
        acc = pred["accuracy"]
        logger.info(f"{name:<33} {acc:.4f}")

    if DEBUG:
        return pred_dict, train_test_dct
    return pred_dict    