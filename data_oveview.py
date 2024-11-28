import numpy as np
import sktime
from sktime.datasets._single_problem_loaders import load_UCR_UEA_dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import site
import os
import pandas as pd
import sys


print(np.zeros(20))
print(sys.executable)

# Print paths to all site-packages directories
for path in site.getsitepackages():
    print(path)



from aeon.datasets import load_classification
X, y, meta_data = load_classification(name="ArrowHead", return_metadata=True) 
#"ElectricDevices" "AbnormalHeartbeat" "Adiac"
print(" Shape of X = ", X.shape)
print(" Meta data = ", meta_data)

X, y = load_UCR_UEA_dataset(name="Beef", split=None, return_X_y=True, return_type=None, extract_path=None)




a = 1671168
b = 170498071
print(a/b)