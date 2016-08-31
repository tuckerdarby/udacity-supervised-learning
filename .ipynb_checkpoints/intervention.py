import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score


student_data = pd.read_csv("student-data.csv")
print student_data