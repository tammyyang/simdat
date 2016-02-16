#!/usr/bin/env python3
import numpy as np
import logging
from simdat.core import ml

log_level = logging.DEBUG
#Train a model
data = np.array([[1, 2, 3, 4], [2, 2, 3, 4],
                 [3, 2, 3, 4], [4, 2, 3, 4],
                 [3, 2, 3, 4], [4, 2, 3, 4],
                 [3, 2, 3, 4], [4, 2, 3, 4],
                 [3, 2, 3, 4], [4, 2, 3, 4],
                 [3, 2, 3, 4], [4, 2, 3, 4],
                 [3, 2, 3, 4], [4, 2, 3, 4]])
target = np.array([1, 0,
                   0, 1,
                   0, 1,
                   1, 0,
                   1, 0,
                   0, 1,
                   1, 0])
a = ml.SVMRun()
results = a.run(data, target)

#Load the existing model and test
td = np.array([[1, 2, 3, 4], [2, 2, 3, 4],
               [3, 2, 3, 4], [4, 2, 3, 4]])
tt = np.array([1, 0, 1, 1])
model = a.read_model('./output/SVC.pkl')
a.test(td, tt, model)

#Predict
a.predict(td, model, outf='./output/result.json')
