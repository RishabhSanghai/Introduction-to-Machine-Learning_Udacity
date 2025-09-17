#!/usr/bin/python3

import sys
import os
from time import time

### -----------------------------------------------------------
### ROBUST PATH FIX - START
### -----------------------------------------------------------
try:
    script_dir = os.path.dirname(__file__)
    tools_path = os.path.abspath(os.path.join(script_dir, '..', 'tools'))
    sys.path.insert(0, tools_path)
except Exception:
    print("An error occurred in the path fix. Please check the file structure.")
### -----------------------------------------------------------
### ROBUST PATH FIX - END
### -----------------------------------------------------------

from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

accuracy = accuracy_score(labels_test, pred)
print("Accuracy:", accuracy)
##############################################################