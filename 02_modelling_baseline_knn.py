# Import libraries
import os
import numpy as np
import pandas as pd

# Import algorithms
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import cohen_kappa_score

# Set working directy, !set manually in top right corner
path = os.getcwd()
os.chdir(path) 

# Read csv and drop first to avoid reading index
train_x = pd.read_csv(filepath_or_buffer = "./generated_data/train/train_feat_l2.csv")
train_y = pd.read_csv(filepath_or_buffer = "./generated_data/train/train_y.csv")
val_x = pd.read_csv(filepath_or_buffer = "./generated_data/validation/val_feat_l2.csv")
val_y = pd.read_csv(filepath_or_buffer = "./generated_data/validation/val_y.csv")

# Create function that selects output variables
def get_y_var(df, a_list):
    return df[a_list]

# Get builing and floor labels into one data frame for both train and validation sets
train_y_clf = train_y.loc[:, ["buildingid", "floor"]]
val_y_clf = val_y.loc[:, ["buildingid", "floor"]]

# Initialize data frame and library that will be used 
accu = {}
kapp = {}
conf_matrix = {}
perf_clf = [accu, kapp, conf_matrix]
mse = {}
mea = {}
r_sqrd = {}
perf_reg = [mse, mea, r_sqrd]

# Prediction log
pred_log = pd.DataFrame()

# Select all variables that need prediction
pred_var = ["longitude", 
            "latitude", 
            "floor", 
            "buildingid"]

# Select all variables that need regression
reg_labels = ["longitude", 
              "latitude"]

# Loop over the variable you want to predict
for i in pred_var:
    # If i is label that requires regression, run regression else classification
    if i in reg_labels:
        # Select label you want to predict from train_y
        train_y_reg = get_y_var(train_y, i)
        # Select label you want to predict from validation
        val_y_reg = get_y_var(val_y, i)        
        # Initiate classifier
        knn = KNeighborsRegressor()
        # Fit model
        knn.fit(train_x, train_y_reg)
        # Predict values for validation data
        pred = knn.predict(val_x) 
        # Write prediction to pred_
        pred_log[i] = pred        
        # Estimate mse, mea, r^2 and write to library for relevant i
        mse[i] = mean_squared_error(val_y_reg, pred)
        mea[i] = mean_absolute_error(val_y_reg, pred)
        r_sqrd[i] = r2_score(val_y_reg, pred)

    else:
        # Select label you want to predict from train_y
        train_y_clf = get_y_var(train_y, i)
        # Select label you want to predict from validation
        val_y_clf = get_y_var(val_y, i)        
        # Initiate classifier
        knn = KNeighborsClassifier()
        # Fit model
        knn.fit(train_x, train_y_clf)
        # Predict values for validation data
        pred = knn.predict(val_x) 
        # Write prediction to pred_log
        pred_log[i] = pred        
        # Estimate mse, mea, r^2 and write to library for relevant i
        accu[i] = accuracy_score(val_y_clf, pred)
        kapp[i] = cohen_kappa_score(val_y_clf, pred)
        conf_matrix[i] = confusion_matrix(val_y_clf, pred,)
    

perf_reg = pd.DataFrame(perf_reg, index=["mse", "mea", "r_sqrd"])
perf_clf = pd.DataFrame(perf_clf, index=["accu", "kapp", "conf_matrix"])

#------------------------Calculate competition score--------------------------#

# Calculate the Euclidian distance between prediction and actual value
distance_squared = (val_y.loc[:, "latitude"] - pred_log.loc[:, "latitude"])**2 + (val_y.loc[:, "longitude"] - pred_log.loc[:, "longitude"])**2
distance_root = distance_squared.apply(lambda x: sqrt(x))

# Calculate penalyty for misclassifying floor
floor = abs(pred_log.loc[:, "floor"] - val_y.loc[:, "floor"]) * 4

# Calculate penalyty for misclassifying building
building = pred_log.loc[:, "buildingid"] != val_y.loc[:, "buildingid"]
building = building.apply(lambda x: int(x)) * 50

# Calculate final score using distance, floor, and building
distance_75 = np.percentile(distance_root + floor + building, 75)

